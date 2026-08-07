"""Parametric tube coefficients: a skeleton plus a Fourier cross-section profile.

Neurons are tube-like, so instead of storing an explicit mesh at several levels of
detail you can store a *parametric tube* over the skeleton and evaluate it at
whatever resolution is wanted. Per skeleton node this keeps not a scalar radius
but a cross-section profile ``r(theta)``, expanded as a truncated Fourier series
in the plane normal to the skeleton tangent::

    r(theta) = a0 + sum_k [ a_k cos(k theta) + b_k sin(k theta) ]

``a0`` alone is the classic SWC tube; ``k=1`` is the centroid offset (a diagnostic
that the skeleton is *not* centred); ``k=2`` is ellipticity - the first term
carrying real shape; ``k>=5`` is surface roughness and segmentation noise.

Coefficients are stored as magnitude/phase, ``m_k = hypot(a_k, b_k)`` and
``phi_k = atan2(b_k, a_k)``, because the rotation-minimizing frame is only defined
up to a constant rotation about the tangent: ``m_k`` is frame-independent and
``a_k``/``b_k`` individually are not. **Any truncation decision must be made on
``m_k``.**

Two consequences make this worth having. The representation is
``O(nodes x coefficients)`` rather than ``O(triangles)`` - 16 floats (64 bytes) a
node at ``K=4`` - and it has two orthogonal, independently truncatable LOD axes:
subsample nodes (axial) or lower ``K`` (angular). It can also be evaluated
directly in a vertex-pulling shader from ``TubeProfile.to_gpu_buffer()``, where
LOD selection is a uniform change rather than a buffer swap;
`TubeProfile.evaluate` is the CPU mirror of that loop.

Extraction runs off the **voxel mask**, not off a mesh: the mask is ground truth
and a mesh is already a lossy derivative. Each node fires ``n_theta`` rays in its
cross-section plane and records the first exit from the object, walked with a
vectorised Amanatides-Woo DDA over the sparse voxel set (`_raycast`). Exit
distances are the exact parametric crossing of a cell face, so radii are
sub-voxel - better than the nearest-voxel-centre estimate `skeleton._radii` and
`measure.distance_transform` give. Spurious handles and merge artifacts also stop
mattering, since only the first exit is taken.

Known limits, all flagged per node rather than papered over (see `TubeProfile`):
frames do not propagate consistently through a bifurcation, and ``r(theta)``
requires each cross-section to be star-shaped about its skeleton point - violated
by spines, self-touching neurites and somata. ``diagnostics=True`` measures that
second failure rate directly by continuing each ray past its first exit.

References
----------
Wang W, Juttler B, Zheng D, Liu Y. Computation of rotation minimizing frames.
ACM Trans. Graph. 2008;27(1):1-18.
"""

from dataclasses import dataclass

import numpy as np

from ._keys import as_spacing, sorted_hit, to_keys, validate
from ._sparse import sparse_aware
from .core import log, pack, unique
from .skeleton import Skeleton

# Same soft-dependency shape as `teasar`/`wavefront`: the library is a hard
# dependency of the package, but the backend is still selected through a
# module-level name so `tests/test_tube.py::test_backends_agree` can null it out
# and pin the pure-numpy path against it.
try:
    import dijkstra3d_sparse as _dijkstra3d_sparse
except ModuleNotFoundError:  # pragma: no cover - exercised only without the library
    _dijkstra3d_sparse = None

__all__ = ["TubeProfile", "tube_profile", "tube_coefficients"]

# Per-node flags. Bits rather than separate arrays so the failure modes stay
# queryable next to the coefficients they qualify, at one byte a node.
FLAG_JUNCTION = 1 << 0  # at, or adjacent to, a degree>=3 node
FLAG_NON_STAR = 1 << 1  # some ray re-entered the object after its first exit
FLAG_SEED_OUTSIDE = 1 << 2  # the node's own voxel is not in the object
FLAG_RAY_ESCAPED = 1 << 3  # some ray reached `max_radius` without exiting
FLAG_BRANCH_END = 1 << 4  # first or last node of a branch (one-sided tangent)

_FLAG_NAMES = {
    "junction": FLAG_JUNCTION,
    "non_star": FLAG_NON_STAR,
    "seed_outside": FLAG_SEED_OUTSIDE,
    "ray_escaped": FLAG_RAY_ESCAPED,
    "branch_end": FLAG_BRANCH_END,
}

# Rays marched at once. Ray state is ~70 bytes (cur, tmax, tdelta, step, ids,
# phase) plus per-step temporaries, so this caps the raycaster near 100 MB
# whatever the node count - the same reason `teasar._NB_CHUNK` exists.
_RAY_BLOCK = 1_000_000

# Default ceiling on how far a ray may travel, as a multiple of the node's own
# radius estimate. Without a ceiling a ray launched nearly tangent to the surface
# runs the length of the neurite; 4x is loose enough not to clip real geometry.
_MAX_RADIUS_FACTOR = 4.0

# Fallback ceiling when the skeleton carries no radii at all, in voxels.
_FALLBACK_RADIUS = 64.0


# ---------------------------------------------------------------------------
# skeleton topology
# ---------------------------------------------------------------------------


def _adjacency(edges, m):
    """CSR-style neighbour lists: ``(adj, start, degree)``.

    ``adj[start[u]:start[u + 1]]`` are the neighbours of node ``u``. Built with one
    argsort over the symmetrised edge list rather than a list of lists, because the
    skeleton of a real neurite runs to ~10^5 nodes.
    """
    degree = (
        np.bincount(edges.ravel(), minlength=m).astype(np.int64)
        if len(edges)
        else np.zeros(m, dtype=np.int64)
    )
    start = np.concatenate([[0], np.cumsum(degree)])
    if len(edges) == 0:
        return np.empty(0, dtype=np.int64), start, degree
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    adj = dst[np.argsort(src, kind="stable")].astype(np.int64, copy=False)
    return adj, start, degree


def _branches(edges, m):
    """Split the skeleton into maximal degree-2 paths.

    Returns ``(flat, starts, lengths)`` describing the branches back to back in
    ``flat``, sorted by length descending. Each branch *includes* its two
    breakpoint endpoints, because the tangent and the frame at a junction still
    have to come from somewhere; a junction therefore appears in every branch
    incident to it, and the last one processed owns its frame. That ambiguity is
    exactly what `FLAG_JUNCTION` marks - the junction region is flagged here, not
    solved.

    Nodes of degree 0 come out as their own length-1 branch. A pure cycle has no
    breakpoint to start from, so it is walked from an arbitrary node and closed
    back onto it.

    The descending sort is not cosmetic: it makes the branches still alive at step
    ``k`` a *prefix* of the branch list, so `_frames` can step every branch forward
    in lockstep with one contiguous slice instead of a gather.
    """
    adj, start, degree = _adjacency(edges, m)
    # The walk is per-node Python; on lists it costs ~0.1 us a step against ~1 us
    # for a numpy slice, which is the difference between negligible and noticeable
    # on a 10^5-node centerline.
    adj_l, start_l, deg_l = adj.tolist(), start.tolist(), degree.tolist()
    seen_half = set()  # directed (a, b) half-edges, packed as a * m + b
    paths = []

    def walk(u, first):
        """Follow the degree-2 chain leaving `u` towards `first`."""
        path = [u, first]
        prev, cur = u, first
        while deg_l[cur] == 2:
            a, b = adj_l[start_l[cur]], adj_l[start_l[cur] + 1]
            nxt = a if a != prev else b
            path.append(nxt)
            if nxt == path[0]:  # closed a loop back onto the seed
                break
            prev, cur = cur, nxt
        return path

    for u in np.flatnonzero(degree != 2).tolist():
        if deg_l[u] == 0:
            paths.append([u])
            continue
        for v in adj_l[start_l[u] : start_l[u + 1]]:
            if u * m + v in seen_half:
                continue
            path = walk(u, v)
            for a, b in zip(path[:-1], path[1:]):
                seen_half.add(a * m + b)
                seen_half.add(b * m + a)
            paths.append(path)

    # Anything left is a pure cycle: every node degree 2, so no breakpoint seeded a
    # walk into it.
    seen = np.zeros(m, dtype=bool)
    for p in paths:
        seen[p] = True
    for u in np.flatnonzero(~seen & (degree == 2)).tolist():
        if seen[u]:
            continue
        path = walk(u, adj_l[start_l[u]])
        seen[path] = True
        paths.append(path)

    if not paths:
        z = np.empty(0, dtype=np.int64)
        return z, z, z

    lengths = np.array([len(p) for p in paths], dtype=np.int64)
    order = np.argsort(-lengths, kind="stable")
    lengths = lengths[order]
    flat = np.concatenate([np.asarray(paths[i], dtype=np.int64) for i in order])
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return flat, starts, lengths


def _neighbour_slots(flat, starts, lengths, m):
    """Per-node branch predecessor/successor, branch id and branch-end mask.

    Written through `flat`, so a node shared by several branches takes the last
    writer's slots - the same last-writer rule `_branches` documents for frames,
    which keeps `branch`, `prev` and `next` mutually consistent.
    """
    total = len(flat)
    prev_of = np.arange(m, dtype=np.int64)
    next_of = np.arange(m, dtype=np.int64)
    branch = np.zeros(m, dtype=np.int32)
    is_end = np.zeros(m, dtype=bool)
    if total == 0:
        return prev_of, next_of, branch, is_end

    at_start = np.zeros(total, dtype=bool)
    at_start[starts] = True
    at_end = np.zeros(total, dtype=bool)
    at_end[starts + lengths - 1] = True

    prev_flat = np.empty(total, dtype=np.int64)
    prev_flat[1:] = flat[:-1]
    prev_flat[0] = flat[0]
    prev_flat[at_start] = flat[at_start]

    next_flat = np.empty(total, dtype=np.int64)
    next_flat[:-1] = flat[1:]
    next_flat[-1] = flat[-1]
    next_flat[at_end] = flat[at_end]

    prev_of[flat] = prev_flat
    next_of[flat] = next_flat
    branch[flat] = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
    is_end[flat[at_start | at_end]] = True
    return prev_of, next_of, branch, is_end


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def _normalize(v):
    """Row-wise unit vectors; rows of length ~0 come back as ``(0, 0, 0)``."""
    n = np.linalg.norm(v, axis=1)
    ok = n > 1e-12
    out = np.zeros_like(v)
    out[ok] = v[ok] / n[ok, None]
    return out


def _seed_frame(t):
    """An arbitrary but well-conditioned ``(u, v)`` orthonormal to each tangent.

    Crossing with the world axis *least* aligned with the tangent is what keeps it
    well-conditioned - crossing with a fixed axis degenerates whenever the tangent
    happens to run along it. The resulting constant rotation about the tangent is
    arbitrary and differs between branches, which is harmless: every stored
    magnitude ``m_k`` is frame-independent.
    """
    e = np.zeros_like(t)
    e[np.arange(len(t)), np.argmin(np.abs(t), axis=1)] = 1.0
    u = _normalize(np.cross(t, e))
    return u, np.cross(t, u)


def _reflect(a, n):
    """Reflect rows of `a` in the plane through the origin normal to rows of `n`."""
    nn = np.einsum("ij,ij->i", n, n)
    ok = nn > 1e-24
    scale = np.zeros(len(a))
    scale[ok] = 2.0 * np.einsum("ij,ij->i", n[ok], a[ok]) / nn[ok]
    return a - scale[:, None] * n


def _orthonormalize(u, t):
    """Project `u` off `t` and renormalise, re-seeding rows that collapsed.

    The double reflection is exact in theory; over a 10^3-node branch the residual
    drift is not, so this runs every step. A row that collapses entirely (a
    degenerate tangent, or a 180-degree turn) falls back to a fresh seed rather
    than to a vector that is no longer perpendicular.
    """
    u = _normalize(u - np.einsum("ij,ij->i", u, t)[:, None] * t)
    bad = np.linalg.norm(u, axis=1) < 0.5
    if bad.any():
        u[bad] = _seed_frame(t[bad])[0]
    return u


def _frames(verts, tangent, flat, starts, lengths):
    """Rotation-minimizing frames, propagated by double reflection along branches.

    Frenet frames are deliberately not used: they flip at inflection points and are
    undefined on straight segments, which is most of a neurite. The double
    reflection method (Wang et al. 2008) is exact to second order and needs no
    curvature.

    Propagation is serial *within* a branch and independent *across* them, so this
    steps every branch forward together: because `_branches` sorts by length
    descending, the branches still alive at step ``k`` are the first ``cnt`` of
    them and one slice picks them all out. Total work is O(nodes), spread over
    ``max(lengths)`` vectorised steps.
    """
    m = len(verts)
    u = np.zeros((m, 3))
    v = np.zeros((m, 3))
    if len(flat) == 0:
        return u, v

    seed = flat[starts]
    u[seed], v[seed] = _seed_frame(tangent[seed])

    neg = -lengths  # ascending, so searchsorted counts the branches still alive
    for k in range(1, int(lengths[0])):
        cnt = int(np.searchsorted(neg, -(k + 1), side="right"))
        if cnt == 0:
            break
        cur = flat[starts[:cnt] + k]
        prv = flat[starts[:cnt] + k - 1]

        v1 = verts[cur] - verts[prv]
        u_l = _reflect(u[prv], v1)
        t_l = _reflect(tangent[prv], v1)
        u_n = _reflect(u_l, tangent[cur] - t_l)

        u[cur] = _orthonormalize(u_n, tangent[cur])
        v[cur] = np.cross(tangent[cur], u[cur])
    return u, v


def _frame_to_quat(u, v, t):
    """Quaternion ``(x, y, z, w)`` of the rotation whose columns are ``(u, v, t)``.

    Shepperd's method - take the branch with the largest denominator, so the square
    root never runs into cancellation. Canonicalised to ``w >= 0``, so the two
    representations of one rotation cannot alternate along a branch and wreck the
    delta coding in `TubeProfile.save_npz`.
    """
    q = np.zeros((len(u), 4))
    xx, yy, zz = u[:, 0], v[:, 1], t[:, 2]
    trace = xx + yy + zz
    case = np.where(trace > 0, 0, 1 + np.argmax(np.stack([xx, yy, zz], axis=1), axis=1))

    def fill(sel, s, qx, qy, qz, qw):
        q[sel, 0], q[sel, 1], q[sel, 2], q[sel, 3] = qx / s, qy / s, qz / s, qw / s

    # Column-major frame, so the rotation matrix is m[i][j] = (u, v, t)[j][i]:
    # m01 = v[0], m10 = u[1], m02 = t[0], m20 = u[2], m12 = t[1], m21 = v[2].
    sel = case == 0
    if sel.any():
        s = 2.0 * np.sqrt(1.0 + trace[sel])
        fill(sel, s, v[sel, 2] - t[sel, 1], t[sel, 0] - u[sel, 2],
             u[sel, 1] - v[sel, 0], 0.25 * s * s)
    sel = case == 1
    if sel.any():
        s = 2.0 * np.sqrt(1.0 + xx[sel] - yy[sel] - zz[sel])
        fill(sel, s, 0.25 * s * s, v[sel, 0] + u[sel, 1],
             t[sel, 0] + u[sel, 2], v[sel, 2] - t[sel, 1])
    sel = case == 2
    if sel.any():
        s = 2.0 * np.sqrt(1.0 + yy[sel] - xx[sel] - zz[sel])
        fill(sel, s, v[sel, 0] + u[sel, 1], 0.25 * s * s,
             t[sel, 1] + v[sel, 2], t[sel, 0] - u[sel, 2])
    sel = case == 3
    if sel.any():
        s = 2.0 * np.sqrt(1.0 + zz[sel] - xx[sel] - yy[sel])
        fill(sel, s, t[sel, 0] + u[sel, 2], t[sel, 1] + v[sel, 2],
             0.25 * s * s, u[sel, 1] - v[sel, 0])

    q[q[:, 3] < 0] *= -1.0
    return q


def _quat_to_frame(q):
    """Inverse of `_frame_to_quat`: the rotation's ``(u, v, t)`` columns."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    u = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], 1)
    v = np.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], 1)
    t = np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], 1)
    return u, v, t


# ---------------------------------------------------------------------------
# membership backends
# ---------------------------------------------------------------------------


def _graph(voxels):
    """A `dijkstra3d_sparse.Graph` over `voxels`, deduplicating only if forced to.

    `Graph` rejects duplicate coordinates, and the obvious response - dedup up
    front - costs 4.3 s on the 5.5 M-voxel arbor against 0.07 s to build the
    index, because `unique(axis=0)` sorts 3 columns of 5.5 M rows. A sparse-cubes
    voxel set is unique by construction, so that was four times the raycast spent
    proving there was nothing to remove. Ask forgiveness instead: the rare caller
    with duplicates pays, and nobody else does.
    """
    try:
        return _dijkstra3d_sparse.Graph(
            np.ascontiguousarray(voxels, dtype=np.int32), index_kind="hash"
        )
    except ValueError:
        clean = unique(np.asarray(voxels).astype(np.int64), axis=0)
        return _dijkstra3d_sparse.Graph(
            np.ascontiguousarray(clean, dtype=np.int32), index_kind="hash"
        )


class _SparseMembership:
    """Preferred backend: one `dijkstra3d_sparse.Graph` hash index, probed per step.

    The index is built once for the whole extraction and reused by every chunk and
    every DDA step - which is the entire reason to prefer it, since a ray walk is
    ~10^8 membership probes on a real neurite. `index_of` is O(1) a probe against
    `sorted_hit`'s O(log N), and releases the GIL.
    """

    name = "sparse"

    def __init__(self, voxels):
        self._g = _graph(voxels)

    def __call__(self, coords):
        return self._g.index_of(coords, strict=False) >= 0


class _KeyMembership:
    """Fallback backend: `searchsorted` over the packed int64 keys.

    A ray leaves the bounding box long before it stops being marched, and a
    coordinate outside the box has no valid packed key - `pack`'s fields would
    borrow into each other and alias onto a genuine voxel. The bounds test is
    therefore a correctness requirement, not an optimisation; the clip exists only
    to keep `pack` in range for rows that test has already rejected.
    """

    name = "keys"

    def __init__(self, voxels):
        self._keys, self._shift = to_keys(voxels, margin=0)
        # `max` first, then widen: the other order materialises a full (N, 3) int64
        # copy - 133 MB on a 5.5M-voxel arbor - to produce three numbers. A per-axis
        # shift is a constant, and subtraction is monotone, so the answer is equal.
        self._hi = voxels.max(axis=0).astype(np.int64) - self._shift

    def __call__(self, coords):
        v = coords.astype(np.int64) - self._shift
        inside_box = np.all((v >= 0) & (v <= self._hi), axis=1)
        np.clip(v, 0, self._hi, out=v)
        return inside_box & sorted_hit(pack(v), self._keys)


# ---------------------------------------------------------------------------
# raycasting backends
# ---------------------------------------------------------------------------

_BACKENDS = ("exits", "sparse", "keys")


class _ExitsCaster:
    """Preferred: the whole DDA in Rust, via `dijkstra3d_sparse.Graph.ray_exits`.

    `_raycast` steps every live ray in lockstep, so the loop runs as long as the
    single longest ray while the median finishes in a fraction of that - on the
    neuron fixture two thirds of the iterations carry under 1% of the probes. A
    scalar loop per ray has no such tail, and a bulk `index_of` measured the probe
    floor at 22 ns against the numpy walk's 305 ns, so nearly all of that gap is
    bookkeeping rather than lookups. This backend hands the whole walk over.

    The `Graph` is built once and reused by every chunk, and doubles as the
    membership probe for the seed test - one index, both questions.
    """

    name = "exits"

    def __init__(self, voxels):
        # The same probe object the numpy walk would use: one index answers both
        # "does this ray leave here" and the seed test, and the membership
        # semantics are stated in exactly one place.
        self.inside = _SparseMembership(voxels)
        self._g = self.inside._g

    def cast(self, origins, dirs, caps, diagnostics, star_window):
        t, n_hits = self._g.ray_exits(
            origins, dirs, max_dist=caps, max_crossings=2 if diagnostics else 1
        )
        radius = t[:, 0].copy()
        escaped = n_hits == 0
        radius[escaped] = caps[escaped]

        if not diagnostics:
            return radius, escaped, np.zeros(len(origins), dtype=bool)
        # `star_window` stays here rather than becoming a second, tighter-capped
        # call: `t[:, 1]` is the *first* re-entry, so testing it against the
        # deadline afterwards gives the same answer as refusing to look past the
        # deadline, for one walk instead of two. See `_raycast` for why the
        # re-entry window must not simply be the escape cap.
        return radius, escaped, (n_hits > 1) & (t[:, 1] <= (1.0 + star_window) * radius)


class _WalkCaster:
    """Fallback: the vectorised numpy DDA in `_raycast` over a membership probe."""

    def __init__(self, probe):
        self.inside = probe
        self.name = probe.name

    def cast(self, origins, dirs, caps, diagnostics, star_window):
        return _raycast(origins, dirs, caps, self.inside, diagnostics, star_window)


def _caster(voxels, backend):
    """Pick a raycasting backend; `backend` forces one, for the parity tests.

    Resolved in one place. Splitting it across two helpers - one choosing Rust vs
    numpy, another choosing the numpy probe - is how ``"exits"`` once fell through
    to the numpy walk unnoticed, since the parity test was then comparing an
    implementation against itself.

    All three produce bit-identical output - that is what ``test_backends_agree``
    pins - so this is a performance knob only.
    """
    if backend is not None and backend not in _BACKENDS:
        raise ValueError(f"backend must be None or one of {_BACKENDS}, got {backend!r}")
    # `_dijkstra3d_sparse is None` is not a version fallback - the library is a hard
    # dependency. The arm exists because `test_backends_agree` nulls the module out
    # to run the numpy walk as an independent oracle.
    have_lib = _dijkstra3d_sparse is not None
    if backend is None:
        backend = "exits" if have_lib else "keys"
    if backend != "keys" and not have_lib:
        raise RuntimeError(f"backend={backend!r} needs dijkstra3d_sparse installed.")
    if backend == "exits":
        return _ExitsCaster(voxels)
    probe = _SparseMembership if backend == "sparse" else _KeyMembership
    return _WalkCaster(probe(voxels))


# ---------------------------------------------------------------------------
# raycasting
# ---------------------------------------------------------------------------


def _raycast(origins, dirs, caps, inside, diagnostics, star_window):
    """First exit of each ray from the voxel set, by vectorised Amanatides-Woo.

    Voxel centres sit on integer coordinates, so cell ``c`` spans ``[c-0.5, c+0.5)``
    and the ray's next face crossing along axis ``a`` is at ``cur + step*0.5``.
    Stepping whichever axis has the smallest ``tmax`` visits every cell the ray
    passes through, in order, and the ``tmax`` at which it steps *out* of the object
    is the exact parametric exit - a sub-voxel radius, not a voxel count.

    `origins` and `dirs` are in index space; the caller has already scaled `dirs` by
    ``1/spacing`` from a physically unit direction, so the returned parameter is a
    physical distance.

    Rays live in one compacted block and are dropped as they finish, so the working
    set shrinks to the few long rays instead of staying at full width. A ray that
    reaches its cap without exiting reports the cap and is flagged escaped.

    With `diagnostics` a ray keeps marching past its first exit and reports whether
    it re-enters - the star-shapedness test. That search gets its own deadline,
    ``(1 + star_window) * t_exit``, rather than running to `caps`: the two questions
    are unrelated, and sharing one window makes the answer meaningless. A ray that
    exits a 100 nm neurite and re-enters 40 nm later has found the same
    cross-section folding back on itself; one that re-enters 400 nm later has found
    a *different* neurite passing nearby, which says nothing about whether this
    cross-section is star-shaped. Measured on the neuron fixture the difference is
    not academic - the same object reports 31% or 69% of nodes non-star depending
    purely on how far the ray is allowed to keep looking.

    Returns ``(radius, escaped, non_star)``, each aligned with `origins`.
    """
    n = len(origins)
    radius = np.zeros(n)
    escaped = np.zeros(n, dtype=bool)
    non_star = np.zeros(n, dtype=bool)
    if n == 0:
        return radius, escaped, non_star

    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.sign(dirs).astype(np.int8)
        cur = np.rint(origins).astype(np.int32)
        tdelta = np.abs(1.0 / dirs)
        tmax = (cur + step * 0.5 - origins) / dirs
    # A zero direction component never crosses that axis at all.
    tmax[~np.isfinite(tmax)] = np.inf
    tdelta[~np.isfinite(tdelta)] = np.inf
    np.maximum(tmax, 0.0, out=tmax)  # an origin exactly on a face must not step back

    # Copied, because the re-entry deadline is written back into it per ray.
    caps = np.array(caps, dtype=float, copy=True)
    ids = np.arange(n, dtype=np.int64)
    # phase 0: inside, hunting the first exit. phase 1: outside, watching for a
    # re-entry - only ever reached when `diagnostics` is on.
    phase = np.zeros(n, dtype=np.int8)

    while len(ids):
        rows = np.arange(len(ids))
        axis = np.argmin(tmax, axis=1)
        t_cross = tmax[rows, axis]

        # `~(t <= cap)` rather than `t > cap` so an all-infinite tmax (a direction
        # with no crossing anywhere) retires instead of looping forever.
        over = ~(t_cross <= caps[ids])
        cur[rows, axis] += step[rows, axis]
        tmax[rows, axis] += tdelta[rows, axis]

        live = ~over
        occupied = np.zeros(len(ids), dtype=bool)
        if live.any():
            occupied[live] = inside(cur[live])

        hunting = phase == 0
        exited = hunting & live & ~occupied
        radius[ids[exited]] = t_cross[exited]
        phase[exited] = 1
        if diagnostics and exited.any():
            # Hand the ray a *new*, tighter deadline for the re-entry search, so
            # `over` retires it there instead of at the escape cap.
            out = ids[exited]
            caps[out] = np.minimum(caps[out], (1.0 + star_window) * t_cross[exited])

        reentered = (~hunting) & live & occupied
        non_star[ids[reentered]] = True

        capped = over & hunting
        radius[ids[capped]] = caps[ids[capped]]
        escaped[ids[capped]] = True

        retire = over | reentered
        if not diagnostics:
            retire |= exited
        if retire.any():
            keep = ~retire
            ids, phase = ids[keep], phase[keep]
            cur, tmax, tdelta, step = cur[keep], tmax[keep], tdelta[keep], step[keep]
    return radius, escaped, non_star


def _fourier(radii, K):
    """rFFT of the sampled ``r(theta)`` into ``(a0, mag, phase, residual)``.

    The angles are a uniform grid, so this is an exact discrete expansion, not a
    least-squares fit. `residual` is the RMS of everything discarded, from Parseval
    rather than by reconstructing: the bin weights are 1 on DC and Nyquist and 2 in
    between, which is why `tube_profile` refuses ``K`` at or above Nyquist and so
    keeps one convention for every stored harmonic.
    """
    n = radii.shape[1]
    c = np.fft.rfft(radii, axis=1)
    c /= n  # in place: `rfft` hands back a fresh array, and it is (M, n/2+1) complex
    a0 = c[:, 0].real
    a_k = 2.0 * c[:, 1 : K + 1].real
    b_k = -2.0 * c[:, 1 : K + 1].imag

    w = np.full(c.shape[1], 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    # |c|^2 directly rather than abs(c)**2: the latter takes a square root (with
    # overflow guards) only to square it back.
    drop = c[:, K + 1 :]
    tail = (drop.real**2 + drop.imag**2) @ w[K + 1 :]
    return a0, np.hypot(a_k, b_k), np.arctan2(b_k, a_k), np.sqrt(np.maximum(tail, 0.0))


# ---------------------------------------------------------------------------
# the profile
# ---------------------------------------------------------------------------


def _delta_encode(x, new_run):
    """Difference consecutive rows, restarting at every ``new_run`` row."""
    d = np.array(x, dtype=np.float32, copy=True)
    if len(d) > 1:
        keep = new_run[1:] if d.ndim == 1 else new_run[1:, None]
        d[1:] = np.where(keep, d[1:], d[1:] - np.asarray(x, dtype=np.float32)[:-1])
    return d


def _delta_decode(d, new_run):
    """Inverse of `_delta_encode`: a prefix sum restarted at every run.

    One global `cumsum` minus the running total at each run's start, rather than a
    Python loop over runs - a real arbor has thousands of branches.
    """
    out = np.cumsum(np.asarray(d, dtype=np.float64), axis=0)
    if len(out) == 0:
        return out.astype(np.float32)
    starts = np.flatnonzero(new_run)
    base = np.zeros((len(starts),) + out.shape[1:], dtype=out.dtype)
    base[1:] = out[starts[1:] - 1]
    return (out - base[np.cumsum(new_run) - 1]).astype(np.float32)


@dataclass
class TubeProfile:
    """A skeleton plus a truncated Fourier cross-section profile per node.

    Struct-of-arrays throughout, float32 where the precision budget allows, laid
    out so `to_gpu_buffer` is a copy rather than a repack.

    Attributes
    ----------
    nodes :     (M, 3) float32 node coordinates in *index* space, with `spacing`
                held separately - the same contract as `Skeleton`, so `vertices`
                gives the physical position.
    edges :     (E, 2) int32 index pairs, carried through from the skeleton.
    a0 :        (M,) float32 mean radius, in **physical units** (like
                `Skeleton.radii`, and unlike `nodes`).
    mag :       (M, K) float32 magnitudes ``m_1..m_K``. Frame-independent - this is
                what a truncation threshold is applied to.
    phase :     (M, K) float32 phases ``phi_1..phi_K``, radians. Frame-dependent
                and smooth along a branch, hence delta-coded on save.
    frame :     (M, 4) float32 quaternion ``(x, y, z, w)`` of the rotation whose
                columns are ``(u, v, tangent)``. Four floats rather than six.
    branch :    (M,) int32 branch id. Marks where delta coding must restart.
    flags :     (M,) uint8 bitfield; see `FLAG_JUNCTION` and friends, and `flag`.
    residual :  (M,) float32 RMS of the discarded harmonics, physical units - the
                per-node error budget of having truncated at ``K``.
    non_star :  (M,) float32 *fraction* of this node's rays that re-entered the
                object, i.e. how badly the cross-section fails to be star-shaped.
                All zero unless the profile was built with ``diagnostics=True``.
                `FLAG_NON_STAR` is just ``non_star > 0``; the fraction is what makes
                that bit actionable, since one grazing ray out of 64 and half the
                ring folding back are very different situations.
    spacing :   (3,) float or None.
    n_theta :   int, how many rays were cast per node.
    """

    nodes: np.ndarray
    edges: np.ndarray
    a0: np.ndarray
    mag: np.ndarray
    phase: np.ndarray
    frame: np.ndarray
    branch: np.ndarray
    flags: np.ndarray
    residual: np.ndarray
    non_star: np.ndarray = None
    spacing: "np.ndarray | None" = None
    n_theta: int = 0

    def __post_init__(self):
        if self.non_star is None:
            self.non_star = np.zeros(len(self.nodes), dtype=np.float32)

    @property
    def K(self):
        """Angular truncation order - the number of stored harmonics."""
        return self.mag.shape[1]

    @property
    def vertices(self):
        """Node coordinates as float, scaled by `spacing` when set."""
        v = self.nodes.astype(float)
        return v * self.spacing if self.spacing is not None else v

    def flag(self, name):
        """Boolean mask of one named flag, e.g. ``profile.flag("non_star")``."""
        if name not in _FLAG_NAMES:
            raise ValueError(f"unknown flag {name!r}; choose from {sorted(_FLAG_NAMES)}")
        return (self.flags & _FLAG_NAMES[name]) != 0

    def frame_vectors(self):
        """Unpack `frame` back into ``(u, v, tangent)``, each ``(M, 3)`` float."""
        return _quat_to_frame(self.frame.astype(float))

    def coefficients(self):
        """The Cartesian view ``(a, b)``, each ``(M, K)``.

        Provided for reconstruction and for checking against a reference
        implementation. Do **not** threshold on these individually: the frame is
        arbitrary up to a rotation about the tangent, which mixes ``a_k`` and
        ``b_k`` into each other. `mag` is the frame-independent quantity.
        """
        m, p = self.mag.astype(float), self.phase.astype(float)
        return m * np.cos(p), m * np.sin(p)

    def radius_at(self, theta, k=None, derivative=False):
        """Evaluate ``r(theta)`` per node, physical units; returns ``(M, T)``.

        `theta` is a scalar or ``(T,)`` array of angles. `k` truncates further than
        the stored ``K`` - the angular LOD axis. Accumulated one harmonic at a time
        so the working set stays ``(M, T)`` rather than ``(M, K, T)``.

        Evaluated in the Cartesian form ``a_k cos(k theta) + b_k sin(k theta)``
        rather than as ``m_k cos(k theta - phi_k)``: the latter puts the phase
        inside the cosine, forcing an ``(M, T)`` pair of transcendentals per
        harmonic, where this needs only ``(T,)`` of them plus the ``(M, K)``
        conversion `coefficients` already does. At 70k nodes and ``n_theta=64``
        that is 38M ``cos``/``sin`` calls against 0.6M - and it is the identity the
        shader relies on for the same reason (see `to_gpu_buffer`).

        With ``derivative=True`` returns ``(r, dr_dtheta)``, the same pair the
        shader's profile evaluation returns from one loop; `evaluate` needs the
        slope for the surface normal.
        """
        kk = self.K if k is None else int(k)
        if not 0 <= kk <= self.K:
            raise ValueError(f"k must be in [0, {self.K}], got {k!r}")
        th = np.atleast_1d(np.asarray(theta, dtype=float))
        r = np.repeat(self.a0.astype(float)[:, None], len(th), axis=1)
        dr = np.zeros_like(r) if derivative else None
        if kk:
            a, b = self.coefficients()
        for j in range(kk):
            ck, sk = np.cos((j + 1) * th)[None, :], np.sin((j + 1) * th)[None, :]
            r += a[:, j : j + 1] * ck + b[:, j : j + 1] * sk
            if derivative:
                # d/dtheta [a cos(k t) + b sin(k t)] = k (b cos(k t) - a sin(k t))
                dr += (j + 1) * (b[:, j : j + 1] * ck - a[:, j : j + 1] * sk)
        return (r, dr) if derivative else r

    def evaluate(self, n_theta=None, k=None, nodes=None, return_normals=False,
                 k_normal=None):
        """Ring points around each node, ``(M', n_theta, 3)``, in physical space.

        The CPU mirror of the vertex-pulling shader: same loop, same two LOD knobs.
        `n_theta` sets the angular sampling (independent of how many rays were
        cast), `k` the angular truncation, and `nodes` selects a subset - subsample
        it for the axial axis.

        Parameters
        ----------
        return_normals :    If set, return ``(points, normals)`` rather than just
                            points; `normals` has the same shape, is unit length
                            and points outward. See the notes - the definition is
                            not quite the one a per-edge shader arrives at.
        k_normal :          Truncation to build the *normal* from, defaulting to
                            `k` - i.e. the exact normal of the surface being
                            returned. Set it lower to shade a detailed silhouette
                            with a smoother normal; see the notes for why that is
                            usually what you want on real data.

        Notes
        -----
        The normal is ``normalize(cross(dp/dtheta, dp/ds))`` flipped to agree with
        the radial direction. ``dp/dtheta`` is analytic (that is what
        ``radius_at(derivative=True)`` is for); ``dp/ds`` is a **centred**
        difference of the surface point at the *same* ``theta`` between the node's
        branch predecessor and successor, one-sided at branch ends. Differencing
        the surface rather than using the stored tangent is what makes a bouton
        shade correctly: the radius changing along the axis tilts the surface even
        where the centreline is straight.

        **The exact normal is noisy on real neurites, and `k_normal` is the
        answer.** ``dr/dtheta`` weights harmonic ``k`` by ``k``, so once the
        ``m_k`` flatten out at the rasterization floor every further harmonic adds
        more slope than shape. On a 16 nm arbor the normal's median tilt away from
        radial runs 24 deg at ``k=1`` and 35 deg at ``k=4`` while the *silhouette*
        keeps improving - the surface really is that bumpy at a 5-voxel radius, so
        this is honest geometry rather than a bug, but it shades like sandpaper.
        ``evaluate(k=4, k_normal=1)`` keeps the detailed outline and drops the
        shading noise. ``k_normal=0`` is the floor and still leaves 13 deg, which
        is the axial term and is mostly real: smoothing `a0` three times along the
        branch only takes it to 10 deg, so there is no point going after it.

        Two more consequences, before differencing this against a shader:

        - A shader that pulls one quad per edge only has that edge's two ends, so
          it gets the *one-sided* normal, which differs from this one by
          O(curvature x node spacing) and is discontinuous between the two quads
          meeting at a ring. That faceting is the expected disagreement; a
          direction mismatch bigger than the local turn angle is a real bug.
        - Frames are seeded per branch, so ``theta`` means different things on
          either side of a junction and the axial difference there is meaningless.
          Those nodes carry `FLAG_JUNCTION`; mask on it rather than trusting the
          normal.

        Points and normals are each ``M' x n_theta x 3`` float64, and the normal
        path peaks at about **eight** arrays that size - it needs the radial and
        tangential bases, the two neighbour rings the axial difference is taken
        from, and the cross product. Measured, not estimated. So size a node subset
        from that figure rather than from the two arrays you get back: 40k nodes at
        ``n_theta=64`` peaks near 510 MB.
        """
        n_th = int(self.n_theta if n_theta is None else n_theta)
        if n_th < 1:
            raise ValueError(f"n_theta must be >= 1, got {n_theta!r}")
        th = 2.0 * np.pi * np.arange(n_th) / n_th

        sel = slice(None) if nodes is None else np.asarray(nodes)
        sub = self if nodes is None else self._subset(sel)
        e_r = sub._radial(th)
        if not return_normals:
            return sub._ring(th, k, e_r)

        # The points describe the `k` surface, the normal the `kn` one; `e_r` has no
        # `k` dependence, so it is built once and serves both.
        kn = k if k_normal is None else k_normal
        pts = sub._ring(th, k, e_r)
        r_n, dr_n = sub.radius_at(th, k=kn, derivative=True)
        dp_dth = sub._tangential(th)
        dp_dth *= r_n[:, :, None]
        dp_dth += e_r * dr_n[:, :, None]

        prev_of, next_of = self._axial_neighbours()
        dp_ds = self._subset(next_of[sel])._ring(th, kn)
        dp_ds -= self._subset(prev_of[sel])._ring(th, kn)

        nrm = np.cross(dp_dth, dp_ds)
        length = np.linalg.norm(nrm, axis=-1)
        # Exactly zero where the axial difference vanishes (an isolated node, or a
        # node whose neighbours coincide) or where the radius does (a0 = 0). Both
        # leave the radial direction as the only sensible answer, and it is the
        # right one for a circular tube.
        flat = length == 0.0
        if flat.any():
            nrm[flat] = e_r[flat]
            length[flat] = 1.0
        nrm /= length[:, :, None]
        outward = np.einsum("ijk,ijk->ij", nrm, e_r) < 0.0
        nrm[outward] *= -1.0
        return pts, nrm

    def _radial(self, th):
        """``e_r``, the radial unit direction per node and angle, ``(M, T, 3)``."""
        u, v, _ = self.frame_vectors()
        e_r = np.cos(th)[None, :, None] * u[:, None, :]
        e_r += np.sin(th)[None, :, None] * v[:, None, :]
        return e_r

    def _tangential(self, th):
        """``d e_r / d theta``, same shape - a fresh array, safe to accumulate into."""
        u, v, _ = self.frame_vectors()
        e_th = np.cos(th)[None, :, None] * v[:, None, :]
        e_th -= np.sin(th)[None, :, None] * u[:, None, :]
        return e_th

    def _ring(self, th, k, e_r=None):
        """Surface points at angles `th`, ``(M, T, 3)``.

        Accumulated in place: at a 70k-node arbor one of these is ~113 MB, so the
        obvious ``vertices + r * e_r`` spelling would hold three of them live where
        two will do. Pass `e_r` when the caller already has it for these same nodes.
        """
        p = (self._radial(th) if e_r is None else e_r) * self.radius_at(th, k=k)[:, :, None]
        p += self.vertices[:, None, :]
        return p

    def _axial_neighbours(self):
        """Per-node branch predecessor and successor; a branch end is its own.

        Recomputed from `edges` rather than stored: the walk is ~8 ms on a
        30k-node arbor against several hundred for the ring arithmetic it feeds,
        which is not worth a cache that `edges` could invalidate.
        """
        m = len(self.nodes)
        flat, starts, lengths = _branches(np.asarray(self.edges, dtype=np.int64), m)
        prev_of, next_of, _, _ = _neighbour_slots(flat, starts, lengths, m)
        return prev_of, next_of

    def _subset(self, sel):
        """A `TubeProfile` over a node subset; `edges` are dropped (indices move)."""
        return TubeProfile(
            nodes=self.nodes[sel], edges=np.empty((0, 2), np.int32),
            a0=self.a0[sel], mag=self.mag[sel], phase=self.phase[sel],
            frame=self.frame[sel], branch=self.branch[sel], flags=self.flags[sel],
            residual=self.residual[sel], non_star=self.non_star[sel],
            spacing=self.spacing, n_theta=self.n_theta,
        )

    # -- storage ----------------------------------------------------------

    def to_gpu_buffer(self, form="cartesian"):
        """``(buffer, header)`` ready for a single storage-buffer upload.

        `buffer` is one C-contiguous ``(M, 8 + 2K)`` float32 array in shader struct
        order ``[pos.xyz, quat.xyzw, a0, <K coefficients>, <K coefficients>]`` -
        16 floats, 64 bytes a node at ``K=4``. Positions are physical, so the
        shader needs no spacing. A vertex-pulling shader derives both the node and
        the angular sample from ``vertex_index``, which is what makes LOD selection
        a uniform change rather than a buffer swap::

            let i = vid / N_theta;  let j = vid % N_theta;
            let theta = 2.0 * PI * f32(j) / f32(N_theta);
            // r = a0 + sum_k [ a_k*cos(k*theta) + b_k*sin(k*theta) ]
            // p = pos + r * (cos(theta)*u + sin(theta)*v)

        Parameters
        ----------
        form :  ``"cartesian"`` (default) | ``"polar"``
                Which pair of coefficients to write. ``"cartesian"`` gives
                ``a_1..a_K`` then ``b_1..b_K``; ``"polar"`` gives ``m_1..m_K`` then
                ``phi_1..phi_K``.

                Cartesian is the default because this method exists to feed a
                shader, and a shader wants it: ``sum a_k cos(k t) + b_k sin(k t)``
                walks ``cos(k t)``/``sin(k t)`` upward by angle addition, so the
                whole series costs one ``cos`` and one ``sin`` for *any* ``K``, and
                ``dr/dtheta`` (i.e. the surface normal) falls out of the same loop.
                The polar form would need a ``cos(phi_k)`` and a ``sin(phi_k)`` per
                harmonic per vertex.

                Polar remains the *storage* form (`mag`/`phase`, `save_npz`) for
                the opposite reason: ``m_k`` is frame-independent, so it is the
                only form a truncation threshold can be applied to, and ``phi_k``
                delta-codes along a branch. Ask for it here only if the consumer
                genuinely needs magnitudes on the GPU - to threshold or fade
                harmonics per node, say.

        Returns
        -------
        buffer :    (M, 8 + 2K) float32, C-contiguous.
        header :    dict with what the uniforms need - ``K``, ``n_theta``,
                    ``n_nodes``, ``stride_floats``, ``spacing`` and ``form``.

        Notes
        -----
        Bind this **flattened**. pygfx (and WGSL generally) has no ``vec16``, so a
        storage buffer of 16-float rows has to be declared ``array<f32>`` and
        indexed as ``i * stride_floats``; handing the 2-D array straight to
        `pygfx.Buffer` raises ``Unexpected vertex format '16xf4'``.

        """
        if form not in ("cartesian", "polar"):
            raise ValueError(f"form must be 'cartesian' or 'polar', got {form!r}")

        m, K = len(self.nodes), self.K
        buf = np.empty((m, 8 + 2 * K), dtype=np.float32)
        buf[:, 0:3] = self.vertices
        buf[:, 3:7] = self.frame
        buf[:, 7] = self.a0
        first, second = (
            self.coefficients() if form == "cartesian" else (self.mag, self.phase)
        )
        buf[:, 8 : 8 + K] = first
        buf[:, 8 + K :] = second
        header = {
            "K": K,
            "n_theta": self.n_theta,
            "n_nodes": m,
            "stride_floats": 8 + 2 * K,
            "spacing": None if self.spacing is None else tuple(map(float, self.spacing)),
            "form": form,
        }
        return buf, header

    def save_npz(self, filepath, compress=True):
        """Write to ``.npz``, delta-coding `a0` and `phase` along each branch.

        Both vary smoothly node to node, so the difference compresses far better
        than the value. The delta restarts at every branch boundary: across a
        junction the frame is discontinuous and a delta there would be noise, which
        is exactly what `branch` is stored for.
        """
        order = np.argsort(self.branch, kind="stable")
        new_run = np.ones(len(order), dtype=bool)
        if len(order) > 1:
            new_run[1:] = self.branch[order][1:] != self.branch[order][:-1]

        payload = {
            "order": order.astype(np.int32),
            "nodes": self.nodes.astype(np.float32),
            "edges": self.edges.astype(np.int32),
            "a0_delta": _delta_encode(self.a0[order], new_run),
            "mag": self.mag[order].astype(np.float32),
            "phase_delta": _delta_encode(self.phase[order], new_run),
            "frame": self.frame[order].astype(np.float32),
            "branch": self.branch[order].astype(np.int32),
            "flags": self.flags[order],
            "residual": self.residual[order].astype(np.float32),
            "non_star": self.non_star[order].astype(np.float32),
            "spacing": np.asarray([] if self.spacing is None else self.spacing, float),
            "n_theta": np.int64(self.n_theta),
        }
        writer = np.savez_compressed if compress else np.savez
        writer(filepath, **payload)

    @classmethod
    def load_npz(cls, filepath):
        """Inverse of `save_npz`."""
        with np.load(filepath) as z:
            data = {k: z[k] for k in z.files}
        order = data["order"].astype(np.int64)
        branch = data["branch"]
        new_run = np.ones(len(order), dtype=bool)
        if len(order) > 1:
            new_run[1:] = branch[1:] != branch[:-1]

        inv = np.empty(len(order), dtype=np.int64)
        inv[order] = np.arange(len(order))
        spacing = data["spacing"]
        return cls(
            nodes=data["nodes"], edges=data["edges"],
            a0=_delta_decode(data["a0_delta"], new_run)[inv],
            mag=data["mag"][inv],
            phase=_delta_decode(data["phase_delta"], new_run)[inv],
            frame=data["frame"][inv], branch=branch[inv], flags=data["flags"][inv],
            residual=data["residual"][inv], non_star=data["non_star"][inv],
            spacing=None if spacing.size == 0 else spacing,
            n_theta=int(data["n_theta"]),
        )

    def to_skeleton(self):
        """Degrade to a plain `Skeleton` with ``a0`` as the radius - a classic tube."""
        return Skeleton(
            self.nodes.astype(float), self.edges.astype(np.int64),
            self.a0.astype(float), self.spacing,
        )

    def to_swc(self, filepath=None, root=None):
        """SWC table via `Skeleton.to_swc`, using ``a0`` as the radius."""
        return self.to_skeleton().to_swc(filepath=filepath, root=root)


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


def _caps(skeleton, m, max_radius, spacing):
    """Per-node ceiling on ray travel, in physical units."""
    if max_radius is not None:
        return np.full(m, float(max_radius))
    step = float(np.linalg.norm(spacing)) if spacing is not None else np.sqrt(3.0)
    if skeleton.radii is not None and len(skeleton.radii) == m:
        r = np.asarray(skeleton.radii, dtype=float)
        # Two graph steps is the floor: a radius of 0 would otherwise stop the ray
        # before it can leave the node's own voxel.
        return np.maximum(r * _MAX_RADIUS_FACTOR, 2.0 * step)
    scale = float(np.max(spacing)) if spacing is not None else 1.0
    return np.full(m, _FALLBACK_RADIUS * scale)


def _bit(mask, flag):
    """`flag` where `mask`, as uint8 - a bool times an int widens to int64."""
    return mask.astype(np.uint8) * flag


def _junction_mask(edges, degree, m):
    """Degree>=3 nodes and their direct neighbours."""
    junction = degree >= 3
    if not len(edges):
        return junction
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    junction[dst[junction[src]]] = True
    return junction


def _empty_profile(K, spacing, n_theta):
    """A well-formed `TubeProfile` with no nodes."""
    return TubeProfile(
        nodes=np.empty((0, 3), np.float32), edges=np.empty((0, 2), np.int32),
        a0=np.empty(0, np.float32), mag=np.empty((0, K), np.float32),
        phase=np.empty((0, K), np.float32), frame=np.empty((0, 4), np.float32),
        branch=np.empty(0, np.int32), flags=np.empty(0, np.uint8),
        residual=np.empty(0, np.float32), non_star=np.empty(0, np.float32),
        spacing=spacing, n_theta=n_theta,
    )


@sparse_aware
def tube_profile(
    voxels,
    skeleton,
    *,
    K=4,
    n_theta=64,
    spacing=None,
    max_radius=None,
    diagnostics=False,
    star_window=1.0,
    backend=None,
    verbose=False,
):
    """Fit a truncated Fourier cross-section profile to every skeleton node.

    Each node fires `n_theta` rays in the plane normal to its skeleton tangent and
    records the first exit from `voxels`; an rFFT of those radii gives the
    coefficients. Rays are walked over the sparse voxel set directly, so nothing is
    ever densified and the cost scales with the node count times the local radius,
    never with the bounding box.

    Parameters
    ----------
    voxels :        (N, 3) integer voxel coordinates (XYZ), as elsewhere in
                    sparse-cubes. The object the skeleton describes.
    skeleton :      `Skeleton`, e.g. from `teasar_skeletonize`. Its `edges` supply
                    the branch decomposition the frames propagate along, and its
                    `radii` (when set) bound how far a ray may travel.
    K :             int, optional. Angular truncation order - how many harmonics to
                    keep. Must satisfy ``K < n_theta // 2``, so the Nyquist bin is
                    never stored. 4 keeps ellipticity and mild lobing.
    n_theta :       int, optional. Rays cast per node; sets the angular resolution
                    and most of the cost. 64 by default.
    spacing :       length-3, optional. Physical voxel spacing. Defaults to the
                    skeleton's. Radii and residuals come out in these units, and
                    rays are cast in *physical* directions - which matters, because
                    at 4x4x40 nm the radial quantization is direction-dependent and
                    aliases into specific ``k`` as a function of the neurite's
                    orientation. The noise floor on ``m_k`` is therefore not flat;
                    `frame_vectors` returns the tangent so that artifact can be
                    measured rather than mistaken for structure.
    max_radius :    float, optional. Hard ceiling on ray travel, physical units.
                    Defaults to 4x the node's own radius, or a fixed fallback when
                    the skeleton has no radii. Rays that reach it set
                    `FLAG_RAY_ESCAPED`.
    diagnostics :   bool, optional. Keep marching each ray past its first exit and
                    record, in `TubeProfile.non_star`, what fraction of a node's
                    rays re-enter the object. That star-shapedness failure rate is
                    the hard ceiling on how well *any* ``r(theta)`` representation
                    can do here; it costs roughly one extra pass. Off by default.
    star_window :   float, optional. How much further past its first exit a ray
                    looks for a re-entry, as a multiple of the exit distance
                    (default 1.0, i.e. out to twice the local radius). This is
                    deliberately *not* `max_radius`: widen it and the rate climbs
                    smoothly as rays start reaching unrelated neighbouring
                    neurites, which is a fact about the arbor's packing density and
                    not about whether this cross-section is star-shaped. Anything
                    of order 1 measures the cross-section; large values measure the
                    neighbourhood.
    backend :       ``"exits"`` | ``"sparse"`` | ``"keys"``, optional. Force a
                    raycasting backend. ``"exits"`` hands the whole DDA to
                    `dijkstra3d_sparse.Graph.ray_exits`; the other two run the
                    numpy walk over a `Graph.index_of` or a `searchsorted`
                    membership probe. The default takes the best available,
                    degrading in that order. Only speed differs - all three are
                    pinned bit-for-bit against each other in the tests.
    verbose :       bool.

    Returns
    -------
    TubeProfile

    Notes
    -----
    Frames do not propagate consistently through a bifurcation and the
    cross-section parameterization is undefined in the junction region; those nodes
    carry `FLAG_JUNCTION` rather than a fix. Somata are not tube-like and are best
    excluded upstream.

    """
    validate(voxels)
    if not isinstance(skeleton, Skeleton):
        raise TypeError(f"skeleton must be a Skeleton, got {type(skeleton).__name__}")
    K, n_theta = int(K), int(n_theta)
    if n_theta < 4:
        raise ValueError(f"n_theta must be >= 4, got {n_theta}")
    if not 0 <= K < n_theta // 2:
        raise ValueError(
            f"K must satisfy 0 <= K < n_theta // 2 = {n_theta // 2}, got {K}. The "
            "Nyquist bin is deliberately excluded so that every stored harmonic "
            "shares one magnitude/phase convention."
        )

    # `as_spacing` already handles None, scalars and the shape/positivity checks.
    spacing = as_spacing(spacing if spacing is not None else skeleton.spacing)

    m = len(skeleton.nodes)
    if m == 0 or len(voxels) == 0:
        return _empty_profile(K, spacing, n_theta)

    obj = np.asarray(voxels)
    pos = np.asarray(skeleton.nodes, dtype=float)  # index space
    verts = pos * spacing if spacing is not None else pos  # physical
    edges = np.asarray(skeleton.edges, dtype=np.int64).reshape(-1, 2)

    # -- topology, tangents, frames --------------------------------------
    flat, starts, lengths = _branches(edges, m)
    prev_of, next_of, branch, is_end = _neighbour_slots(flat, starts, lengths, m)
    tangent = _normalize(verts[next_of] - verts[prev_of])
    # An isolated node has no direction of travel at all; +z is as good as any and
    # keeps the frame orthonormal instead of zero.
    tangent[np.linalg.norm(tangent, axis=1) < 0.5] = (0.0, 0.0, 1.0)
    u, v = _frames(verts, tangent, flat, starts, lengths)
    log(
        f"Tube: {m} nodes, {len(lengths)} branches, longest "
        f"{int(lengths[0]) if len(lengths) else 0}.",
        verbose=verbose,
    )

    flags = np.zeros(m, dtype=np.uint8)
    flags[is_end] |= FLAG_BRANCH_END
    flags[_junction_mask(edges, skeleton.node_degrees(), m)] |= FLAG_JUNCTION

    # -- raycast, chunked over nodes -------------------------------------
    caster = _caster(obj, backend)
    caps = _caps(skeleton, m, max_radius, spacing)

    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Fold 1/spacing into a *copy* of the frame once rather than scaling every
    # chunk's directions: the ray parameter still comes out as a physical distance,
    # with no per-chunk (chunk, n_theta, 3) rescale. `u`/`v` themselves must stay
    # unit - the stored quaternion is built from them further down.
    u_idx, v_idx = (u, v) if spacing is None else (u / spacing, v / spacing)

    # A node whose own voxel is empty has nothing to measure - a wavefront ring
    # centroid at a junction can land outside the object. Resolved *before* the
    # cast and skipped, not after: a ray launched from an empty cell exits at the
    # first face it meets, which is a plausible-looking radius of about half a
    # voxel rather than an obvious one, and the backends do not even agree on it.
    # Leaving `a0 = 0` alongside the flag is both cheaper and unambiguous.
    seeded = caster.inside(np.rint(pos).astype(np.int32))
    flags |= _bit(~seeded, FLAG_SEED_OUTSIDE)

    radii = np.zeros((m, n_theta))
    non_star = np.zeros(m)
    chunk = max(1, _RAY_BLOCK // n_theta)
    log(
        f"Tube: casting {int(seeded.sum()) * n_theta} rays ({chunk} nodes/chunk, "
        f"'{caster.name}' backend).",
        verbose=verbose,
    )

    for lo in range(0, m, chunk):
        hi = min(lo + chunk, m)
        rows = lo + np.flatnonzero(seeded[lo:hi])
        if not len(rows):
            continue
        # (c, n_theta, 3): a physically unit direction already divided by spacing,
        # i.e. index space, so the DDA parameter is a physical distance.
        d_idx = (
            cos_t[None, :, None] * u_idx[rows, None, :]
            + sin_t[None, :, None] * v_idx[rows, None, :]
        ).reshape(-1, 3)
        origins = np.repeat(pos[rows], n_theta, axis=0)

        r, escaped, reentered = caster.cast(
            origins, d_idx, np.repeat(caps[rows], n_theta), diagnostics,
            float(star_window),
        )
        c = len(rows)
        radii[rows] = r.reshape(c, n_theta)
        non_star[rows] = reentered.reshape(c, n_theta).mean(axis=1)
        flags[rows] |= _bit(escaped.reshape(c, n_theta).any(axis=1), FLAG_RAY_ESCAPED)
        flags[rows] |= _bit(non_star[rows] > 0, FLAG_NON_STAR)

    # -- Fourier ----------------------------------------------------------
    a0, mag, phase, residual = _fourier(radii, K)
    log(
        f"Tube: mean a0 {a0.mean():.3f}, mean residual {residual.mean():.4f}, "
        f"{int(((flags & FLAG_NON_STAR) != 0).sum())} non-star node(s) "
        f"({100 * non_star.mean():.1f}% of rays).",
        verbose=verbose,
    )

    return TubeProfile(
        nodes=pos.astype(np.float32),
        edges=edges.astype(np.int32),
        a0=a0.astype(np.float32),
        mag=mag.astype(np.float32),
        phase=phase.astype(np.float32),
        frame=_frame_to_quat(u, v, tangent).astype(np.float32),
        branch=branch,
        flags=flags,
        residual=residual.astype(np.float32),
        non_star=non_star.astype(np.float32),
        spacing=spacing,
        n_theta=n_theta,
    )


_SKELETONIZERS = ("teasar", "wavefront", "thin")


@sparse_aware
def tube_coefficients(
    voxels,
    *,
    method="teasar",
    spacing=None,
    K=4,
    n_theta=64,
    max_radius=None,
    diagnostics=False,
    star_window=1.0,
    backend=None,
    verbose=False,
    **skeleton_kwargs,
):
    """Skeletonize `voxels` and fit a tube profile in one call.

    Convenience wrapper: ``tube_profile(voxels, <method>_skeletonize(voxels), ...)``.
    See `tube_profile` for the profile parameters.

    Parameters
    ----------
    method :            ``"teasar"`` (default), ``"wavefront"`` or ``"thin"``.
                        TEASAR is the default because it produces a tree with a
                        medial-axis radius per node, which is what a tube wants.
                        `wavefront_skeletonize` is faster and gives sub-voxel node
                        positions, but places nodes on level sets of the wave rather
                        than perpendicular to the local tube axis.
    skeleton_kwargs :   forwarded to the skeletonizer (e.g. ``min_branch_length``).

    Returns
    -------
    TubeProfile

    """
    if method not in _SKELETONIZERS:
        raise ValueError(f"method must be one of {_SKELETONIZERS}, got {method!r}")
    if method == "teasar":
        from .teasar import teasar_skeletonize as _skeletonize
    elif method == "wavefront":
        from .wavefront import wavefront_skeletonize as _skeletonize
    else:
        from .skeleton import thin_skeletonize as _skeletonize

    skel = _skeletonize(voxels, spacing=spacing, verbose=verbose, **skeleton_kwargs)
    return tube_profile(
        voxels, skel, K=K, n_theta=n_theta, spacing=spacing, max_radius=max_radius,
        diagnostics=diagnostics, star_window=star_window, backend=backend,
        verbose=verbose,
    )
