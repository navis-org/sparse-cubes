"""Shared fixtures and topology helpers for the thinning / skeleton tests."""

import numpy as np

from sparsecubes.skeleton import _edges_26


# --- synthetic voxel shapes ------------------------------------------------

def line(n=8, axis=0):
    """A straight 1-voxel-wide line of length `n` along `axis`."""
    v = np.zeros((n, 3), dtype=np.int64)
    v[:, axis] = np.arange(n)
    return v


def l_shape():
    """A right-angle bend (two straight arms sharing a corner)."""
    a = line(8, axis=0)
    b = line(8, axis=1)
    return np.unique(np.vstack([a, b]), axis=0)


def y_branch(arm=7):
    """Three straight arms meeting at the origin (a clean degree-3 junction)."""
    arms = [np.zeros((arm, 3), dtype=np.int64) for _ in range(3)]
    for ax in range(3):
        arms[ax][:, ax] = np.arange(1, arm + 1)
    junction = np.zeros((1, 3), dtype=np.int64)
    return np.unique(np.vstack([junction, *arms]), axis=0)


def solid_cube(k):
    """A solid k x k x k cube."""
    g = np.indices((k, k, k)).reshape(3, -1).T
    return g.astype(np.int64)


def slab(k):
    """A flat k x k x 1 slab in the z=0 plane."""
    xs, ys = np.meshgrid(np.arange(k), np.arange(k))
    return np.stack([xs.ravel(), ys.ravel(), np.zeros(k * k, dtype=int)], axis=1).astype(
        np.int64
    )


def solid_cylinder(radius=4, length=20):
    """A solid cylinder of the given radius running along +z."""
    r = radius
    rng = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(rng, rng)
    disk = np.stack([xx.ravel(), yy.ravel()], axis=1)
    disk = disk[(disk[:, 0] ** 2 + disk[:, 1] ** 2) <= r * r]
    out = []
    for z in range(length):
        col = np.column_stack([disk[:, 0] + r, disk[:, 1] + r, np.full(len(disk), z)])
        out.append(col)
    return np.vstack(out).astype(np.int64)


# The profile fixtures below all put their true axis at ``(PAD, PAD)`` in xy,
# independent of their radii. The bounding-box centre will NOT do: a shape with odd
# angular symmetry (3 or 5 lobes) has an asymmetric bbox, and seeding the skeleton
# at its centre puts a real offset into the k=1 harmonic - which is exactly what
# k=1 is for, and so would silently corrupt any test that used it.
PAD = 16


def _extrude(keep, xx, yy, length):
    """Stack a boolean cross-section mask along +z into a solid cylinder."""
    disk = np.stack([xx[keep] + PAD, yy[keep] + PAD], axis=1)
    out = [np.column_stack([disk, np.full(len(disk), z)]) for z in range(length)]
    return np.vstack(out).astype(np.int64)


def _section_grid(reach):
    """Symmetric integer meshgrid wide enough for a cross-section of `reach`."""
    rng = np.arange(-int(np.ceil(reach)) - 1, int(np.ceil(reach)) + 2)
    return np.meshgrid(rng, rng)


def elliptic_cylinder(a=10, b=5, length=40, rot=0.0):
    """A solid elliptic cylinder along +z, semi-axes `a`/`b`, rotated by `rot`.

    An ellipse is the shape a tube profile should resolve almost entirely in the
    ``k=2`` harmonic. `rot` (radians, about the cylinder's own axis) is there to
    test that the stored magnitudes are frame-independent while the phases are not.
    Axis at ``(PAD, PAD)``.
    """
    xx, yy = _section_grid(max(a, b))
    c, s = np.cos(-rot), np.sin(-rot)
    xr, yr = c * xx - s * yy, s * xx + c * yy
    return _extrude((xr / float(a)) ** 2 + (yr / float(b)) ** 2 <= 1.0, xx, yy, length)


def lobed_cylinder(radius=8, n_lobes=3, amp=1.5, length=40):
    """A solid cylinder along +z with radius ``radius + amp*cos(n_lobes*theta)``.

    Puts a known amount of energy in exactly one harmonic, ``k = n_lobes``, with
    ``a0 = radius`` and ``m_k = amp``. That makes it an oracle for the angular
    decomposition rather than just a smoke test. Axis at ``(PAD, PAD)``.
    """
    xx, yy = _section_grid(radius + abs(amp))
    keep = np.hypot(xx, yy) <= radius + amp * np.cos(n_lobes * np.arctan2(yy, xx))
    return _extrude(keep, xx, yy, length)


def disc_cylinder(radius=8, length=40):
    """A plain circular cylinder on the `lobed_cylinder` axis, i.e. at ``(PAD, PAD)``.

    `solid_cylinder` cannot stand in: it centres on its own radius rather than on
    `PAD`, and the tests that read a single ray direction depend on the axis being
    at a known point. Spelling this as ``lobed_cylinder(n_lobes=4, amp=0.0)`` works
    but reads as though the lobe count meant something.
    """
    return lobed_cylinder(radius=radius, n_lobes=1, amp=0.0, length=length)


def axial_skeleton(voxels, centre=(PAD, PAD), spacing=None):
    """The exact straight +z centerline of a cylinder fixture, as a `Skeleton`.

    Using the known axis instead of a skeletonizer's output keeps the coefficient
    tests measuring the *profile* rather than the skeletonizer: TEASAR roots a fat
    cylinder at a cap corner, which drags its first nodes well off the axis - a
    property of TEASAR, not of the extraction.
    """
    from sparsecubes.skeleton import Skeleton

    lo, hi = int(voxels[:, 2].min()), int(voxels[:, 2].max())
    n = hi - lo + 1
    nodes = np.empty((n, 3), dtype=float)
    nodes[:, 0], nodes[:, 1] = centre[0], centre[1]
    nodes[:, 2] = np.arange(lo, hi + 1)
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    return Skeleton(nodes, edges, None, None if spacing is None else np.asarray(spacing))


def annulus(outer=9, inner=5):
    """A flat ring (annulus) in the z=0 plane - a shape with one loop."""
    rng = np.arange(-outer, outer + 1)
    xx, yy = np.meshgrid(rng, rng)
    d2 = xx.ravel() ** 2 + yy.ravel() ** 2
    keep = (d2 <= outer * outer) & (d2 >= inner * inner)
    pts = np.stack([xx.ravel()[keep] + outer, yy.ravel()[keep] + outer], axis=1)
    return np.column_stack([pts, np.zeros(len(pts), dtype=int)]).astype(np.int64)


def hollow_box(outer=7, void=3):
    """A solid `outer` cube with a centered `void` cube of empty interior cells.

    The empty core is fully surrounded by object voxels (an enclosed cavity, a
    non-zero b2 generator), so topological thinning cannot collapse the shell.
    Returns the shell voxels; `full_box`/`void_cells` give the pieces.
    """
    cube = solid_cube(outer)
    lo = (outer - void) // 2
    hi = lo + void
    inside = np.all((cube >= lo) & (cube < hi), axis=1)
    return cube[~inside]


def box_with_tunnel(k=5):
    """A solid `k` cube with a straight column removed along z (open both ends).

    A through-tunnel (a b1 loop), *not* an enclosed void: the empty column
    reaches the exterior at both faces, so `fill_cavities` must leave it be.
    """
    cube = solid_cube(k)
    c = k // 2
    column = (cube[:, 0] == c) & (cube[:, 1] == c)
    return cube[~column]


# Geometry of the self-touch hairpin fixture, shared with the tests that probe
# its arms (so the detector references the same numbers the builder used).
HAIRPIN = dict(soma_r=10, arm_r=3, x0=30, x1=150, y_left=30, y_right=50, cz=30)


def _ball(center, r):
    rr = int(np.ceil(r))
    g = np.mgrid[-rr : rr + 1, -rr : rr + 1, -rr : rr + 1].reshape(3, -1).T
    return g[(g ** 2).sum(1) <= r * r] + np.asarray(center)


def _cylinder_x(x0, x1, cy, cz, r):
    """A solid cylinder of radius `r` running along +x from x0 to x1 (inclusive)."""
    rr = int(np.ceil(r))
    dy, dz = np.mgrid[-rr : rr + 1, -rr : rr + 1].reshape(2, -1)
    keep = dy * dy + dz * dz <= r * r
    dy, dz = dy[keep], dz[keep]
    return np.vstack(
        [np.column_stack([np.full(dy.size, x), cy + dy, cz + dz]) for x in range(x0, x1 + 1)]
    )


def _wire(waypoints):
    """A dense 1-voxel-wide 26-connected polyline through integer waypoints."""
    pts = [np.asarray(p, float) for p in waypoints]
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = int(np.abs(b - a).max()) + 1
        out.append(np.round(a + (b - a) * np.linspace(0, 1, n)[:, None]).astype(int))
    return np.vstack(out)


def self_touch_hairpin():
    """A thick bar folded back on itself, self-touching through a thin bridge.

    Two thick parallel arms share a thick base (a soma-like blob - the globally
    thickest region) and are joined at their far ends by a short *thin* wire: the
    kind of fine self-touch a neuron makes when a neurite curls back and contacts
    itself. The shape has one loop, so a skeletonizer must break it either at the
    thin bridge (correct - both thick arms stay intact) or by severing a thick arm
    (the backbone break this fixture exposes). The soma makes the arms thin
    *relative to the global-max* thickness, which is exactly what lets the default
    penalty field prefer the thin-bridge short-cut and cut an arm; `pdrf_ref`
    saturates that away. Geometry is fixed in `HAIRPIN`.
    """
    g = HAIRPIN
    cy = (g["y_left"] + g["y_right"]) // 2
    parts = [
        _ball((g["x0"], cy, g["cz"]), g["soma_r"]),
        _cylinder_x(g["x0"], g["x1"], g["y_left"], g["cz"], g["arm_r"]),
        _cylinder_x(g["x0"], g["x1"], g["y_right"], g["cz"], g["arm_r"]),
        _wire([(g["x1"], g["y_left"], g["cz"]), (g["x1"], g["y_right"], g["cz"])]),
    ]
    return np.unique(np.vstack(parts).astype(np.int64), axis=0)


# --- topology helpers ------------------------------------------------------

def _union_find(m, edges):
    parent = list(range(m))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    roots = {find(i) for i in range(m)}
    return len(roots)


def betti(nodes, edges):
    """Return (n_components, first_betti) of the 26-connectivity graph."""
    m = len(nodes)
    if m == 0:
        return 0, 0
    c = _union_find(m, edges)
    return c, len(edges) - m + c


def n_components(voxels):
    """Number of 26-connected components of a raw voxel set."""
    voxels = np.asarray(voxels)
    if len(voxels) == 0:
        return 0
    nodes, edges = _edges_26(voxels.astype(np.int64))
    return betti(nodes, edges)[0]


def has_2x2x2_block(voxels):
    """True if the voxel set contains a fully occupied 2x2x2 block."""
    s = set(map(tuple, np.asarray(voxels).tolist()))
    for x, y, z in s:
        if all(
            (x + dx, y + dy, z + dz) in s
            for dx in (0, 1)
            for dy in (0, 1)
            for dz in (0, 1)
        ):
            return True
    return False


def is_subset(sub, full):
    """True if every row of `sub` is a row of `full`."""
    fs = set(map(tuple, np.asarray(full).tolist()))
    return all(tuple(r) in fs for r in np.asarray(sub).tolist())
