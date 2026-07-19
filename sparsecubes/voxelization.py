"""Sparse voxelization of triangle meshes.

This is the inverse of `sparsecubes.mesh`: it turns a triangle mesh into the
library's canonical sparse `(N, 3)` integer voxel representation. As everywhere
else in this package, **no dense 3D grid is ever allocated** - memory and work
scale with the mesh surface area (plus the number of voxels actually emitted),
never with the bounding-box volume.

Two stages, both fully vectorized over batches of triangles:

1. **Surface** (`_surface_keys`). Exact conservative rasterization. A triangle
   overlaps a voxel cube iff (a) the triangle's *plane* overlaps the cube and
   (b) in each of the three coordinate projections the 2D triangle overlaps the
   2D square (Schwarz & Seidel 2010). That is the 13-axis triangle/AABB
   separating-axis test reorganized into 1 + 3x3 half-space tests, so the result
   is exact, not merely conservative. Candidate cells come from each triangle's
   own integer bounding box, which is why long triangles are pre-split
   (`_iter_triangles`) - a triangle spanning `L` cells has `O(L^2)` area but an
   `O(L^3)` bounding box, and splitting keeps total work proportional to area.

2. **Interior** (`_interior_keys`). Sparse scanline parity fill. Each triangle is
   rasterized in the XY projection over voxel *column centres*; every hit yields
   one Z crossing. Crossings are sorted per column and paired up, and the cells
   between a pair are emitted as runs. Memory is `O(#crossings + #output)`.
   The even-odd rule makes this independent of face winding (so meshes with
   inconsistent normals still work) and it naturally leaves enclosed cavities
   empty.

   The subtle part is a column centre that lands exactly on a shared triangle
   edge or vertex, which real meshes do constantly - any axis-aligned face whose
   split diagonal crosses column centres, any vertex sitting on the axis such as
   a sphere's pole or a cone's apex. Such a point has to be claimed by exactly one
   triangle; claimed twice or not at all, the column's parity flips and the solid
   gets a hole (or a plug) driven right through it. This is handled exactly, with
   no epsilon anywhere: `_edge_functions` arranges the arithmetic so neighbouring
   triangles produce bitwise negated edge values, and `_point_in_triangle` breaks
   the resulting exact ties by symbolic perturbation.

Trimesh can already produce sparse *surface* voxels (`mesh.voxelized(pitch)`),
but every one of its interior-fill paths materializes a dense bounding box; the
solid path here is the reason this module exists.
"""

import warnings

import numpy as np
import trimesh as tm

from .core import unpack, log, unique, _PACK_FIELD

# Triangles are split until no edge is longer than this many voxels. Total
# candidate cells scale as ~area * (K+1)^3 / K^2, which is flat-ish around K=2-6;
# larger K means less subdivision work but more wasted candidate tests.
_MAX_EDGE = 4.0

# Candidate (triangle, cell) pairs tested at once. Each pair costs ~60 bytes of
# coordinate columns and test temporaries.
_CAND_BUDGET = 1 << 20

# Triangles held in memory at once. This is the dominant term in peak memory:
# the surface stage precomputes 27 per-triangle half-plane coefficients, so a
# batch costs ~250 bytes per triangle. Smaller batches trade a few percent of
# runtime for a large drop in peak - worth it here, where staying small is the
# whole point of the library.
_TRI_BUDGET = 1 << 16

# Voxels emitted per run-expansion chunk in the fill stage.
_RUN_BUDGET = 1 << 22

# Surface keys buffered before they are merged and deduplicated (8 bytes each).
_MERGE_BUDGET = 1 << 23

# (a, b, c) cyclic axis triples: project onto axes (a, b); the 2D cross product
# of the projected edges then equals the 3D normal's `c` component.
_PROJECTIONS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))

# Field width of core.pack()'s default (21, 21, 21) layout. We build and combine
# keys from separate coordinate columns here (to avoid materializing (N, 3)
# temporaries), so the layout has to be spelled out rather than delegated.
_BITS = _PACK_FIELD.bit_length() - 1
assert _PACK_FIELD == 1 << _BITS


def _key(x, y, z):
    """Packed key identical to `core.pack`, built from separate columns."""
    return (x << (2 * _BITS)) | (y << _BITS) | z


def voxelize(mesh, spacing=1.0, *, solid=True, verbose=False):
    """Voxelize a triangle mesh into sparse `(N, 3)` integer voxel indices.

    The inverse of `sparsecubes.mesh`. Voxel `i` along an axis covers the
    half-open interval ``[(i - 0.5) * spacing, (i + 0.5) * spacing)``, i.e. its
    *centre* sits at ``i * spacing``. This matches trimesh's ``VoxelGrid``
    convention and means ``sc.mesh(sc.voxelize(m, s), spacing=s)`` lands back on
    top of the original mesh.

    No dense grid is allocated at any point: work scales with the mesh surface
    area (and, for ``solid=True``, the number of interior voxels emitted), not
    with the bounding-box volume.

    Parameters
    ----------
    mesh :      `trimesh.Trimesh` or `(vertices, faces)` tuple
                `vertices` is `(V, 3)` float, `faces` is `(F, 3)` integer.
    spacing :   float or length-3 sequence, optional
                Physical size of one voxel; may be anisotropic. Default 1.0.
    solid :     bool, optional
                If True (default) fill the interior using an even-odd scanline
                parity test and return surface + interior voxels. This is exact
                for watertight meshes, ignores face winding, and leaves enclosed
                cavities empty. If the mesh is not watertight some columns cannot
                be paired up; those are left unfilled and a warning is emitted.
                If False, return only the conservative surface shell.
    verbose :   bool, optional. Log progress.

    Returns
    -------
    (N, 3) array of XYZ voxel indices, sorted, deduplicated. Indices are
    absolute and may be negative (dtype is ``int32`` when the values fit,
    otherwise ``int64``).

    Notes
    -----
    The per-axis extent is limited to `pack`'s 21-bit fields (~2.1e6 voxels along
    any one axis); coarsen `spacing` for very large meshes.

    See Also
    --------
    sparsecubes.mesh :          The inverse operation.
    sparsecubes.fill_cavities : Fill enclosed voids in an existing voxel set.
    """
    vertices, faces = _as_mesh(mesh)

    spacing = np.asarray(spacing, dtype=np.float64).ravel()
    if spacing.size == 1:
        spacing = np.repeat(spacing, 3)
    if spacing.size != 3:
        raise ValueError("`spacing` must be a scalar or a length-3 sequence.")
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"`spacing` must be finite and positive, got {spacing}.")

    if len(faces) == 0 or len(vertices) == 0:
        return np.empty((0, 3), dtype=np.int32)

    # Work in "cell units" where voxel `i` occupies [i, i + 1), so the cell
    # containing a point is simply floor(). The +0.5 encodes the centre-sampling
    # convention documented above.
    verts = vertices.astype(np.float64) / spacing + 0.5
    if not np.all(np.isfinite(verts)):
        raise ValueError("Mesh contains non-finite vertices.")

    imin = np.floor(verts.min(axis=0)).astype(np.int64)
    imax = np.floor(verts.max(axis=0)).astype(np.int64)
    # `pack` needs non-negative coordinates; keep a one-cell margin on both sides
    # so neighbour arithmetic on the returned keys stays in range.
    shift = imin - 1
    extent = int((imax - imin).max()) + 3
    if extent >= _PACK_FIELD:
        raise ValueError(
            f"Voxel extent reaches {extent}, exceeding pack()'s {_PACK_FIELD}-per-axis "
            f"range. Use a coarser `spacing` (currently {spacing.tolist()})."
        )

    log(
        f"voxelize: {len(faces)} faces, spacing {spacing.tolist()}, "
        f"grid extent {(imax - imin + 1).tolist()}",
        verbose=verbose,
    )

    surface = _surface_keys(verts, faces, shift, verbose)
    keys = surface
    if solid:
        interior = _interior_keys(verts, faces, shift, verbose)
        if len(interior):
            keys = unique(np.concatenate([surface, interior]))
    log(
        f"voxelize: {len(surface)} surface voxel(s), {len(keys)} total.",
        verbose=verbose,
    )

    if len(keys) == 0:
        return np.empty((0, 3), dtype=np.int32)

    keys.sort()
    coords = unpack(keys)
    coords += shift  # in place: the (N, 3) int64 temporary is the big one here
    lo, hi = int(coords.min()), int(coords.max())
    if lo >= np.iinfo(np.int32).min and hi <= np.iinfo(np.int32).max:
        return coords.astype(np.int32)
    return coords


def _as_mesh(mesh):
    """Accept a `trimesh.Trimesh` or a `(vertices, faces)` pair."""
    if isinstance(mesh, tm.Trimesh):
        return np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        vertices = np.asarray(mesh[0], dtype=np.float64)
        faces = np.asarray(mesh[1])
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise TypeError("`vertices` must be a (V, 3) array.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise TypeError("`faces` must be a (F, 3) array of triangles.")
        return vertices, faces
    raise TypeError(
        "Expected a trimesh.Trimesh or a (vertices, faces) tuple, "
        f"got {type(mesh).__name__}."
    )


def _batch_bounds(cost, budget):
    """Split a cost vector into contiguous batches of at most `budget` total.

    Loops over batches, not items, so this is cheap even for millions of faces.
    A single item exceeding `budget` gets a batch to itself.
    """
    if len(cost) == 0:
        return []
    # float64 accumulation regardless of input dtype: a float32 running sum over
    # millions of faces loses enough precision to mis-size the batches.
    cum = np.cumsum(cost, dtype=np.float64)
    bounds = []
    start = 0
    n = len(cost)
    while start < n:
        base = cum[start - 1] if start else 0
        end = int(np.searchsorted(cum, base + budget, side="right"))
        end = max(end, start + 1)
        bounds.append((start, end))
        start = end
    return bounds


def _expand_runs(starts, lengths):
    """Expand inclusive integer runs into a flat array of values.

    ``starts[i]`` repeated for ``lengths[i]`` steps, incrementing by one.
    """
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    run = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    offsets = np.cumsum(lengths) - lengths
    return starts[run] + (np.arange(total, dtype=np.int64) - offsets[run])


def _iter_triangles(verts, faces, max_edge=None):
    """Yield `(M, 3, 3)` batches of triangles, optionally split to `max_edge`.

    Splitting keeps the *surface* stage proportional to area rather than to the
    sum of per-triangle bounding boxes (a triangle spanning `L` cells has `O(L^2)`
    area but an `O(L^3)` box). Batches are sized by the *predicted* sub-triangle
    count so peak memory stays bounded however skewed the size distribution is.

    `max_edge=None` yields the mesh's own triangles untouched. The fill stage
    needs that: splitting introduces T-junctions between neighbours that end up at
    different subdivision levels, and a T-junction destroys the shared-edge
    pairing that `_edge_functions` relies on to resolve ties exactly.
    """
    n = len(faces)
    if max_edge is None:
        for start, end in _batch_bounds(np.ones(n), _TRI_BUDGET):
            yield verts[faces[start:end]]
        return

    # Predicted sub-triangle count per face, measured a chunk at a time: a
    # `verts[faces]` for the whole mesh would be 72 bytes per face (hundreds of MB
    # on a multi-million-face mesh) and is never needed all at once.
    predicted = np.empty(n, dtype=np.float32)
    for s in range(0, n, _TRI_BUDGET):
        e = min(s + _TRI_BUDGET, n)
        t = verts[faces[s:e]]
        edges = np.stack([t[:, 1] - t[:, 0], t[:, 2] - t[:, 1], t[:, 0] - t[:, 2]], 1)
        longest = np.sqrt((edges**2).sum(axis=2)).max(axis=1)
        # subdivide_to_size does 1->4 splits, so a triangle needing `r` halvings
        # yields ~4^r sub-triangles.
        halvings = np.ceil(np.log2(np.maximum(longest / max_edge, 1.0)))
        predicted[s:e] = np.minimum(4.0**halvings, 1e12)
        del t, edges, longest, halvings

    for start, end in _batch_bounds(predicted, _TRI_BUDGET):
        batch = verts[faces[start:end]]
        if predicted[start:end].max() <= 1.0:
            yield batch  # already fine grained - skip the remesh entirely
            continue
        soup = batch.reshape(-1, 3)
        idx = np.arange(len(soup), dtype=np.int64).reshape(-1, 3)
        sub_v, sub_f = tm.remesh.subdivide_to_size(
            soup, idx, max_edge=max_edge, max_iter=32
        )
        yield sub_v[sub_f]


def _surface_keys(verts, faces, shift, verbose):
    """Exact conservative surface voxelization -> sorted, unique packed keys.

    Every geometric coefficient is precomputed *per triangle* so the inner loop
    over candidate cells only ever gathers scalar columns. That keeps peak memory
    at a few tens of bytes per in-flight candidate instead of the ~150 that
    materializing `(T, 3)` vertex/edge/normal arrays would cost.
    """
    out = []
    held = 0
    n_cand = 0
    for tris in _iter_triangles(verts, faces, _MAX_EDGE):
        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
        e0, e1, e2 = v1 - v0, v2 - v1, v0 - v2
        n = np.cross(e0, e1)

        # Zero-area triangles have no plane and cannot contribute a surface.
        keep = (n != 0).any(axis=1)
        if not keep.all():
            tris, n = tris[keep], n[keep]
            v0, e0, e1, e2 = v0[keep], e0[keep], e1[keep], e2[keep]
        if len(tris) == 0:
            continue

        # Signed distances from the triangle plane to the two cube corners that
        # are extreme along the normal; both are per-triangle constants.
        near = (n > 0).astype(np.float64)
        d_near = (n * (near - v0)).sum(axis=1)
        d_far = (n * ((1.0 - near) - v0)).sum(axis=1)
        coeffs = _projection_halfplanes(v0, e0, e1, e2, n)
        del near, e0, e1, e2, v0, v1, v2

        lo = np.floor(tris.min(axis=1)).astype(np.int64) - shift
        hi = np.floor(tris.max(axis=1)).astype(np.int64) - shift
        del tris
        dims = hi - lo + 1
        counts = dims.prod(axis=1)
        n_cand += int(counts.sum())

        for start, end in _batch_bounds(counts, _CAND_BUDGET):
            cell, tri_id = _bbox_cells(lo, dims, counts, start, end)
            p = [(cell[k] + shift[k]).astype(np.float64) for k in range(3)]

            # Plane vs cube: the two extreme corners must bracket the plane.
            npd = n[:, 0][tri_id] * p[0]
            npd += n[:, 1][tri_id] * p[1]
            npd += n[:, 2][tri_id] * p[2]
            ok = (npd + d_near[tri_id]) * (npd + d_far[tri_id]) <= 0
            del npd

            # Three 2D projections, three edges each.
            for a, b, ca, cb, cc in coeffs:
                if not ok.any():
                    break
                side = ca[tri_id] * p[a]
                side += cb[tri_id] * p[b]
                side += cc[tri_id]
                ok &= side >= 0
                del side

            del p
            if ok.any():
                out.append(unique(_key(cell[0][ok], cell[1][ok], cell[2][ok])))
                held += len(out[-1])
            del cell, tri_id, ok
            # Neighbouring triangles keep re-emitting the same cells, so collapse
            # the accumulator whenever the pending duplicates outweigh the cost of
            # merging. Without this it grows with the mesh, not with the result.
            if held > _MERGE_BUDGET:
                out = [unique(np.concatenate(out))]
                held = len(out[0])

    if not out:
        return np.empty(0, dtype=np.int64)
    keys = unique(np.concatenate(out)) if len(out) > 1 else out[0]
    log(
        f"voxelize: surface stage tested {n_cand} candidate cell(s) -> {len(keys)} voxel(s).",
        verbose=verbose,
    )
    return keys


def _projection_halfplanes(v0, e0, e1, e2, n):
    """Per-triangle half-plane coefficients for the three 2D projection tests.

    Yields ``(axis_a, axis_b, coef_a, coef_b, const)`` for each of the nine
    (projection, edge) pairs, such that a cell at ``p`` passes iff
    ``coef_a * p[a] + coef_b * p[b] + const >= 0``.
    """
    out = []
    bases = (v0, v0 + e0, v0 + e0 + e1)  # == v0, v1, v2
    for a, b, c in _PROJECTIONS:
        # Sign of the 3rd normal component is the projected triangle's winding.
        # When it is exactly zero the projection degenerates to a segment and the
        # three half-planes are symmetric under a global flip, so either choice
        # of sign gives the same (correct) answer.
        s = np.where(n[:, c] >= 0, 1.0, -1.0)
        for edge, base in zip((e0, e1, e2), bases):
            ca = -s * edge[:, b]
            cb = s * edge[:, a]
            # Support function of the unit square along (ca, cb).
            cc = (
                np.maximum(ca, 0.0)
                + np.maximum(cb, 0.0)
                - ca * base[:, a]
                - cb * base[:, b]
            )
            out.append((a, b, ca, cb, cc))
    return out


def _bbox_cells(lo, dims, counts, start, end):
    """Enumerate every integer cell in the bounding boxes of triangles [start, end).

    Returns three int64 coordinate columns plus, for each cell, the index of the
    triangle that produced it (indexing the *full* per-triangle arrays, not the
    slice). Columns rather than an `(T, 3)` array so callers can free each axis
    independently.
    """
    cnt = counts[start:end]
    total = int(cnt.sum())
    tri_id = np.repeat(np.arange(start, end, dtype=np.int64), cnt)
    within = np.arange(total, dtype=np.int64)
    within -= np.repeat(np.cumsum(cnt) - cnt, cnt)

    dz = dims[:, 2][tri_id]
    k = within % dz
    within //= dz
    del dz
    dy = dims[:, 1][tri_id]
    j = within % dy
    within //= dy
    del dy

    within += lo[:, 0][tri_id]
    j += lo[:, 1][tri_id]
    k += lo[:, 2][tri_id]
    return (within, j, k), tri_id


def _interior_keys(verts, faces, shift, verbose):
    """Sparse scanline parity fill -> sorted packed keys of interior voxels."""
    cols, zs = _z_crossings(verts, faces, shift)
    if len(cols) == 0:
        return np.empty(0, dtype=np.int64)

    order = np.lexsort((zs, cols))
    cols, zs = cols[order], zs[order]

    # Column boundaries in the sorted crossing list.
    starts = np.flatnonzero(np.concatenate([[True], cols[1:] != cols[:-1]]))
    counts = np.diff(np.concatenate([starts, [len(cols)]]))
    n_odd = int((counts % 2).sum())
    if n_odd:
        warnings.warn(
            f"{n_odd} of {len(starts)} voxel columns have an odd number of surface "
            "crossings, which means the mesh is not watertight. Those columns were "
            "left unfilled; the result may be incomplete. Repair the mesh (e.g. "
            "trimesh's `fill_holes`) or use `solid=False`.",
            stacklevel=3,
        )

    # Pair crossings up within each column: (0, 1), (2, 3), ... An unpaired
    # trailing crossing in an open column is simply dropped.
    n_pairs = counts // 2
    keep = n_pairs > 0
    starts, n_pairs = starts[keep], n_pairs[keep]
    if len(n_pairs) == 0:
        return np.empty(0, dtype=np.int64)

    # Pair `p` of a column starting at `s` uses crossings s + 2p and s + 2p + 1.
    within = _expand_runs(np.zeros(len(n_pairs), dtype=np.int64), n_pairs)
    lower = np.repeat(starts, n_pairs) + 2 * within
    col_of = cols[lower]
    z0, z1 = zs[lower], zs[lower + 1]

    # A voxel is interior iff its centre (iz + 0.5 in cell units) lies inside the
    # span. Empty spans (thinner than one voxel) drop out via lengths <= 0.
    a = np.ceil(z0 - 0.5).astype(np.int64) - shift[2]
    b = np.floor(z1 - 0.5).astype(np.int64) - shift[2]
    lengths = b - a + 1
    valid = lengths > 0
    if not valid.any():
        return np.empty(0, dtype=np.int64)
    a, lengths, col_of = a[valid], lengths[valid], col_of[valid]

    # Expand in chunks so a huge solid never needs one giant temporary.
    out = []
    for start, end in _batch_bounds(lengths, _RUN_BUDGET):
        z = _expand_runs(a[start:end], lengths[start:end])
        col = np.repeat(col_of[start:end], lengths[start:end])
        # `col` is already (x << _BITS) | y in shifted coordinates, so shifting it
        # up by another _BITS and OR-ing z reproduces exactly pack()'s layout.
        out.append((col << _BITS) | z)
    keys = np.concatenate(out) if len(out) > 1 else out[0]
    log(f"voxelize: fill stage emitted {len(keys)} interior voxel(s).", verbose=verbose)
    return keys


def _z_crossings(verts, faces, shift):
    """Z coordinates where the mesh crosses each voxel column's centre line.

    Returns `(column_key, z)` with `column_key = (x << _BITS) | y` in shifted
    coordinates and `z` in unshifted cell units. As in the surface stage all
    coefficients are per-triangle, so the per-candidate cost is a gather.
    """
    cols_out, zs_out = [], []
    for tris in _iter_triangles(verts, faces):
        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
        n = np.cross(v1 - v0, v2 - v1)

        # A triangle parallel to the Z axis (n_z == 0) is edge-on to every column
        # and can only be hit on a measure-zero set - it contributes no crossing.
        keep = n[:, 2] != 0
        if not keep.any():
            continue
        v0, v1, v2, n = v0[keep], v1[keep], v2[keep], n[keep]

        # Orient the XY projection counter-clockwise so the edge functions are
        # positive inside. `n`/`v0` stay untouched for the plane solve below.
        flip = n[:, 2] < 0
        w0 = v0[:, :2]
        w1 = np.where(flip[:, None], v2[:, :2], v1[:, :2])
        w2 = np.where(flip[:, None], v1[:, :2], v2[:, :2])
        edges = _edge_functions(w0, w1, w2)

        # Plane solve reduced to z = zc + zx * qx + zy * qy, per triangle.
        inv = 1.0 / n[:, 2]
        zx = -n[:, 0] * inv
        zy = -n[:, 1] * inv
        zc = (n * v0).sum(axis=1) * inv
        del v0, v1, v2, inv, flip

        # Columns whose centre could fall inside: centre of column i is i + 0.5.
        mn = np.minimum(np.minimum(w0, w1), w2)
        mx = np.maximum(np.maximum(w0, w1), w2)
        del w0, w1, w2
        lo = np.ceil(mn - 0.5).astype(np.int64) - shift[:2]
        hi = np.floor(mx - 0.5).astype(np.int64) - shift[:2]
        del mn, mx
        dims = np.maximum(hi - lo + 1, 0)
        if not dims.prod(axis=1).any():
            continue

        # Expand to (triangle, x-row) pairs first, then batch those rows by their
        # y-length. Going straight to cells would give a single triangle with a
        # huge column box one enormous batch; this way peak memory is bounded no
        # matter how few or how large the triangles are.
        nx = dims[:, 0]
        row_tri = np.repeat(np.arange(len(nx), dtype=np.int64), nx)
        row_x = np.arange(len(row_tri), dtype=np.int64)
        row_x -= np.repeat(np.cumsum(nx) - nx, nx)
        row_x += lo[:, 0][row_tri]
        row_len = dims[:, 1][row_tri]

        for start, end in _batch_bounds(row_len, _CAND_BUDGET):
            rl = row_len[start:end]
            total = int(rl.sum())
            if total == 0:
                continue
            tri_id = np.repeat(row_tri[start:end], rl)
            cx = np.repeat(row_x[start:end], rl)
            cy = np.arange(total, dtype=np.int64)
            cy -= np.repeat(np.cumsum(rl) - rl, rl)
            cy += lo[:, 1][tri_id]

            qx = (cx + shift[0]).astype(np.float64) + 0.5
            qy = (cy + shift[1]).astype(np.float64) + 0.5
            inside = _point_in_triangle(qx, qy, edges, tri_id)
            if not inside.any():
                continue

            tri_id = tri_id[inside]
            cx, cy, qx, qy = cx[inside], cy[inside], qx[inside], qy[inside]
            del inside
            z = zc[tri_id] + zx[tri_id] * qx + zy[tri_id] * qy
            cols_out.append(_key2(cx, cy))
            zs_out.append(z)

    if not cols_out:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    return np.concatenate(cols_out), np.concatenate(zs_out)


def _key2(x, y):
    """2D column key, laid out so ``(_key2(x, y) << _BITS) | z == _key(x, y, z)``."""
    return (x << _BITS) | y


def _edge_functions(w0, w1, w2):
    """Per-triangle 2D edge functions ``E(q) = a * qx + b * qy + c``, plus a tie rule.

    Returns one ``(a, b, c, claims_ties)`` tuple per edge of the CCW-oriented
    projected triangle.

    Returns ``(fx, fy, base_x, base_y)`` per edge, defining

        E(q) = fx * (qy - base_y) - fy * (qx - base_x)

    which is positive on the triangle's interior side.

    Two properties matter, and both come from *how* this is arranged rather than
    from the formula itself.

    The edge is described by its two endpoints in canonical (lexicographic)
    order, with the triangle's own orientation folded into the sign of ``fx``/
    ``fy`` afterwards. Two triangles sharing an edge therefore run bitwise
    identical arithmetic on a bitwise identical base point and obtain exactly
    negated ``E``. Letting each triangle use its own first vertex as the base
    instead gives constants that are algebraically equal but not bitwise equal,
    so a point lying on the shared edge can round negative on *both* sides and be
    dropped by both - losing a crossing, flipping the column's parity and punching
    a hole clean through the solid. That fires as soon as a mesh has an
    axis-aligned face whose split diagonal passes through column centres, which is
    common rather than exotic.

    Keeping the form *relative to the base point*, rather than expanding it to
    ``a*qx + b*qy + c``, is what makes ``E`` come out as exactly zero at either
    endpoint: at the base both differences vanish, and at the far endpoint the
    expression reduces to ``fl(fx*fy) - fl(fy*fx)``. The expanded form is exact
    only at the base, so a point sitting on the *other* end of the edge - a shared
    mesh vertex - would not be recognized as a tie at all, and the perturbation
    rule in `_point_in_triangle` would never get to run.
    """
    out = []
    for p, q in ((w0, w1), (w1, w2), (w2, w0)):
        swap = (p[:, 0] > q[:, 0]) | ((p[:, 0] == q[:, 0]) & (p[:, 1] > q[:, 1]))
        lo = np.where(swap[:, None], q, p)
        hi = np.where(swap[:, None], p, q)
        # Multiplying by +-1.0 is exact in IEEE, so folding the orientation into
        # the edge vector preserves the exact-negation property.
        sgn = np.where(swap, -1.0, 1.0)
        out.append(
            (
                (hi[:, 0] - lo[:, 0]) * sgn,
                (hi[:, 1] - lo[:, 1]) * sgn,
                lo[:, 0],
                lo[:, 1],
            )
        )
    return out


def _point_in_triangle(qx, qy, edges, tri_id):
    """Point-in-triangle for CCW 2D triangles, ties broken by symbolic perturbation.

    Getting ties right is what makes the parity fill work at all: a column centre
    lying exactly on the boundary between triangles must be claimed by exactly
    one of them, or the column's crossing count changes parity and the solid is
    left with a hole (or a spurious plug) running right through it.

    Rather than a geometric fill rule, the query point is treated as nudged by an
    infinitesimal ``(eps, eps^2)``. Then ``E`` becomes ``E - fy*eps + fx*eps^2``
    and its sign for vanishing eps is just the first non-zero of
    ``(E, -fy, fx)`` - no actual epsilon, no tolerance, and the perturbed point is
    never exactly on any edge. Because `_edge_functions` gives two triangles
    sharing an edge exactly negated ``fx``/``fy`` and ``E``, this ordering is
    antisymmetric across that edge, so precisely one of them claims the point.

    The classic top-left rule is the usual answer here and it does handle a point
    on the interior of a shared edge, but it can hand a point sitting exactly on a
    shared *vertex* to two triangles of the surrounding fan (a fan edge lying
    along +X is enough to break it). Meshes put vertices on the axis all the time
    - a sphere's poles, a cone's apex - so that is a routine input, not a corner
    case. Perturbation is immune: an infinitesimally displaced point lies in the
    interior of exactly one wedge of the fan, whatever the fan looks like.
    """
    inside = None
    for fx, fy, base_x, base_y in edges:
        tfx, tfy = fx[tri_id], fy[tri_id]
        e = tfx * (qy - base_y[tri_id])
        e -= tfy * (qx - base_x[tri_id])
        # sign of E - fy*eps + fx*eps^2 as eps -> 0+
        ok = (e > 0) | ((e == 0) & ((tfy < 0) | ((tfy == 0) & (tfx > 0))))
        del e, tfx, tfy
        inside = ok if inside is None else (inside & ok)
    return inside
