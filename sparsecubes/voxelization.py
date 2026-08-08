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

2. **Interior** (`_interior_keys`). Sparse scanline fill. Each triangle is
   rasterized in the XY projection over voxel *column centres*; every hit yields
   one Z crossing. Crossings are sorted per column and paired up, and the cells
   between a pair are emitted as runs. Memory is `O(#crossings + #output)`.
   The even-odd rule makes this independent of face winding (so meshes with
   inconsistent normals still work) and it naturally leaves enclosed cavities
   empty.

   Each crossing also carries the *sign* of the surface it crossed - one `int8`
   column and one `cumsum` - which pays for the `fill="winding"` rule and for the
   diagnostics (`_diagnose`). `axes=3` runs the whole fill once per axis and
   takes the majority, which is the only thing here that recovers a hole. See
   `voxelize`'s docstring for when to reach for either.

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
from collections import namedtuple

import numpy as np
import trimesh as tm

from .core import unpack, log, unique, _PACK_FIELD, _sorted_hit
from ._keys import unique_sorted

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


def voxelize(mesh, spacing=1.0, *, solid=True, fill="parity", axes=1, verbose=False):
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
                If True (default) fill the interior with a scanline test (see
                `fill`) and return surface + interior voxels. If False, return
                only the conservative surface shell.
    fill :      {"parity", "winding"}, optional
                Which rule decides what is inside, for ``solid=True``.

                ``"parity"`` (default) is the even-odd rule: crossings are paired
                up along the column. It ignores face winding entirely, so meshes
                with inconsistent normals fill correctly, and it leaves enclosed
                cavities empty. Its blind spot is anything that makes a column's
                crossings *not* alternate inside/outside: overlapping or nested
                shells come out with the overlap carved away as a void, and a
                duplicated face flips the column's parity and drills a hole.

                ``"winding"`` is the nonzero rule: fill wherever the running
                signed crossing count is nonzero. That unions overlapping and
                nested shells and shrugs off duplicated faces, at the price of
                needing consistent winding - with normals flipped at random it
                welds concave gaps shut.

                Neither rule recovers an actual hole; see `axes`. Whichever is
                selected, both are evaluated for the diagnostics, and a warning
                is emitted where they disagree.
    axes :      {1, 3}, optional
                How many directions to sweep when filling. ``1`` (default) fills
                along Z only. ``3`` fills along X, Y and Z independently and
                keeps the voxels at least two sweeps claim. A defect that breaks
                the columns along one axis usually leaves the other two intact,
                so the majority repairs holes that no single sweep can - at three
                times the cost of the fill stage.
    verbose :   bool, optional. Log progress.

    Returns
    -------
    (N, 3) array of XYZ voxel indices, sorted, deduplicated. Indices are
    absolute and may be negative (dtype is ``int32`` when the values fit,
    otherwise ``int64``).

    Warns
    -----
    Once if some columns do not close (the mesh is not watertight there), with an
    upper bound on how much of the solid that could cost, and once if the parity
    and winding rules disagree (self-intersection, nesting, duplicated faces, or
    inconsistent winding). Neither warning fires on a clean mesh.

    Notes
    -----
    The per-axis extent is limited to `pack`'s 21-bit fields (~2.1e6 voxels along
    any one axis); coarsen `spacing` for very large meshes.

    Only vertices actually referenced by `faces` constrain the grid, so a stray
    unused vertex far from the mesh costs nothing.

    See Also
    --------
    sparsecubes.mesh :          The inverse operation.
    sparsecubes.binary.fill_cavities : Fill enclosed voids in an existing voxel set.
    sparsecubes.binary.make_manifold : Seal the edge/corner-only voxel contacts a
                                partial fill leaves behind, which would otherwise
                                make `mesh`'s output non-manifold.
    """
    vertices, faces = _as_mesh(mesh)
    if fill not in ("parity", "winding"):
        raise ValueError(f"`fill` must be 'parity' or 'winding', got {fill!r}.")
    if axes not in (1, 3):
        raise ValueError(f"`axes` must be 1 or 3, got {axes!r}.")

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

    # Only *referenced* vertices constrain the grid. An unused vertex parked far
    # from the mesh (common in hand-edited or partially-decimated files) would
    # otherwise inflate the extent and trip the range guard below on a mesh that
    # fits comfortably. The `.all()` short-circuit keeps the usual case copy-free.
    used = np.zeros(len(verts), dtype=bool)
    used[faces] = True
    ref = verts if used.all() else verts[used]
    del used
    if not np.all(np.isfinite(ref)):
        raise ValueError("Mesh contains non-finite vertices.")

    imin = np.floor(ref.min(axis=0)).astype(np.int64)
    imax = np.floor(ref.max(axis=0)).astype(np.int64)
    del ref
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
        interior, diag = _fill_interior(verts, faces, shift, fill, axes, verbose)
        # The counts behind the warnings, on the channel the rest of the library
        # uses - so a caller can see them without scraping a warning string.
        log(
            f"voxelize: fill({fill}) over {diag.columns} column(s): {diag.open_} "
            f"open ({diag.odd} odd), {diag.ambiguous} rule-ambiguous.",
            verbose=verbose,
        )
        _warn_broken(diag, fill, axes, len(interior))
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
    """Normalize a `trimesh.Trimesh` or `(vertices, faces)` pair, and vet it.

    The one seam mesh input comes through, so it is where every check belongs.
    Beyond shape, `faces` has to be checked against `len(vertices)`: a stale index
    would otherwise surface as numpy's bare `IndexError` from deep inside the
    surface stage, and a *negative* one would not raise at all - it wraps around
    and silently voxelizes a triangle the file never described.
    """
    if isinstance(mesh, tm.Trimesh):
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    elif isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        vertices = np.asarray(mesh[0], dtype=np.float64)
        faces = np.asarray(mesh[1])
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise TypeError("`vertices` must be a (V, 3) array.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise TypeError("`faces` must be a (F, 3) array of triangles.")
    else:
        raise TypeError(
            "Expected a trimesh.Trimesh or a (vertices, faces) tuple, "
            f"got {type(mesh).__name__}."
        )

    if faces.size:
        if not np.issubdtype(faces.dtype, np.integer):
            raise TypeError(f"`faces` must have an integer dtype, got {faces.dtype}.")
        lo, hi = int(faces.min()), int(faces.max())
        if lo < 0 or hi >= len(vertices):
            raise ValueError(
                f"`faces` indexes vertices [{lo}, {hi}], outside the {len(vertices)} "
                "vertices provided (negative indices are rejected, not wrapped)."
            )
    return vertices, faces


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


# What the fill stage learned about the mesh on the way past. `open_` and `odd`
# count columns whose crossings do not describe a closed solid; `ambiguous`
# counts those where the two fill rules disagree.
_Diagnostics = namedtuple("_Diagnostics", "columns open_ odd ambiguous")
_CLEAN = _Diagnostics(0, 0, 0, 0)


def _fill_interior(verts, faces, shift, rule, axes, verbose):
    """Interior voxels: one sweep along Z, or a majority vote over all three axes."""
    if axes == 1:
        return _interior_keys(verts, faces, shift, rule, verbose)

    # `_PROJECTIONS` is cyclic, so its three permutations put each axis in the
    # sweep (Z) slot exactly once. Permuting the *keys* back afterwards beats
    # unpacking to coordinates: it is three shift-and-mask passes over one int64
    # column instead of an (N, 3) temporary per sweep.
    swept, diags = [], []
    for perm in _PROJECTIONS:
        keys, diag = _interior_keys(
            verts[:, perm], faces, shift[list(perm)], rule, verbose
        )
        # A sweep emits its keys in (column, z) order, so the identity permutation
        # is already sorted; the other two shuffle the fields and are not.
        keys = _permute_key(keys, perm)
        swept.append(unique_sorted(keys) if perm == _PROJECTIONS[0] else unique(keys))
        diags.append(diag)
    sizes = [len(s) for s in swept]
    keys = _majority(swept)
    log(f"voxelize: 3-axis vote over {sizes} -> {len(keys)} voxel(s).", verbose=verbose)
    return keys, _Diagnostics(*map(sum, zip(*diags)))


def _permute_key(keys, perm):
    """Reinterpret keys packed in `perm` axis order as keys in canonical XYZ order.

    Field `j` of the input holds the coordinate of original axis ``perm[j]``, so
    the fix-up is a pure field shuffle - no unpack/repack, and the shifts are
    identical either way because `voxelize` sizes its range guard on the largest
    axis.
    """
    if tuple(perm) == (0, 1, 2) or len(keys) == 0:
        return keys
    mask = _PACK_FIELD - 1
    out = np.zeros_like(keys)
    for j, axis in enumerate(perm):
        out |= ((keys >> ((2 - j) * _BITS)) & mask) << ((2 - axis) * _BITS)
    return out


def _majority(sets):
    """Keys claimed by at least two of three sorted, deduplicated key arrays.

    ``(A&B) | (A&C) | (B&C)``, evaluated as membership tests rather than by
    sorting the concatenation. **Consumes `sets`**: the three sweeps are each the
    size of the solid, and dropping the caller's references before the merge is
    what keeps this from peaking at five copies of it.

    Both selections preserve their (sorted) input's order, so the merge only has
    to interleave two runs.
    """
    a, b, c = sets
    del sets[:]  # `a`/`b`/`c` are now the only references
    both = b[_sorted_hit(b, c)]
    kept = a[_sorted_hit(a, b) | _sorted_hit(a, c)]
    del a, b, c
    return unique_sorted(kept, both)


def _interior_keys(verts, faces, shift, rule, verbose):
    """Sparse scanline fill -> (packed keys of interior voxels, `_Diagnostics`)."""
    cols, zs, sgn = _z_crossings(verts, faces, shift)
    if len(cols) == 0:
        return np.empty(0, dtype=np.int64), _CLEAN

    order = np.lexsort((zs, cols))
    cols, zs, sgn = cols[order], zs[order], sgn[order]
    del order

    # Column boundaries in the sorted crossing list. `same` does double duty: the
    # column starts are where it is False, and it is also the span mask below.
    same = cols[1:] == cols[:-1]
    starts = np.flatnonzero(np.concatenate([[True], ~same]))
    counts = np.diff(np.concatenate([starts, [len(cols)]]))

    # Winding number just *above* each crossing, within its own column: the net
    # signed crossing count so far. One cumsum, minus the running total the
    # column started from. int32 is ample - the running sum is bounded by the
    # crossing count - and halves the two per-crossing arrays that set peak here.
    running = np.cumsum(sgn, dtype=np.int32)
    base = np.concatenate([[0], running[starts[1:] - 1]])
    winding = running - np.repeat(base, counts)
    del running, base, sgn

    # Span `i` runs from crossing `i` up to crossing `i + 1` of the same column;
    # the last crossing of a column starts no span.
    span = np.concatenate([same, [False]])
    del same
    # Even-odd pairs crossings (0, 1), (2, 3), ...: fill from every even-indexed
    # crossing. An unpaired trailing crossing starts no span and is dropped.
    within = np.arange(len(cols), dtype=np.int64) - np.repeat(starts, counts)
    parity_fill = span & (within & 1 == 0)  # `& 1`: `% 2` is division-based
    del within
    winding_fill = span & (winding != 0)
    fill = winding_fill if rule == "winding" else parity_fill

    diag = _diagnose(starts, counts, winding, parity_fill, winding_fill)
    del span, winding, parity_fill, winding_fill

    lower = np.flatnonzero(fill)
    del fill
    if len(lower) == 0:
        return np.empty(0, dtype=np.int64), diag
    col_of = cols[lower]
    z0, z1 = zs[lower], zs[lower + 1]
    del lower, cols, zs

    # A voxel is interior iff its centre (iz + 0.5 in cell units) lies inside the
    # span. Empty spans (thinner than one voxel) drop out via lengths <= 0.
    a = np.ceil(z0 - 0.5).astype(np.int64) - shift[2]
    b = np.floor(z1 - 0.5).astype(np.int64) - shift[2]
    lengths = b - a + 1
    valid = lengths > 0
    if not valid.any():
        return np.empty(0, dtype=np.int64), diag
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
    return keys, diag


def _diagnose(starts, counts, winding, parity_fill, winding_fill):
    """Summarize what the crossing list says about the mesh's integrity.

    All of it is a handful of passes over the *crossings*, which scale with
    surface area - cheap next to the fill they accompany, which scales with
    volume. Two independent signals:

    - A column whose winding does not return to zero has an unmatched surface
      somewhere along it: the mesh is open there. This subsumes the odd-crossing
      test (odd count implies odd, hence nonzero, sum) and additionally catches
      the even-count breakages an odd test cannot see, such as a duplicated face.
    - The two fill rules disagreeing means the crossings do not simply alternate
      inside/outside. That is the fingerprint of a self-intersecting, nested or
      duplicated surface - the failure the parity fill would otherwise commit
      silently, handing back a plausible solid with the overlap carved out.
    """
    open_cols = winding[starts + counts - 1] != 0
    # On a clean mesh the two rules agree everywhere, and `any` bails on the
    # first crossing that differs - so the segmented reduce, which would walk
    # every column, only runs when there is actually something to report.
    differs = parity_fill != winding_fill
    ambiguous = (
        int(np.bitwise_or.reduceat(differs, starts).sum()) if differs.any() else 0
    )
    return _Diagnostics(
        columns=len(starts),
        open_=int(open_cols.sum()),
        odd=int((counts & 1).sum()),
        ambiguous=ambiguous,
    )


def _warn_broken(diag, rule, axes, n_filled):
    """Turn the fill diagnostics into at most two actionable warnings."""
    sweeps = "" if axes == 1 else " (summed over 3 sweeps)"
    if diag.open_:
        # What a broken column *would* have filled is unknowable - the surface
        # that closes it is simply absent - so scale by the mean depth of the
        # sound columns. That runs low where the damage sits on the thickest part
        # of the object, but an order of magnitude is what a user deciding
        # whether to trust the result needs, and a column count alone gives none.
        sound = diag.columns - diag.open_
        if sound > 0:
            missing = round(diag.open_ * n_filled / sound)
            share = 100.0 * missing / max(missing + n_filled, 1)
            scale = (
                f"At the mean depth of the {sound} sound column(s) that is roughly "
                f"{missing} voxel(s) missing, ~{share:.0f}% of the solid."
            )
        else:
            scale = "No column closed, so nothing was filled."
        hint = "Repair the mesh (e.g. trimesh's `fill_holes`)"
        if axes == 1:
            hint += ", pass `axes=3` to recover the columns from the other two sweeps,"
        warnings.warn(
            f"{diag.open_} of {diag.columns} voxel columns{sweeps} do not close "
            f"({diag.odd} of them from an odd crossing count), so the mesh is not "
            f"watertight along them. They were left unfilled. {scale} "
            f"{hint} or use `solid=False`.",
            stacklevel=3,
        )
    if diag.ambiguous:
        consequence = (
            "`fill='winding'` unions them, but needs consistent winding; "
            "cross-check against `fill='parity'`."
            if rule == "winding"
            else "`fill='parity'` carves the overlaps out as enclosed voids; "
            "`fill='winding'` unions them instead, or fill them afterwards with "
            "`sparsecubes.binary.fill_cavities`."
        )
        warnings.warn(
            f"{diag.ambiguous} of {diag.columns} voxel columns{sweeps} where the "
            "even-odd and winding rules disagree: the mesh self-intersects, has "
            f"nested or duplicated surfaces, or has inconsistent face winding. "
            f"{consequence}",
            stacklevel=3,
        )


def _z_crossings(verts, faces, shift):
    """Z coordinates where the mesh crosses each voxel column's centre line.

    Returns `(column_key, z, sign)` with `column_key = (x << _BITS) | y` in
    shifted coordinates and `z` in unshifted cell units. As in the surface stage
    all coefficients are per-triangle, so the per-candidate cost is a gather.

    `sign` is +1 where the surface faces along +Z and -1 where it faces along -Z,
    i.e. the direction the column passes through it. Summing it gives the winding
    number the fill and the diagnostics both run on; it costs one int8 column.
    """
    cols_out, zs_out, sg_out = [], [], []
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
        # The same test is the crossing direction, so the sign comes for free.
        flip = n[:, 2] < 0
        sign = np.where(flip, np.int8(-1), np.int8(1))
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
            sg_out.append(sign[tri_id])

    if not cols_out:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int8),
        )
    return np.concatenate(cols_out), np.concatenate(zs_out), np.concatenate(sg_out)


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

    One caveat on the tie-break, as opposed to the crossing *count*: the exact
    negation assumes the two triangles traverse their shared edge in opposite
    directions, which is what a consistently wound mesh does. Where the winding
    is inconsistent they traverse it the same way, both get the same sign, and a
    point exactly on that edge is claimed twice or not at all. So the even-odd
    fill's winding-independence is exact for the pairing and merely
    near-exact at the ties - measurably so: flipping half a torus's faces moves a
    couple of voxels out of ~19k. Everything else about it is winding-free.
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
