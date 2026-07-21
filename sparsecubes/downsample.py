"""Connectivity-safe voxel downsampling (block-quotient graph coarsening).

Naive block-pooling ``unique(voxels // f)`` is monotone: 26-adjacent fine voxels
map to equal-or-26-adjacent coarse cells, so it never *disconnects* a component
but readily *fuses* separate ones - two structures a sub-``2 * f`` gap apart
collapse into touching coarse cells and merge. A plain coarse coordinate cloud
cannot avoid this: adjacency is implicit in the coordinates (cells at ``(0,0,0)``
and ``(1,0,0)`` are 26-adjacent by definition), so there is no way to say "these
coarse cells exist but are not connected".

`downsample_graph` therefore returns the coarse cells *plus an explicit edge
list* obtained by lifting the fine 26-connectivity graph through the block
quotient ``v -> v // f``: two coarse cells are joined if some fine pair mapping
to them was 26-adjacent. Adjacency is carried by the edges, not by coordinate
geometry, so no connection is introduced that did not exist in the fine set and
the connected-component partition is preserved exactly. Everything stays sparse
(no dense grid is ever allocated): the fine graph reuses the chunked
`teasar._neighbors_26`, and coarse-cell identity is assigned by `pack` + `unique`
(``O(N)`` in the voxel count, never ``O(volume)``).

The result feeds straight into `teasar_skeletonize(coords, edges=...)` (or its
``downsample=`` convenience) and `centerline(coords, edges=...)`, which consume
the explicit edges instead of re-deriving 26-adjacency - the whole point, since
re-deriving would reintroduce the very links this avoids.
"""

import numpy as np

from .core import pack, unpack, unique
from .graph import _edge_pairs
from .thinning import _validate, _check_extent
from ._sparse import sparse_aware

__all__ = ["downsample", "downsample_graph"]

# Reductions `downsample` can apply when pooling several fine values into one
# coarse cell. All are `np.ufunc.reduceat`-able over the run of rows belonging to
# a cell, so none of them needs the notoriously slow `ufunc.at` scatter.
_AGGS = {"max": np.maximum, "min": np.minimum, "sum": np.add, "mean": np.add}


def _as_factor(factor):
    """Normalise `factor` to a length-3 int64 array (isotropic int -> repeated)."""
    f = np.asarray(factor)
    if not np.issubdtype(f.dtype, np.integer):
        raise ValueError(f"factor must be integer, got {factor!r}")
    if f.ndim == 0:
        f = np.repeat(f, 3)
    if f.shape != (3,):
        raise ValueError(f"factor must be an int or length-3, got shape {f.shape}")
    if np.any(f < 1):
        raise ValueError(f"factor must be >= 1, got {factor!r}")
    return f.astype(np.int64)


def _pool_keys(voxels, factor):
    """Pool `voxels` into ``factor``-sized cells; return ``(cell_keys, shift)``.

    Cell identity is assigned by `pack` on the quotient ``v // factor`` - an
    exact integer hash, so cells are numbered in ``O(N)`` in the voxel count and
    never ``O(volume)``. Floor division is a correct quotient for negatives too.
    The extent check runs on the *coarse* cells, not the fine voxels: pooling is
    exactly what brings an over-wide volume back inside `pack`'s range, so it
    would be perverse to reject the input for a fine extent the caller is in the
    middle of removing.
    """
    cells = voxels.astype(np.int64, copy=False) // factor
    _check_extent(cells)
    shift = cells.min(axis=0)
    return pack(cells - shift), shift


def _aggregate(values, order, starts, agg):
    """Reduce `values` over the runs of equal cells given by `order`/`starts`."""
    try:
        ufunc = _AGGS[agg]
    except KeyError:
        raise ValueError(
            f"agg must be one of {sorted(_AGGS)}, got {agg!r}"
        ) from None

    vals = values[order]
    if agg in ("sum", "mean") and vals.dtype.kind in "bui":
        # Widen before accumulating: pooling a 3x3x3 block of uint8 intensities
        # overflows almost immediately otherwise, and `np.add` on bools is an OR,
        # not a count.
        vals = vals.astype(np.int64)
    out = ufunc.reduceat(vals, starts, axis=0)
    if agg == "mean":
        counts = np.diff(np.append(starts, len(order)))
        out = out / counts.reshape((-1,) + (1,) * (out.ndim - 1))
    return out


@sparse_aware(mirror=True)
def downsample(voxels, factor, values=None, agg="max"):
    """Downsample a sparse voxel cloud onto a coarser lattice.

    Pools voxels into ``factor``-sized cells: fine voxel ``v`` becomes coarse
    cell ``v // factor``, deduplicated. The sparse counterpart of zooming a dense
    grid down (`scipy.ndimage.zoom`) - no volume is ever allocated, so this works
    on clouds whose bounding box would not fit in memory.

    Adjacency is *not* preserved: two structures less than ``2 * factor`` apart
    can land in touching coarse cells and read as connected. When that matters -
    skeletonizing, counting components - use `downsample_graph`, which returns
    the coarse cells with an explicit edge list and cannot fuse them.

    Parameters
    ----------
    voxels :    (N, 3) integer voxel coordinates (XYZ).
    factor :    positive int, or length-3 of positive ints for anisotropic
                downsampling.
    values :    (N,) or (N, C) array, optional
                Per-voxel data (intensity, confidence, ...) to carry along,
                aligned with `voxels` row for row. Several fine voxels collapse
                into one coarse cell, so the values have to be *reduced* rather
                than merely re-indexed - hence `agg`. Without this the function
                returns coordinates only.
    agg :       {"max", "min", "mean", "sum"}, optional
                How to combine the values falling in one cell. "max" (the
                default) preserves peaks, which is usually what you want for
                intensity; "mean" smooths; "sum" conserves total mass. Integer
                values are accumulated in int64 for "sum"/"mean", and "mean"
                returns floats.

    Returns
    -------
    coarse_coords : (M, 3) coarse cell coordinates in the input dtype, sorted by
                    packed key, ``M <= N``.
    coarse_values : (M,) or (M, C), only when `values` was given.

    Notes
    -----
    The coarse grid is `factor` times coarser, so scale any `spacing` you carry
    alongside by `factor` - mirroring `downsample_graph` and the mesher's
    ``step_size``.

    Examples
    --------
    >>> import numpy as np, sparsecubes as sc
    >>> v = np.array([[0, 0, 0], [1, 0, 0], [4, 0, 0]])
    >>> sc.downsample(v, 2)
    array([[0, 0, 0],
           [2, 0, 0]])
    >>> sc.downsample(v, 2, values=np.array([1.0, 5.0, 3.0]), agg="max")[1]
    array([5., 3.])
    """
    _validate(voxels)
    dtype = voxels.dtype
    f = _as_factor(factor)

    if len(voxels) == 0:
        empty = np.empty((0, 3), dtype=dtype)
        if values is None:
            return empty
        values = np.asarray(values)
        return empty, values[:0]

    ckeys, shift = _pool_keys(voxels, f)
    if values is None:
        return (unpack(unique(ckeys)) + shift).astype(dtype)

    values = np.asarray(values)
    if len(values) != len(voxels):
        raise ValueError(
            f"values must be aligned with voxels: got {len(values)} values for "
            f"{len(voxels)} voxels."
        )

    # One sort serves both jobs: it groups the rows of each cell into a
    # contiguous run, so the unique cells and the per-cell reduction both fall
    # out of the same run boundaries.
    order = np.argsort(ckeys, kind="stable")
    ks = ckeys[order]
    starts = np.flatnonzero(np.concatenate(([True], ks[1:] != ks[:-1])))
    coarse = (unpack(ks[starts]) + shift).astype(dtype)
    return coarse, _aggregate(values, order, starts, agg)


@sparse_aware
def downsample_graph(voxels, factor):
    """Downsample a sparse voxel cloud without introducing new connections.

    Pools voxels into coarse ``factor``-sized cells but returns the adjacency
    *explicitly* (as an edge list lifted from the fine 26-connectivity graph)
    rather than leaving it implicit in the coarse coordinates. Two coarse cells
    are connected if some fine voxel in one was 26-adjacent to a fine voxel in
    the other, so the connected-component partition is preserved exactly and no
    spurious links are created (see the module docstring for why a plain coarse
    cloud cannot achieve this). Stays sparse - no dense volume is allocated.

    Parameters
    ----------
    voxels :    (N, 3) integer voxel coordinates (XYZ), as elsewhere in
                sparse-cubes. Deduplicated internally.
    factor :    positive int, or length-3 of positive ints for anisotropic
                downsampling. Fine voxel ``v`` maps to coarse cell ``v // factor``.

    Returns
    -------
    coarse_coords : (M, 3) coarse cell coordinates, in the input dtype, sorted by
                    packed key.
    coarse_edges :  (E, 2) int64 undirected edges as index pairs into
                    ``coarse_coords`` (canonical ``lo < hi``, deduplicated). These
                    define the connectivity - do **not** re-derive adjacency from
                    ``coarse_coords`` geometry, which would be a strict superset.

    Notes
    -----
    When `spacing` matters downstream, scale it by `factor` (the coarse grid is
    `factor` times coarser), mirroring the mesher's ``step_size`` handling. The
    ``downsample=`` argument of `teasar_skeletonize` does this for you.
    """
    _validate(voxels)
    dtype = voxels.dtype
    f = _as_factor(factor)

    vox = unique(voxels, axis=0).astype(np.int64)
    empty_edges = np.empty((0, 2), dtype=np.int64)
    if len(vox) == 0:
        return vox.astype(dtype), empty_edges

    # The fine graph packs fine coordinates, so the fine extent must fit pack()'s
    # budget - the same precondition `thin`/`teasar` already impose, so anything
    # skeletonizable is downsamplable.
    _check_extent(vox)

    # Fine 26-adjacency, as one edge per neighbouring pair. Node index i is row i
    # of `nodes`, i.e. `vox` in packed-key order.
    fine_shift = vox.min(axis=0) - 1
    fkeys = pack(vox - fine_shift)
    order = np.argsort(fkeys, kind="stable")
    nodes = vox[order]
    fine_edges = _edge_pairs(fkeys[order], 26, sort=False)

    # Assign each fine node a compact coarse-cell id without a dense grid: pool to
    # cells, pack to keys, and let `unique(return_inverse)` number the cells.
    ckeys, shift = _pool_keys(nodes, f)
    uniq, inv = np.unique(ckeys, return_inverse=True)
    inv = inv.reshape(-1)  # guard against numpy version inverse-shape quirks
    coarse_coords = (unpack(uniq) + shift).astype(dtype)
    n = len(coarse_coords)

    # Lift every fine edge to its coarse endpoints; drop self-edges (both ends in
    # the same cell - a legitimate intra-cell collapse, not a connection).
    a = inv[fine_edges[:, 0]]
    b = inv[fine_edges[:, 1]]
    keep = a != b
    a = a[keep]
    b = b[keep]
    if len(a) == 0:
        return coarse_coords, empty_edges

    # Canonicalise (lo < hi) and dedup via a single packed int64 per pair. With
    # n < 2**31 (guarded downstream by the same int32 node-index limit), the key
    # lo * n + hi stays below 2**62 and is exact in int64.
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    upair = np.unique(lo * n + hi)
    coarse_edges = np.stack([upair // n, upair % n], axis=1).astype(np.int64)
    return coarse_coords, coarse_edges
