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
from .thinning import _validate, _check_extent

__all__ = ["downsample_graph"]


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

    # Lazy import breaks the teasar <-> downsample module cycle.
    from .teasar import _neighbors_26

    nodes, src, dst, _ = _neighbors_26(vox)  # directed half-edges; weights unused

    # Assign each fine node a compact coarse-cell id without a dense grid: pool to
    # cells, pack to keys, and let `unique(return_inverse)` number the cells.
    cells = nodes // f  # floor division is a correct quotient for negatives too
    shift = cells.min(axis=0) - 1
    ckeys = pack(cells - shift)
    uniq, inv = np.unique(ckeys, return_inverse=True)
    inv = inv.reshape(-1)  # guard against numpy version inverse-shape quirks
    coarse_coords = (unpack(uniq) + shift).astype(dtype)
    n = len(coarse_coords)

    # Lift every fine edge to its coarse endpoints; drop self-edges (both ends in
    # the same cell - a legitimate intra-cell collapse, not a connection).
    a = inv[src]
    b = inv[dst]
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
