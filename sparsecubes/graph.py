"""Voxel adjacency as an explicit edge list.

The one primitive here turns a sparse voxel cloud into a graph: which voxels
touch which, under 6-, 18- or 26-connectivity. Downstream that is what feeds
`centerline`, `teasar_skeletonize(edges=...)` and any conversion to networkx /
igraph - and it is what `downsample_graph` lifts through the block quotient.

The implementation walks *positive* packed-key deltas only. Under `pack`'s
layout a neighbour step is a constant addition on the key, and a positive delta
always lands on a key that sorts later, so every undirected edge is discovered
exactly once, already canonical (``lo < hi``) and already unique - no
symmetrisation, no dedup pass, and half the neighbour lookups of a full 26-way
scan. Each delta costs one ``(N,)`` searchsorted, so the transient never exceeds
the voxel count (a KD-tree ball query, or a dense ``(N, 26, 3)`` neighbour block,
would both be far heavier).
"""

import numpy as np

from ._keys import from_keys, key_deltas, to_keys
from ._sparse import sparse_aware

__all__ = ["edges"]


def _edge_pairs(keys, connectivity, sort=True):
    """Undirected adjacency of a *sorted, unique* packed-key array.

    Returns ``(E, 2)`` int64 index pairs into `keys`, canonical (``lo < hi``) and
    deduplicated by construction. `keys` must come from `to_keys` (or an
    equivalent ``min - margin`` shift) with at least a one-voxel margin, so that
    stepping a neighbour delta cannot carry between the packed fields.

    `sort` lexsorts the result for a deterministic edge order; callers that
    re-sort downstream anyway (see `downsample_graph`) can skip it.
    """
    m = len(keys)
    if m < 2:
        return np.empty((0, 2), dtype=np.int64)

    deltas = key_deltas(connectivity)
    parts = []
    for d in deltas[deltas > 0]:
        nb = keys + d
        pos = np.clip(np.searchsorted(keys, nb), 0, m - 1)
        hit = keys[pos] == nb
        if not hit.any():
            continue
        # `d > 0` and `keys` ascending => the neighbour sorts after the voxel, so
        # `src < dst` holds without an explicit min/max.
        parts.append(np.stack([np.flatnonzero(hit), pos[hit]], axis=1))

    if not parts:
        return np.empty((0, 2), dtype=np.int64)
    e = np.concatenate(parts).astype(np.int64, copy=False)
    return e[np.lexsort((e[:, 1], e[:, 0]))] if sort else e


@sparse_aware
def edges(voxels, connectivity=26):
    """Adjacency of a sparse voxel cloud, as an explicit edge list.

    The graph primitive underneath `centerline` and `downsample_graph`, exposed
    for callers that want the voxel graph itself - to hand to networkx/igraph, to
    inject via ``teasar_skeletonize(edges=...)``, or to run their own traversal.
    Stays sparse: cost and memory scale with the voxel count, never the bounding
    box, and no KD-tree is built.

    Parameters
    ----------
    voxels :        (N, 3) integer array of XYZ voxel coordinates. Deduplicated
                    internally.
    connectivity :  {6, 18, 26}, optional
                    Which neighbours count as adjacent: 6 = faces only, 18 =
                    faces + edges, 26 = faces + edges + corners (default).

    Returns
    -------
    nodes : (M, 3) array of the input dtype - the deduplicated voxels, sorted.
            ``M <= N``, and it is these rows the edge indices refer to, so keep
            them rather than re-using the input array.
    edges : (E, 2) int64 undirected edges as index pairs into `nodes`, canonical
            (``lo < hi``), deduplicated, sorted. Each undirected edge appears
            once - mirror the pairs yourself if you need both directions.

    Examples
    --------
    >>> import numpy as np, sparsecubes as sc
    >>> nodes, e = sc.edges(np.array([[0, 0, 0], [1, 0, 0], [5, 5, 5]]))
    >>> len(nodes), e
    (3, array([[0, 1]]))

    See Also
    --------
    downsample_graph : the same adjacency, lifted onto a coarser lattice.
    """
    keys, shift = to_keys(voxels, margin=1)
    return from_keys(keys, shift, voxels.dtype), _edge_pairs(keys, connectivity)
