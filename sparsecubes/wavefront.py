"""Wavefront (Reeb-graph) skeletonization straight on sparse voxels.

A geodesic wave is propagated across the voxel graph from a seed. Voxels reached
at the same distance form a *level set*; each connected group within a level set
is a "ring" (a cross-section of the object) which collapses to its centroid -
one skeleton node. Contracting the voxel graph onto those groups yields the
skeleton edges. This is the Reeb graph of the geodesic distance function
(Verroust & Lazarus 2000), and is the algorithm behind `skeletor`'s
``by_wavefront``, ported from mesh surfaces onto the voxel graph.

Why it suits sparse voxels: the whole pipeline is one Dijkstra field plus two
connected-components-style passes - no distance transform, no iterative thinning,
no per-path search. With `dijkstra3d_sparse` installed all four run straight over
the coordinates and *no edge list is ever built*, so cost scales with the voxel
count rather than with the ~11x larger 26-connectivity adjacency (5.56M voxels:
10.6 s / 1.7 GB, against 63M edges and 5.1 GB for the scipy fallback). Without it
the fallback materialises that edge list and uses scipy's csgraph; the two paths
are checked against each other in `tests/test_wavefront.py::test_backends_agree`.

The trade-off versus `teasar_skeletonize` is that the wave, not a medial-axis
penalty field, decides where nodes land: level sets are cross-sections *of the
wave*, so they sit perpendicular to the direction of travel rather than to the
local tube axis. Mid-tube that is very accurate (0.1 voxels off the axis of an
r=8 cylinder), but for roughly the first radius' worth of wave the level sets
are still spherical caps around the seed rather than cross-sections, so the
first few nodes bulge off-axis. See `benchmarks/bench_wavefront.py`.

Two things in `skeletor`'s mesh version deliberately did not come across:

- Its ``waves`` parameter (several waves from random seeds, node positions
  averaged) does not port. Averaging gives each *voxel* an almost unique mean
  position, and the nodes are the distinct positions - on a cylinder that turns
  68 nodes into 2987 without improving centring. Here the seed is instead pinned
  at a geodesic tip (`_seeds`), which removes the seed dependence it was
  compensating for.
- Its unweighted hop-count distance is replaced by a metric geodesic field, so
  level sets are not biased towards the lattice diagonals.
"""

import numpy as np

from ._keys import from_keys, key_deltas, sorted_hit, to_keys
from ._sparse import sparse_aware
from .core import log
from .graph import _edge_pairs
from .skeleton import Skeleton, _as_spacing, _prune_spurs
from .thinning import _validate

try:
    import dijkstra3d_sparse as _dijkstra3d_sparse

    # The edge-list-free path needs `connected_components(group=)` and
    # `label_adjacency`, both new in dijkstra3d-sparse 0.1.1. Older versions are
    # fine for the rest of the library, so degrade to scipy here rather than
    # failing the import - `label_adjacency` is the marker for both.
    if not hasattr(_dijkstra3d_sparse.Graph, "label_adjacency"):
        _dijkstra3d_sparse = None
except ImportError:
    _dijkstra3d_sparse = None

__all__ = ["wavefront_skeletonize"]

# Not a parameter, unlike everywhere else in the library. Lower connectivities
# make the wave's own distances degenerate: with 6-connectivity every step costs
# the same, so the graph is bipartite by distance parity and *no* two adjacent
# voxels ever land in the same level - every voxel becomes its own ring (5.5M
# nodes on the neuron fixture instead of 30k). 18 is less extreme but still
# tilts rings towards the lattice. It also matches the rest of the library's
# skeletonizers, and is what `dijkstra3d_sparse` derives internally.
_CONNECTIVITY = 26

# "volume" is not an aggregation of the ring's spread like the others - it is
# derived from the voxel count instead, and is computed once the final edges are
# known. See `_volume_radii`.
_RADIUS_AGGS = frozenset(
    {"volume", "mean", "median", "max", "min", "percentile25", "percentile75"}
)


# ---------------------------------------------------------------------------
# geodesic field
# ---------------------------------------------------------------------------


def _shell_keys(keys):
    """Keep only the voxels with at least one exposed face - the object's shell.

    Same test as one 6-connected erosion (`binary._erode_keys`), kept on keys so
    nothing is unpacked and re-packed in between.
    """
    from .binary import _erode_keys

    interior = _erode_keys(keys, key_deltas(6))
    return keys[~sorted_hit(keys, interior)]


def _edge_lengths(nodes, edges, spacing):
    """Euclidean length of every edge, in physical units when `spacing` is set."""
    d = (nodes[edges[:, 1]] - nodes[edges[:, 0]]).astype(float)
    if spacing is not None:
        d = d * spacing
    return np.linalg.norm(d, axis=1)


def _longest_step(spacing):
    """Length of the longest edge in the 26-graph - the level-set floor.

    A level thinner than one voxel step lets a *single* edge jump a whole level,
    which contracts into an edge between non-adjacent rings. Those shortcuts run
    parallel to the two-hop path through the level in between, so every one of
    them closes a triangle: at ``step_size=1`` a plain r=8 cylinder comes out with
    65 spurious cycles, at ``sqrt(3)`` with none. Bounding `step_size` from below
    by the longest possible step is what makes the contraction a Reeb graph, so
    the only cycles left are the object's own.
    """
    s = np.ones(3) if spacing is None else np.asarray(spacing, dtype=float)
    s = np.abs(s)
    # The 1e-6 is not a fudge factor for the geometry: the guarantee is that two
    # adjacent voxels' geodesic distances differ by at most one edge length, and
    # Dijkstra sums that edge length in floating point, so the difference lands a
    # few ulps *over* the exact bound and a bin boundary falling inside those ulps
    # would let the pair skip a level after all. Covers rounding on geodesics up
    # to ~1e10 units long; the packed-key range caps them far below that.
    return float(np.linalg.norm(s)) * (1.0 + 1e-6)


class _ScipyGraph:
    """Fallback backend: an explicit edge list plus scipy's csgraph routines.

    Materialises the full 26-connectivity edge list (63M rows on the 5.5M-voxel
    neuron) because every operation here needs it. `_SparseGraph` is the same
    interface without that array; see the module docstring.
    """

    def __init__(self, nodes, edges, spacing):
        from scipy.sparse import csr_matrix

        w = _edge_lengths(nodes, edges, spacing)
        self._edges, self._m = edges, len(nodes)
        # Symmetrise explicitly: `dijkstra(directed=False)` would do it per call.
        src = np.concatenate([edges[:, 0], edges[:, 1]])
        dst = np.concatenate([edges[:, 1], edges[:, 0]])
        self._g = csr_matrix((np.concatenate([w, w]), (src, dst)), shape=(self._m, self._m))

    def __call__(self, sources):
        from scipy.sparse.csgraph import dijkstra

        # min_only collapses the multi-source run to one (N,) field. Every
        # component holds exactly one source, so "distance to the nearest source"
        # is "distance to my component's source" - all components in one pass.
        return dijkstra(self._g, directed=False, indices=sources, min_only=True)

    def components(self):
        return _components(self._edges, self._m)

    def rings(self, lvl):
        e = self._edges
        return _components(e[lvl[e[:, 0]] == lvl[e[:, 1]]], self._m)

    def contract(self, labels, n_labels):
        return _contract(labels, self._edges, n_labels)


class _SparseGraph:
    """Preferred backend: every operation runs straight over the coordinates.

    All four calls share one `dijkstra3d_sparse.Graph`, so the spatial index is
    built once and *no edge list is ever materialised* - the whole reason this
    backend exists. `rings` is a `connected_components` constrained to union only
    within a level (`group=`), and `contract` is `label_adjacency`, which turns
    63M adjacencies into 30k label pairs while deduplicating as it goes.
    """

    def __init__(self, nodes, spacing):
        self._aniso = (
            tuple(float(s) for s in spacing) if spacing is not None else (1.0, 1.0, 1.0)
        )
        self._g = _dijkstra3d_sparse.Graph(nodes, index_kind="hash")

    def __call__(self, sources):
        dist, _ = self._g.dijkstra_field(
            sources, cost_mode="geometric", anisotropy=self._aniso,
            connectivity=_CONNECTIVITY, min_only=True,
        )
        return dist

    def components(self):
        return self._g.connected_components(connectivity=_CONNECTIVITY)

    def rings(self, lvl):
        return self._g.connected_components(group=lvl, connectivity=_CONNECTIVITY)

    def contract(self, labels, n_labels):
        return self._g.label_adjacency(labels, connectivity=_CONNECTIVITY)


def _first_per_label(labels, order, n_labels):
    """Row index of the first entry of each label under the ordering `order`."""
    out = np.full(n_labels, -1, dtype=np.int64)
    # Reverse assignment: later writes win, so walking `order` backwards leaves
    # each label holding its *first* row. One vectorised scatter, no groupby.
    out[labels[order][::-1]] = order[::-1]
    return out


def _seeds(labels, n_comp, field):
    """One seed row index per connected component: its geodesic-farthest voxel.

    Found by the standard double sweep - a field from an arbitrary voxel, then
    the argmax per component. Seeding at a tip makes the wave travel *along* the
    structure, so its level sets cut across it; seeding mid-object splits the
    wave in two and doubles every node between the seed and the nearer end.
    `skeletor` seeds randomly instead and averages several waves out of it, which
    does not port: see the module docstring.
    """
    arbitrary = _first_per_label(labels, np.arange(len(labels)), n_comp)
    d0 = field(arbitrary[arbitrary >= 0].tolist())
    d0 = np.where(np.isfinite(d0), d0, -1.0)
    # Farthest voxel per component = first row when sorted by (label, -distance).
    order = np.lexsort((-d0, labels))
    far = _first_per_label(labels, order, n_comp)
    return far[far >= 0]


# ---------------------------------------------------------------------------
# ring collapse
# ---------------------------------------------------------------------------


def _components(edges, m):
    """Connected components of the voxel graph, from the edge list."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    g = coo_matrix(
        (np.ones(len(edges), bool), (edges[:, 0], edges[:, 1])), shape=(m, m)
    )
    return connected_components(g, directed=False)


def _rings(lvl, edges, m):
    """Label every voxel by the ring (level-set component) it belongs to.

    One global connected-components call on the sub-graph of *intra-level* edges
    does all levels at once: a path between two voxels of the same level that
    stays within that level is exactly what makes them one ring, and no such path
    can leave the level, so rings never merge across levels. `skeletor` builds a
    subgraph per level instead - that loop is the bulk of its runtime.
    """
    return _components(edges[lvl[edges[:, 0]] == lvl[edges[:, 1]]], m)


def _agg(values, labels, n_labels, how):
    """Aggregate `values` within each label group."""
    if how == "mean":
        counts = np.bincount(labels, minlength=n_labels)
        return np.bincount(labels, weights=values, minlength=n_labels) / np.maximum(counts, 1)
    if how == "max":
        out = np.zeros(n_labels)
        np.maximum.at(out, labels, values)
        return out
    if how == "min":
        out = np.full(n_labels, np.inf)
        np.minimum.at(out, labels, values)
        return np.where(np.isfinite(out), out, 0.0)
    # median / percentile: sort once by (label, value), then index within group.
    q = {"median": 50.0, "percentile25": 25.0, "percentile75": 75.0}[how]
    order = np.lexsort((values, labels))
    counts = np.bincount(labels, minlength=n_labels)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pick = start + np.floor((counts - 1) * q / 100.0).astype(np.int64)
    out = np.zeros(n_labels)
    nonempty = counts > 0
    out[nonempty] = values[order[pick[nonempty]]]
    return out


def _volume_radii(nodes, edges, counts, spacing):
    """Radius of the tube that reproduces each node's own share of the object.

    Every voxel collapses onto exactly one node, so a node's territory has an
    exactly known volume ``V``. Modelling it as a length of tube - ``L`` being the
    node's share of the skeleton path, half of each incident edge - gives
    ``r = sqrt(V / (pi * L))``.

    Unlike the spread-based aggregations this is *volume-preserving by
    construction*: sweeping the radius along the skeleton recovers the object's
    volume instead of over- or under-shooting it by a shape factor. The spread of
    a ring about its centroid depends on the ring's shape (2/3 R for a filled
    disc under "mean", R for "max", more for an oblique or branching ring), which
    is exactly the bias measured in `benchmarks/bench_wavefront.py`.

    A node with no incident edges - a whole component collapsed to a point - has
    no length to sweep, so it falls back to the sphere of the same volume.
    """
    cell = float(np.prod(spacing)) if spacing is not None else 1.0
    vol = counts.astype(float) * cell

    d = (nodes[edges[:, 1]] - nodes[edges[:, 0]]).astype(float)
    if spacing is not None:
        d = d * spacing
    seg = np.linalg.norm(d, axis=1)
    length = np.zeros(len(nodes))
    np.add.at(length, edges[:, 0], seg)
    np.add.at(length, edges[:, 1], seg)
    length *= 0.5  # each edge is shared by the two nodes it joins

    r = np.zeros(len(nodes))
    tube = length > 0
    r[tube] = np.sqrt(vol[tube] / (np.pi * length[tube]))
    r[~tube] = np.cbrt(3.0 * vol[~tube] / (4.0 * np.pi))
    return r


def _centroids(pts, labels, n_labels):
    """Mean position of each label group - the collapsed ring centres."""
    counts = np.maximum(np.bincount(labels, minlength=n_labels), 1)[:, None]
    sums = np.stack(
        [np.bincount(labels, weights=pts[:, k], minlength=n_labels) for k in range(3)],
        axis=1,
    )
    return sums / counts


def _contract(labels, edges, n_labels):
    """Skeleton edges: voxel edges lifted onto ring labels, deduplicated.

    Self-loops (both ends in the same ring) drop out. Packing the pair into one
    int64 keeps the dedup a 1-D `unique`, which matters because the voxel edge
    list is the largest array in the pipeline.
    """
    # The voxel edge list is the biggest array in the pipeline (63M rows on the
    # 5.5M-voxel neuron), so drop the intra-ring edges *before* widening to the
    # int64 the packing needs, and write the min/max back over their inputs.
    a, b = labels[edges[:, 0]], labels[edges[:, 1]]
    keep = a != b
    if not keep.any():
        return np.empty((0, 2), dtype=np.int64)
    a, b = a[keep].astype(np.int64), b[keep].astype(np.int64)
    del keep
    lo = np.minimum(a, b)  # `out=a` here would clobber the max's own input
    hi = np.maximum(a, b, out=b)
    packed = np.unique(lo * n_labels + hi)
    return np.stack([packed // n_labels, packed % n_labels], axis=1)


# ---------------------------------------------------------------------------
# cycle breaking
# ---------------------------------------------------------------------------


def _dotprops(pts, k=20):
    """Local straightness (`alpha`) from each node's k nearest neighbours.

    ``alpha`` is near 1 where the neighbourhood is collinear (a backbone running
    through) and near 0 where it fans out. `skeletor` uses it to bias the
    spanning tree towards cutting cycles at thin, off-axis branches.
    """
    from scipy.spatial import cKDTree

    k = min(len(pts), k)
    _, ix = cKDTree(pts).query(pts, k=k, workers=-1)
    nb = pts[ix.reshape(len(pts), k)]
    cpt = nb - nb.mean(axis=1, keepdims=True)
    s = np.linalg.svd(cpt.transpose(0, 2, 1) @ cpt, compute_uv=False)
    total = np.sum(s, axis=1)
    return np.where(total > 0, (s[:, 0] - s[:, 1]) / np.maximum(total, 1e-12), 0.0)


def _spanning_forest(centers, edges, radii, preserve_backbone):
    """Break cycles with a maximum spanning forest.

    The Reeb graph has a cycle wherever the wave split and re-merged - around a
    genuine hole in the object, but also around any self-touch. Weighting edges
    by radius (times local straightness) and *maximising* means the cut lands on
    the thinnest, most off-axis edge of each cycle, leaving the backbone intact.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    if len(edges) == 0:
        return edges

    w = radii[edges].mean(axis=1)
    if preserve_backbone:
        w = w * _dotprops(centers)[edges].mean(axis=1)
    pos = w[w > 0]
    w = np.where(w > 0, w, (pos.min() / 2) if len(pos) else 1.0)

    m = len(centers)
    # Maximise w  ==  minimise 1/w (all weights positive).
    g = csr_matrix((1.0 / w, (edges[:, 0], edges[:, 1])), shape=(m, m))
    mst = minimum_spanning_tree(g).tocoo()
    e = np.stack([mst.row, mst.col], axis=1).astype(np.int64)
    return e[np.lexsort((e[:, 1], e[:, 0]))]


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


@sparse_aware
def wavefront_skeletonize(
    voxels,
    *,
    spacing=None,
    step_size=None,
    surface=False,
    radius_agg="volume",
    tree=True,
    preserve_backbone=True,
    min_branch_length=0,
    verbose=False,
):
    """Skeletonize a sparse voxel set by propagating a geodesic wavefront.

    Rings of voxels reached by the wave at the same distance are collapsed to
    their centroid; the contracted voxel graph is the skeleton. Unlike the other
    two backends this needs neither a thinning pass nor a distance transform -
    one Dijkstra field and one connected-components call over the voxel graph do
    the work - and node positions are sub-voxel (ring centroids), not lattice
    points. Requires scipy; uses `dijkstra3d_sparse` for the field when installed.

    Parameters
    ----------
    voxels :            (N, 3) integer voxel coordinates (XYZ), as elsewhere in
                        sparse-cubes.
    spacing :           length-3, optional. Physical voxel spacing (anisotropy).
                        Applied to the geodesic field, `step_size`, radii and
                        `min_branch_length`; `Skeleton.vertices` is in index
                        space scaled by it, as for the other backends.
    step_size :         float, optional. Width of a level set, in voxels (or
                        physical units when `spacing` is set) - i.e. the spacing
                        of the skeleton nodes along the structure. Defaults to
                        the longest step the graph allows (``sqrt(3)``), which is
                        the smallest value that keeps the contraction a Reeb
                        graph; anything below it manufactures cycles (see
                        `_longest_step`) and is rejected. Larger bins adjacent
                        rings together, trading resolution for a smoother
                        centerline.
    surface :           bool, optional. Run the wave over the shell voxels only
                        (`_shell_keys`), so rings are hollow loops around the
                        object rather than solid cross-sections - the literal
                        mesh algorithm, and much cheaper on solid objects. On
                        thin structures it saves nothing (almost every voxel is
                        shell) and it can fragment the skeleton where the surface
                        is not a clean tube.
    radius_agg :        how a ring becomes a radius.

                        ``"volume"`` (default) ignores the ring's shape and
                        instead solves for the tube that reproduces the volume of
                        the voxels assigned to the node - see `_volume_radii`.
                        Being volume-preserving rather than shape-dependent, it is
                        both the least biased and the most accurate option
                        measured: 7.90 against a true 8.00 on an r=8 cylinder, and
                        RMSE 0.66 against the distance transform on the neuron
                        fixture.

                        The rest aggregate each ring's spread about its centroid,
                        as `skeletor` does, and all carry a shape-dependent bias:
                        ``"mean"`` (`skeletor`'s default) reads 5.62 on that same
                        r=8 cylinder because the mean spread of a filled disc is
                        2/3 R, ``"max"`` reads 9.18 and blows up to 1.8x on thin
                        neurites where a ring's corners dominate it. Also
                        ``"median"``, ``"min"``, ``"percentile25"``,
                        ``"percentile75"``. Prefer these only when the voxel
                        counts are not trustworthy - e.g. where rings fuse two
                        touching branches, which inflates the volume of a node
                        without lengthening its tube.
    tree :              bool, optional. Break cycles with a maximum spanning
                        forest (default), as the other two backends produce
                        trees. Set False to keep the raw Reeb graph: at the
                        default `step_size` its cycles are the object's own (an
                        annulus comes out with exactly one), which no
                        tree-producing backend can express.
    preserve_backbone : bool, optional. Weight that forest by radius *and* local
                        straightness rather than radius alone, so cuts land on
                        thin off-axis branches. Ignored when ``tree=False``.
    min_branch_length : prune terminal branches shorter than this (physical units
                        when `spacing` is set), via `skeleton._prune_spurs`.
    verbose :           bool.

    Returns
    -------
    Skeleton
                        With float (sub-voxel) `nodes`, unlike the voxel-valued
                        nodes of `thin_skeletonize` / `teasar_skeletonize`, and
                        with `n_voxels` set: the rings partition the object, so
                        every voxel is accounted for by exactly one node and
                        `Skeleton.volume` gives each node's share of the object.

    References
    ----------
    Verroust A, Lazarus F. Extracting skeletal curves from 3D scattered data.
    The Visual Computer. 2000;16(1):15-25.

    """
    _validate(voxels)
    spacing = _as_spacing(spacing)
    if radius_agg not in _RADIUS_AGGS:
        raise ValueError(
            f"radius_agg must be one of {sorted(_RADIUS_AGGS)}, got {radius_agg!r}"
        )
    floor = _longest_step(spacing)
    if step_size is None:
        step_size = floor
    elif step_size < floor:
        raise ValueError(
            f"step_size={step_size!r} is below the longest graph step ({floor:.4f}), "
            "which would contract non-adjacent rings together and manufacture "
            "cycles. Pass None for the smallest safe value."
        )

    keys, shift = to_keys(voxels, margin=1)
    if surface:
        keys = _shell_keys(keys)
    nodes = from_keys(keys, shift, np.int64)
    if len(nodes) == 0:
        return Skeleton(
            np.empty((0, 3)), np.empty((0, 2), np.int64), np.empty(0), spacing,
            np.empty(0, np.int64),
        )

    if _dijkstra3d_sparse is not None:
        graph = _SparseGraph(nodes, spacing)
        log(f"Wavefront: {len(nodes)} voxels, no edge list (dijkstra3d_sparse).",
            verbose=verbose)
    else:
        # `sort=False`: the edge list is only ever used as an unordered bag (ring
        # membership, then contraction), and the lexsort over 63M rows is half of
        # this call's peak (5.0 -> 2.6 GB on the full neuron).
        vox_edges = _edge_pairs(keys, _CONNECTIVITY, sort=False)
        graph = _ScipyGraph(nodes, vox_edges, spacing)
        log(f"Wavefront: {len(nodes)} voxels, {len(vox_edges)} voxel edges.",
            verbose=verbose)

    n_comp, comp = graph.components()
    log(f"Wavefront: {n_comp} connected component(s).", verbose=verbose)

    pts = nodes.astype(float)
    if spacing is not None:
        pts = pts * spacing

    dist = graph(_seeds(comp, n_comp, graph))
    # Unreached voxels (every component holds a seed, so this only catches
    # numerical oddities) fall into level 0 of their own ring.
    lvl = np.floor(np.where(np.isfinite(dist), dist, 0.0) / step_size).astype(np.int64)

    n_nodes, labels = graph.rings(lvl)
    sk_nodes = _centroids(pts, labels, n_nodes)
    # Every voxel lands in exactly one ring, so this is a partition of the object.
    counts = np.bincount(labels, minlength=n_nodes)

    # The spanning forest weights edges by radius, so a radius has to exist
    # *before* the final edge set does - which is precisely what "volume" needs.
    # Fall back to the mean spread for the weights in that case: they only have to
    # rank edges for cycle-breaking, and the reported radii are replaced below.
    spread = np.linalg.norm(pts - sk_nodes[labels], axis=1)
    sk_radii = _agg(spread, labels, n_nodes, "mean" if radius_agg == "volume" else radius_agg)

    sk_edges = graph.contract(labels, n_nodes)
    log(f"Wavefront: {n_nodes} nodes, {len(sk_edges)} edges.", verbose=verbose)

    if tree:
        sk_edges = _spanning_forest(sk_nodes, sk_edges, sk_radii, preserve_backbone)
        log(f"After spanning forest: {len(sk_edges)} edges.", verbose=verbose)

    # Centres were built in physical space (the wave has to be metric); the
    # Skeleton contract is index-space nodes plus a spacing, so divide back out.
    if spacing is not None:
        sk_nodes = sk_nodes / spacing

    if min_branch_length and min_branch_length > 0:
        sk_nodes, sk_edges, kept = _prune_spurs(
            sk_nodes, sk_edges, float(min_branch_length), spacing, return_index=True
        )
        sk_radii, counts = sk_radii[kept], counts[kept]
        log(f"After pruning: {len(sk_nodes)} nodes, {len(sk_edges)} edges.", verbose=verbose)

    # Deferred to here so it sees the final nodes and edges: the tube length a
    # node's volume is spread over is defined by the edges that survive.
    if radius_agg == "volume":
        sk_radii = _volume_radii(sk_nodes, sk_edges, counts, spacing)

    return Skeleton(sk_nodes, sk_edges, sk_radii, spacing, counts)
