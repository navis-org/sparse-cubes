"""Sparse-voxel TEASAR skeletonization (the kimimaro medial-axis algorithm).

This is an alternative to the topological thinner in `sparsecubes.thinning`.
Where `thin` peels voxels while preserving topology (loops survive), TEASAR
traces medial-axis *paths*: it roots the object at its geodesically furthest
point, then repeatedly draws a shortest path - through a penalty field that hugs
the centerline - to the most distant remaining voxel, invalidating a tube of
voxels around each path as it goes. The result is always an acyclic tree/forest
(loops are broken, matching SWC conventions) with a natural per-node radius (the
distance-from-boundary field, DBF).

The reference implementation, `kimimaro`, is *dense*: it runs an exact Euclidean
distance transform and Dijkstra over the full label volume (multiple gigabytes
for a large block). Here every stage is reformulated on the sparse ``(N, 3)``
voxel set so memory scales with the number of voxels, never the bounding-box
volume - no dense grid is ever allocated:

* DBF: a KD-tree over the sparse boundary shell (`core.boundary_shell`).
* connected components / geodesic distance / path finding: over a
  26-connectivity graph built from one packed neighbour lookup (`_neighbors_26`),
  the same exact-integer-hashing trick used by the mesher and thinner. The
  Dijkstra passes run through `dijkstra3d_sparse` (straight over the voxel
  coordinates, no CSR graph) when it is installed, else `scipy.sparse.csgraph`;
  connected components always use scipy.
* invalidation: KD-tree ball queries around each extracted path.

See TEASAR (Sato et al. 2000) and kimimaro
(https://github.com/seung-lab/kimimaro) for the original algorithm.
"""

import numpy as np

from .core import pack, log, unique, boundary_shell
from .thinning import _OFF26, _validate, _check_extent
from .skeleton import Skeleton, _prune_spurs, _as_spacing, _validate_edges
from ._sparse import sparse_aware

# Optional accelerator: multi-source Dijkstra straight over the sparse voxel
# coordinates (no CSR graph materialised). When present it replaces scipy's
# csgraph.dijkstra for the coordinate-derived 26-graph; scipy stays the fallback
# and is used unconditionally for injected/downsampled graphs (see backend
# selection in `teasar_skeletonize`). Same soft-dependency shape as `core.unique`.
try:
    import dijkstra3d_sparse as _dijkstra3d_sparse
except ModuleNotFoundError:  # pragma: no cover - fallback exercised when the lib is absent
    _dijkstra3d_sparse = None

# `dijkstra3d_sparse.Graph` (added after the initial release) caches the
# coordinate->row index once and reuses it across queries, so a component's many
# Dijkstra/graft calls don't each rebuild the O(N) index. Guard on the attribute
# so older library builds still work through the free functions.
_HAS_GRAPH = _dijkstra3d_sparse is not None and hasattr(_dijkstra3d_sparse, "Graph")

__all__ = ["teasar_skeletonize"]

# Cost given to graph edges that lead into the already-built skeleton when
# `fix_branching` is on: near-zero so later paths ride the existing skeleton for
# free before diverging, but strictly positive to keep Dijkstra well behaved.
_BRANCH_EPS = 1e-6

# SciPy's csgraph uses this sentinel for "no predecessor" (the source node).
_NO_PRED = -9999

# Default number of paths grafted per Dijkstra pass in ``branching='fast'`` - the
# sweet spot from benchmarking (roughly an order of magnitude faster than exact
# extraction on large objects while keeping most of the branch structure).
_DEFAULT_BATCH = 16

# Named `branching` modes -> (fix_branching, batch_size). fix_branching only
# applies to the exact (batch_size is None) extractor.
_BRANCHING_MODES = {
    "exact": (True, None),   # faithful per-path grafting; slowest
    "tree": (False, None),   # single root Dijkstra tree; fastest, coarsest
    "fast": (None, _DEFAULT_BATCH),  # multi-source batched; middle ground
}

# Rows processed per chunk when building the neighbour half-edges, so the
# transient (chunk, 26, 3) index array never blows up for large voxel sets (a
# 5M-voxel object would otherwise need a ~3 GB temporary). Matches the chunking
# `thinning._neighbor_mask` uses for the same reason.
_NB_CHUNK = 250_000


def _neighbors_26(vox):
    """Packed 26-neighbour lookup over a sparse voxel set (chunked).

    Returns ``(nodes, src, dst, off_idx)`` where ``nodes`` is ``vox`` reordered by
    packed key (so node index ``i`` is row ``i``) and ``src``/``dst``/``off_idx``
    describe the directed half-edges of the 26-connectivity graph: for each
    ordered pair of occupied 26-neighbours there is one entry with
    ``nodes[dst] - nodes[src] == _OFF26[off_idx]``. Every undirected edge appears
    twice (once per direction), making the adjacency symmetric.

    Dtypes are deliberately narrow so the edge list - which is ``~26 * N`` entries
    for a solid cloud and dominates memory on large inputs - stays small: node
    indices are ``int32`` (so ``N < 2**31``) and ``off_idx`` is an ``int8`` index
    into the 26 possible offsets rather than a wide ``(E, 3)`` coordinate array.
    Since there are only 26 offsets there are only 26 possible edge weights, so
    the caller maps ``off_idx`` through a 26-entry table instead of taking norms
    per edge. Same ``min - 1`` shift + `pack` + `searchsorted` idiom as
    `skeleton._edges_26`, built in row chunks so the transient stays bounded.
    """
    shift = vox.min(axis=0) - 1
    base = vox - shift
    keys = pack(base)
    order = np.argsort(keys, kind="stable")
    vox = vox[order]
    base = base[order]
    keys_sorted = keys[order]
    m = len(vox)

    col_idx = np.arange(26, dtype=np.int8)  # which offset each of the 26 columns is
    src_parts, dst_parts, idx_parts = [], [], []
    for lo in range(0, m, _NB_CHUNK):
        hi = min(lo + _NB_CHUNK, m)
        c = hi - lo
        nb = base[lo:hi, None, :] + _OFF26[None, :, :]  # (c, 26, 3)
        nb_keys = pack(nb.reshape(-1, 3)).reshape(c, 26)
        pos = np.clip(np.searchsorted(keys_sorted, nb_keys), 0, m - 1)
        hit = keys_sorted[pos] == nb_keys  # (c, 26)
        rows = lo + np.repeat(np.arange(c), 26)[hit.ravel()]
        src_parts.append(rows.astype(np.int32, copy=False))
        dst_parts.append(pos[hit].astype(np.int32, copy=False))
        idx_parts.append(np.broadcast_to(col_idx, (c, 26))[hit])

    src = np.concatenate(src_parts) if src_parts else np.empty(0, np.int32)
    dst = np.concatenate(dst_parts) if dst_parts else np.empty(0, np.int32)
    off_idx = np.concatenate(idx_parts) if idx_parts else np.empty(0, np.int8)
    return vox, src, dst, off_idx


def _dbf(nodes, object_voxels, spacing, workers=-1):
    """Distance-from-boundary field: nearest-background distance per voxel.

    Sparse Euclidean distance transform - `core.boundary_shell` gives the empty
    cells hugging the surface, and the KD-tree returns each node's distance to the
    nearest one (anisotropy applied by scaling coordinates). A surface voxel sits
    ~1 (physical unit) from its shell, never 0, so even one-voxel-thick neurites
    get a positive radius.

    `workers` is handed to the KD-tree query, which dominates this call: the
    default -1 uses every core and is several times faster on large clouds. It
    only affects speed, never the result.
    """
    from scipy.spatial import cKDTree

    shell = boundary_shell(object_voxels).astype(float)
    pts = nodes.astype(float)
    if spacing is not None:
        shell = shell * spacing
        pts = pts * spacing
    tree = cKDTree(shell)
    try:  # `workers` needs scipy >= 1.6
        dist, _ = tree.query(pts, k=1, workers=workers)
    except TypeError:  # pragma: no cover - older scipy
        dist, _ = tree.query(pts, k=1)
    return np.asarray(dist, dtype=float)


def _backtrack(pred, target, visited):
    """Walk predecessors from `target` back to the source, or the first visited node.

    Returns the node list ordered source/attachment -> target. Stops early when it
    reaches a node already on the skeleton so branches graft onto it instead of
    re-tracing all the way to the root.
    """
    path = [int(target)]
    v = int(target)
    while True:
        if visited[v]:  # grafted onto an existing branch
            break
        p = int(pred[v])
        if p == _NO_PRED or p < 0:  # reached the root (source)
            break
        path.append(p)
        v = p
    path.reverse()
    return path


def _invalidate(tree, pts, path, dbf, scale, const, valid):
    """Roll a DBF-scaled ball down `path`, marking every covered voxel invalid.

    Radius ``scale * DBF + const`` per path voxel; no dedup needed since setting
    `valid` to False twice is idempotent.

    Applied one path point at a time rather than querying every ball at once and
    concatenating: on a thick object a single DBF-scaled ball can cover a large
    fraction of the cloud (radius grows with DBF), so materialising all of a path's
    hits together - let alone the concatenated copy - is a multi-GB transient. Ball
    by ball bounds it to the largest single ball, and the point/ball counts trade
    off (thick objects give short paths with big balls, thin ones long paths with
    tiny balls) so the loop stays cheap either way.
    """
    radii = scale * dbf[path] + const
    for p, r in zip(pts[path], radii):
        valid[tree.query_ball_point(p, r, return_sorted=False)] = False
    valid[path] = False  # the path itself is consumed


class _ScipyDijkstra:
    """scipy.csgraph backend: materialise CSR graphs and run `dijkstra` on them.

    Holds the component's geometric graph and, after `set_pdrf`, the penalty-field
    graph. `fix_branching` rewrites the penalty graph's `.data` in place (the
    sparsity pattern never changes), so `col` - each stored edge's destination
    node, valid once `sort_indices()` has run - is cached to index it.
    """

    def __init__(self, vox, src, dst, w_geom, spacing):
        from scipy.sparse import csr_matrix

        self._csr = csr_matrix
        self._src, self._dst, self._w_geom = src, dst, w_geom
        self._m = len(vox)
        self._g_geom = csr_matrix((w_geom, (src, dst)), shape=(self._m, self._m))
        self._g_pdrf = None
        self._col = None

    def geom_dist(self, source):
        from scipy.sparse.csgraph import dijkstra

        return dijkstra(self._g_geom, directed=False, indices=source)

    def set_pdrf(self, pdrf_node):
        # Each node's cost is charged to its incoming edges, so an edge-summing
        # Dijkstra minimizes the integral of the node field. Built and canonicalised
        # ONCE; `fix_branching` then rewrites only the `.data` of edges into the
        # skeleton (rebuilding COO->CSR every path was the dominant cost).
        w_pdrf = (self._w_geom + pdrf_node[self._dst]).astype(np.float32, copy=False)
        g = self._csr((w_pdrf, (self._src, self._dst)), shape=(self._m, self._m))
        g.sort_indices()
        self._g_pdrf, self._col = g, g.indices

    def pdrf_field(self, sources, free_mask=None, min_only=False):
        from scipy.sparse.csgraph import dijkstra

        if free_mask is not None:
            self._g_pdrf.data[free_mask[self._col]] = _BRANCH_EPS  # edges into skeleton ~free
        dist, pred = dijkstra(
            self._g_pdrf, directed=True, indices=sources,
            min_only=min_only, return_predecessors=True,
        )[:2]  # min_only=True adds a 3rd (sources) return the caller never uses
        return dist, pred


class _FreeFuncGraph:
    """Fallback handle for `dijkstra3d_sparse` builds without `Graph`.

    Binds the voxels and forwards to the free functions, which rebuild the
    coordinate index on every call - so `_SparseDijkstra` can drive both this and
    the real reusable `Graph` through one interface.
    """

    __slots__ = ("_vox",)

    def __init__(self, vox):
        self._vox = vox

    def dijkstra_field(self, sources, **kw):
        return _dijkstra3d_sparse.dijkstra_field(self._vox, sources, **kw)

    def shortest_path_to_set(self, source, stop_mask, **kw):
        return _dijkstra3d_sparse.shortest_path_to_set(self._vox, source, stop_mask, **kw)

    def index_of(self, coords, **kw):
        return _dijkstra3d_sparse.index_of(self._vox, coords, **kw)


class _SparseDijkstra:
    """`dijkstra3d_sparse` backend: Dijkstra straight over the voxel coordinates.

    Derives 26-connectivity internally, so it is valid ONLY for the
    coordinate-derived graph - never an injected/downsampled one. The geometric
    graph is `cost_mode="geometric"`; the penalty graph is `cost_mode="additive"`
    with the node field as `node_cost` (edge cost = step length + node_cost[dst],
    exactly teasar's `w_geom + pdrf_node[dst]`); `fix_branching` is
    `free_mask`/`free_eps`.

    A single `Graph` handle is built per component and reused by every query, so
    the O(N) coordinate index is built once rather than rebuilt per Dijkstra/graft
    call - the big win in `branching="exact"`, which issues one `graft` per path.
    """

    def __init__(self, vox, spacing):
        self._aniso = (
            tuple(float(s) for s in spacing) if spacing is not None else (1.0, 1.0, 1.0)
        )
        self._pdrf_node = None
        # Build (and dedup-check) the coordinate index once; reuse for every query.
        self._g = (
            _dijkstra3d_sparse.Graph(vox, index_kind="hash")
            if _HAS_GRAPH
            else _FreeFuncGraph(vox)
        )

    def geom_dist(self, source):
        dist, _ = self._g.dijkstra_field(
            source, cost_mode="geometric", anisotropy=self._aniso, connectivity=26,
        )
        return dist

    def set_pdrf(self, pdrf_node):
        self._pdrf_node = np.asarray(pdrf_node, dtype=np.float32)

    def pdrf_field(self, sources, free_mask=None, min_only=False):
        # Both callers want distance/predecessor to the *nearest* source: a single
        # root (min_only degenerates) or the whole skeleton set. min_only=True
        # returns that as a flat (N,) field - matching scipy's single-source
        # default (identical for one source) and its explicit multi-source min_only.
        dist, pred = self._g.dijkstra_field(
            sources, node_cost=self._pdrf_node, cost_mode="additive",
            anisotropy=self._aniso, connectivity=26,
            free_mask=free_mask, free_eps=_BRANCH_EPS, min_only=True,
        )
        return dist, pred

    def graft(self, target, stop_mask):
        """Shortest penalty-field path from `target` to the nearest `stop_mask` voxel.

        `shortest_path_to_set` settles only `target`'s catchment up to the first
        anchor it reaches, so the search shrinks as the skeleton (stop set) fills -
        the early-termination primitive the root-sourced extractor can't use.
        Returns the path as row indices ordered ``target -> anchor`` plus the
        anchor's row index, or ``(None, -1)`` if no anchor is reachable.
        """
        path_xyz, hit, _ = self._g.shortest_path_to_set(
            int(target), stop_mask, node_cost=self._pdrf_node,
            cost_mode="additive", anisotropy=self._aniso, connectivity=26,
        )
        if hit < 0:
            return None, -1
        # shortest_path_to_set yields coordinates; map them back to row indices.
        idx = self._g.index_of(path_xyz).astype(np.int64, copy=False)
        return idx, int(hit)


def _target_order(daf):
    """Reachable voxel rows ordered farthest-first, ties broken by ascending index.

    Every extraction step targets the still-valid voxel with the largest `daf`
    (geodesic distance from the root), and `np.argmax` breaks ties on the lowest
    row index. Because `daf` never changes during extraction, that whole target
    sequence is one fixed order: walking it and skipping invalidated rows gives the
    identical selection as re-running `argmax` each path, but turns an O(N) rescan
    per path (O(N * paths)) into a single O(N log N) sort.
    """
    finite = np.flatnonzero(np.isfinite(daf))
    # lexsort's last key is primary: -daf sorts farthest-first, and `finite` (already
    # ascending) as the secondary key reproduces argmax's first-occurrence tie-break.
    return finite[np.lexsort((finite, -daf[finite]))]


def _extract_incremental(backend, root, daf, dbf, tree, pts, scale, const):
    """Exact extraction by incremental grafting (dijkstra3d_sparse only).

    Equivalent in spirit to `_extract_perpath` with `fix_branching`, but instead of
    re-running a full root-sourced Dijkstra per path it grafts each new target onto
    the growing skeleton with `shortest_path_to_set` - which explores only the
    catchment between the target and its nearest skeleton node, so the search
    *shrinks* as the tree fills rather than covering the whole component every time.

    Targets are still the geodesically most distant remaining voxels (`daf`), and
    invalidation is unchanged, so the result is the same acyclic tree grafting the
    same branches; graft points can differ slightly from the root-sourced extractor
    where the directed penalty cost prefers a different attachment.
    """
    valid = np.isfinite(daf)
    valid[root] = False
    visited = np.zeros(len(daf), dtype=bool)
    visited[root] = True  # seed the skeleton so the first graft has an anchor
    edges = []

    for target in _target_order(daf):  # farthest-first; daf is fixed for the run
        target = int(target)
        if not valid[target]:  # consumed by an earlier path's ball (or the root)
            continue

        path, hit = backend.graft(target, visited)
        if hit < 0:  # unreachable (never within a connected component); drop it
            valid[target] = False
            continue

        # path runs target -> ... -> hit, whose last node is already on the skeleton.
        for a, b in zip(path[:-1], path[1:]):
            edges.append((int(a), int(b)))
        visited[path] = True
        _invalidate(tree, pts, path, dbf, scale, const, valid)

    return edges, visited


def _extract_perpath(backend, root, daf, dbf, tree, pts, fix_branching, scale, const):
    """Exact extraction: one root-sourced Dijkstra per path (kimimaro-faithful).

    Targets are the geodesically most distant remaining voxels. With
    `fix_branching` the edges leading into the growing skeleton are zeroed in
    place so each new path grafts onto it - a fresh Dijkstra per path. Without it
    a single root Dijkstra tree is reused for every path.
    """
    valid = np.isfinite(daf)  # only reachable voxels are skeletonizable
    visited = np.zeros(len(daf), dtype=bool)
    edges = []

    if not fix_branching:
        _, pred = backend.pdrf_field(root)

    for target in _target_order(daf):  # farthest-first; daf is fixed for the run
        target = int(target)
        if not valid[target]:  # consumed by an earlier path's invalidation ball
            continue

        if fix_branching:
            _, pred = backend.pdrf_field(root, free_mask=visited)

        path = _backtrack(pred, target, visited)
        for a, b in zip(path[:-1], path[1:]):
            edges.append((a, b))
        visited[path] = True
        _invalidate(tree, pts, path, dbf, scale, const, valid)

    return edges, visited


def _extract_multisource(backend, root, daf, dbf, tree, pts, batch_size, scale, const):
    """Approximate extraction: multi-source Dijkstra from the whole skeleton.

    A single ``min_only`` Dijkstra finds every remaining voxel's shortest path to
    the *nearest* skeleton node at once; up to `batch_size` of the most distant
    are grafted on (with invalidation) before the source set is refreshed. This
    cuts the Dijkstra count from O(paths) to ~O(paths / batch_size) - far faster
    on large objects - at the cost of a slightly coarser skeleton (branches
    within a batch attach to the pre-batch skeleton, not to each other).
    """
    valid = np.isfinite(daf)
    valid[root] = False
    visited = np.zeros(len(daf), dtype=bool)
    visited[root] = True
    edges = []

    while valid.any():
        sources = np.flatnonzero(visited)
        dist, pred = backend.pdrf_field(sources, min_only=True)
        cand = valid & np.isfinite(dist)
        if not cand.any():
            break
        nodes = np.flatnonzero(cand)
        taken = 0
        for target in nodes[np.argsort(dist[nodes])[::-1]]:  # most distant first
            if not valid[target]:
                continue
            path = _backtrack(pred, target, visited)
            for a, b in zip(path[:-1], path[1:]):
                edges.append((a, b))
            visited[path] = True
            _invalidate(tree, pts, path, dbf, scale, const, valid)
            taken += 1
            if taken >= batch_size:
                break

    return edges, visited


def _skeletonize_component(vox, src, dst, w_geom, spacing, params, use_sparse):
    """TEASAR one 26-connected component. Returns (nodes, edges, radii).

    `nodes` is a subset of `vox` (the voxels that ended up on a path), `edges`
    are index pairs into `nodes`, `radii` is the DBF sampled at those nodes.
    `src`/`dst`/`w_geom` are the component's directed half-edges (indices into
    `vox`) and their Euclidean lengths. `use_sparse` picks the Dijkstra backend:
    `dijkstra3d_sparse` (coordinate-derived 26-graph) when True, else scipy.
    """
    from scipy.spatial import cKDTree

    m = len(vox)
    dbf = _dbf(vox, vox, spacing, params["workers"])

    # Degenerate component (single voxel, or no interior contrast): one node.
    if m == 1 or dbf.max() <= 0:
        return vox.copy(), np.empty((0, 2), np.int64), dbf.copy()

    if use_sparse:
        backend = _SparseDijkstra(vox, spacing)
    else:
        backend = _ScipyDijkstra(vox, src, dst, w_geom, spacing)

    # --- root -------------------------------------------------------------
    root = params["root"]
    if isinstance(root, str) and root == "geodesic":
        start = int(np.argmax(dbf))
        d0 = backend.geom_dist(start)
        d0[~np.isfinite(d0)] = -1
        root = int(np.argmax(d0))
    elif isinstance(root, str) and root == "dbf":
        root = int(np.argmax(dbf))
    else:  # explicit coordinate -> nearest node
        root = int(np.argmin(np.sum((vox - np.asarray(root)) ** 2, axis=1)))

    # --- distance-from-root (geodesic) and penalty field ------------------
    daf = backend.geom_dist(root)

    ref = params["pdrf_ref"]
    if ref is None:
        # Normalise by the global-max thickness (the soma in a neuron).
        t = dbf / (dbf.max() ** 1.01)
    else:
        # Saturate DBF at a reference thickness (a high DBF quantile) so the whole
        # thick backbone - not just the single thickest voxel - reads as ~zero
        # penalty. This keeps long thick routes cheaper than short thin ones, so
        # paths stop short-cutting through fine-neurite self-touches.
        finite = np.isfinite(dbf) & (dbf > 0)
        thr = np.quantile(dbf[finite], ref) if finite.any() else 0.0
        thr = thr if thr > 0 else dbf.max()
        t = np.minimum(dbf, thr) / thr
    pdrf_node = params["pdrf_scale"] * (1.0 - t) ** params["pdrf_exponent"]
    if params["dbf_term"]:
        finite = np.isfinite(daf)
        dafmax = daf[finite].max() or 1.0
        pdrf_node = pdrf_node + np.where(finite, daf / dafmax, 0.0)

    # Charge each node's penalty to its incoming edges, so an edge-summing Dijkstra
    # minimizes the integral of the node field. The backend prepares this once;
    # `fix_branching` then only marks skeleton nodes free (scipy rewrites the CSR
    # `.data` in place, the sparse backend passes a `free_mask`).
    backend.set_pdrf(pdrf_node)

    # --- extraction + invalidation ----------------------------------------
    scale, const = params["scale"], params["const"]
    pts = vox.astype(float) * (spacing if spacing is not None else 1.0)
    tree = cKDTree(pts)

    if params["batch_size"] is None:
        if use_sparse and params["fix_branching"]:
            # Exact + fix_branching on the accelerator: graft incrementally with
            # early-terminating searches instead of a full root Dijkstra per path.
            edges, visited = _extract_incremental(
                backend, root, daf, dbf, tree, pts, scale, const,
            )
        else:
            edges, visited = _extract_perpath(
                backend, root, daf, dbf, tree, pts,
                params["fix_branching"], scale, const,
            )
    else:
        edges, visited = _extract_multisource(
            backend, root, daf, dbf, tree, pts,
            params["batch_size"], scale, const,
        )

    node_idx = np.flatnonzero(visited)
    if len(node_idx) == 0:  # e.g. root only reachable as an isolated point
        node_idx = np.array([root], dtype=np.int64)

    remap = np.full(m, -1, dtype=np.int64)
    remap[node_idx] = np.arange(len(node_idx))
    if edges:
        e = remap[np.asarray(edges, dtype=np.int64)]
        e = np.sort(e, axis=1)
        e = unique(e, axis=0)
    else:
        e = np.empty((0, 2), dtype=np.int64)
    return vox[node_idx], e, dbf[node_idx]


def _directed_from_edges(vox, edges, spacing):
    """Directed half-edges + geometric weights from an undirected edge list.

    Mirrors `_neighbors_26`'s output contract (each undirected edge appears once
    per direction so the adjacency is symmetric) but for an injected graph. The
    per-edge Euclidean length is computed directly - cheap on an already
    downsampled graph, and unlike the 26-offset table it needs no assumption
    about the coarse-cell spacing.
    """
    e = edges.astype(np.int32, copy=False)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    delta = vox[dst].astype(float) - vox[src].astype(float)
    if spacing is not None:
        delta = delta * spacing
    w_geom = np.sqrt((delta ** 2).sum(axis=1)).astype(np.float32)
    return src, dst, w_geom


@sparse_aware
def teasar_skeletonize(
    voxels,
    *,
    spacing=None,
    downsample=None,
    edges=None,
    scale=1.5,
    const=None,
    pdrf_scale=100_000.0,
    pdrf_exponent=4.0,
    pdrf_ref=None,
    dbf_term=False,
    branching="exact",
    root="geodesic",
    min_branch_length=0,
    check_unique=True,
    workers=-1,
    verbose=False,
):
    """Skeletonize a sparse voxel set with the TEASAR (kimimaro) algorithm.

    Traces medial-axis paths on the sparse voxels directly - no dense volume is
    ever allocated, so memory scales with the number of voxels. The output is an
    acyclic tree/forest with per-node radii (the distance-from-boundary field).
    Requires scipy. If `dijkstra3d_sparse` is installed it runs the Dijkstra
    passes straight over the voxel coordinates (faster, no CSR graph); scipy's
    `csgraph.dijkstra` is the fallback and is always used for an injected/
    downsampled `edges` graph (whose cells need not be coordinate-26-adjacent).

    Parameters
    ----------
    voxels :            (N, 3) integer voxel coordinates (XYZ), as elsewhere in
                        sparse-cubes.
    spacing :           length-3, optional. Physical voxel spacing (anisotropy).
                        Applied to the DBF, edge weights and the invalidation
                        ball; scales `Skeleton.vertices` but never the topology.
    downsample :        positive int or length-3, optional. Downsample the voxels
                        by this factor *before* skeletonizing to run on fewer
                        voxels, via `downsample.downsample_graph` - which coarsens
                        without fusing structures that a plain ``voxels // factor``
                        pool would merge. `spacing` is scaled by the same factor.
                        Mutually exclusive with `edges`.
    edges :             (E, 2) int, optional. Pre-built undirected 26-graph as
                        index pairs into `voxels` (e.g. from `downsample_graph`).
                        When given, this graph is authoritative: 26-adjacency is
                        *not* re-derived from coordinates and `voxels` is used as
                        supplied (must already be unique, indexed by `edges`).
                        Mutually exclusive with `downsample`.
    scale, const :      invalidation-ball radius ``r = scale * DBF + const`` rolled
                        along each path. `const` is in the same units as `spacing`;
                        it defaults to ``4.0`` voxels (``4.0 * min(spacing)`` when
                        `spacing` is set). Note kimimaro's default ``const=300``
                        assumes nanometre EM data - do not reuse it in index space.
    pdrf_scale,
    pdrf_exponent :     penalty field ``pdrf_scale * (1 - DBF/ref) ** pdrf_exponent``
                        that pulls paths onto the centerline. The reference
                        thickness ``ref`` defaults to ``max(DBF)**1.01`` (see
                        `pdrf_ref`).
    pdrf_ref :          float in ``(0, 1]`` or None (default). Quantile of the
                        per-component distance-from-boundary field used as the
                        penalty-field reference thickness: DBF is saturated at
                        that thickness so every voxel at or above it (the thick
                        backbone *and* the soma) gets ~zero penalty, making long
                        thick routes cheap relative to short thin ones. Use this
                        to stop paths short-cutting through fine-neurite
                        self-touches and breaking the backbone (start ~0.8; lower
                        counts more of the object as backbone). ``None`` normalises
                        by the global-max thickness (the soma), reproducing the
                        prior behaviour exactly.
    dbf_term :          add the normalised distance-from-root to the penalty field
                        (kimimaro found this perturbs paths off-centre; default off).
    branching :         extraction strategy, ordered by the speed/fidelity
                        tradeoff (all still yield an acyclic tree):
                         - ``'exact'`` (default) traces one root-sourced Dijkstra per
                           path, grafting each branch onto the skeleton (kimimaro's
                           fix_branching) - most faithful, ``O(paths)`` Dijkstra runs,
                           slowest.
                         - ``'tree'`` reuses a single root Dijkstra tree -
                           fastest, but junctions are coarser.
                         - ``'fast'`` grafts a batch (16) of paths per multi-source
                           Dijkstra pass - a middle ground, roughly an order of magnitude
                           faster than ``'exact'`` on large objects, slightly coarser.
                           Pass a positive int to use the batched extractor with an explicit
                           batch size (larger = faster and coarser).
    root :              ``'geodesic'`` (default; furthest point from the deepest
                        voxel), ``'dbf'`` (deepest voxel), or an explicit ``(3,)``
                        coordinate (snapped to the nearest voxel).
    min_branch_length : prune terminal branches shorter than this (physical units
                        when `spacing` is set), via `skeleton._prune_spurs`.
    check_unique :      bool. If True (default) deduplicate `voxels` before
                        skeletonizing; if False, `voxels` is used as supplied (must
                        already be unique).
    workers :           int, optional
                        Threads for the distance-from-boundary KD-tree query, which
                        is a large share of the runtime on big clouds. ``-1``
                        (default) uses every core; ``1`` disables threading. Purely
                        a speed knob - the skeleton is identical either way. Set it
                        to ``1`` when skeletonizing many objects in parallel
                        yourself (e.g. inside a `multiprocessing` pool), where the
                        default would oversubscribe the CPUs. Needs scipy >= 1.6;
                        silently ignored on older versions.
    verbose :           bool.

    Returns
    -------
    Skeleton

    """
    _validate(voxels)
    dtype = voxels.dtype
    spacing = _as_spacing(spacing)

    if downsample is not None and edges is not None:
        raise ValueError("pass either `downsample` or `edges`, not both.")
    if downsample is not None:
        from .downsample import downsample_graph

        voxels, edges = downsample_graph(voxels, downsample)
        if spacing is not None:
            spacing = spacing * np.asarray(downsample)

    # With an injected graph the coordinates and edges are authoritative and
    # already aligned (edges index into `voxels` as given), so we must not
    # dedup/reorder them; `_neighbors_26` reorders, so it is skipped below too.
    if edges is None:
        if check_unique:
            vox = unique(voxels, axis=0).astype(np.int64)
        else:
            vox = np.ascontiguousarray(voxels).astype(np.int64)
    else:
        vox = np.ascontiguousarray(voxels).astype(np.int64)
        edges = _validate_edges(edges, len(vox))

    empty_edges = np.empty((0, 2), dtype=np.int64)
    if len(vox) == 0:
        return Skeleton(vox.astype(dtype), empty_edges, None, spacing)

    _check_extent(vox)

    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError(
            "teasar_skeletonize requires scipy; install `sparse-cubes[skeleton]` "
            "or `pip install scipy`."
        ) from exc

    if const is None:
        const = 4.0 if spacing is None else 4.0 * float(np.min(spacing))

    if pdrf_ref is not None and not (0.0 < float(pdrf_ref) <= 1.0):
        raise ValueError(
            f"pdrf_ref must be a quantile in (0, 1] or None; got {pdrf_ref!r}."
        )

    # `branching` -> (fix_branching, batch_size). batch_size None = exact per-path.
    if isinstance(branching, str):
        try:
            fix_branching, batch_size = _BRANCHING_MODES[branching]
        except KeyError:
            raise ValueError(
                f"branching must be one of {sorted(_BRANCHING_MODES)} or a positive "
                f"int batch size; got {branching!r}."
            ) from None
    elif isinstance(branching, bool) or not isinstance(branching, (int, np.integer)):
        raise ValueError(
            f"branching must be one of {sorted(_BRANCHING_MODES)} or a positive int "
            f"batch size; got {branching!r}."
        )
    else:
        batch_size = int(branching)
        fix_branching = None
        if batch_size < 1:
            raise ValueError("branching batch size must be a positive int.")

    params = {
        "scale": float(scale),
        "const": float(const),
        "pdrf_scale": float(pdrf_scale),
        "pdrf_exponent": float(pdrf_exponent),
        "pdrf_ref": None if pdrf_ref is None else float(pdrf_ref),
        "dbf_term": bool(dbf_term),
        "fix_branching": fix_branching,
        "batch_size": batch_size,
        "root": root,
        "workers": int(workers),
    }

    if len(vox) >= 2 ** 31:  # int32 node indices in the edge list
        raise ValueError("teasar_skeletonize supports up to 2**31 voxels.")

    # Dijkstra backend: prefer `dijkstra3d_sparse` when installed, but only for the
    # coordinate-derived 26-graph. An injected graph (`edges`/`downsample`) is
    # authoritative and its cells need not be coordinate-26-adjacent, so the
    # library - which re-derives adjacency from coordinates - can't honour it;
    # scipy stays the fallback there and whenever the library is absent.
    use_sparse = _dijkstra3d_sparse is not None and edges is None
    m = len(vox)

    if use_sparse:
        # The sparse backend works straight off the coordinates and never touches
        # the edge list, so materialising the ~26*N directed half-edges and a
        # weighted CSR just to label components is pure overhead - on a big solid
        # cloud it is the dominant allocation (tens of GB: a 44M-voxel object spends
        # ~70 GB here alone). Label components on the coordinates instead - no edge
        # array, no CSR - and hand `_skeletonize_component` only the coordinates.
        n_comp, labels = _dijkstra3d_sparse.connected_components(vox, connectivity=26)
        src = dst = w_geom = None
    else:
        # scipy backend / injected graph: one neighbour lookup drives both the
        # connected components and the weighted graph. `off_idx` (int8) indexes the
        # 26 possible offsets, hence the 26 possible edge weights - so we map through
        # a small table (float32) instead of materialising a wide per-edge array.
        if edges is None:
            vox, src, dst, off_idx = _neighbors_26(vox)
            sp = spacing if spacing is not None else 1.0
            w_table = np.sqrt(((_OFF26.astype(float) * sp) ** 2).sum(axis=1)).astype(
                np.float32
            )
            w_geom = w_table[off_idx]  # (E,) float32
            del off_idx
        else:
            # Injected graph: honour the given edges verbatim (`vox` already aligned).
            src, dst, w_geom = _directed_from_edges(vox, edges, spacing)
        m = len(vox)
        graph = csr_matrix((w_geom, (src, dst)), shape=(m, m))
        n_comp, labels = connected_components(graph, directed=False)
        del graph  # only the labels are needed; free the CC graph before the loop

    log(
        f"TEASAR: {m} voxels, {n_comp} component(s); "
        f"Dijkstra backend={'dijkstra3d_sparse' if use_sparse else 'scipy'}.",
        verbose=verbose,
    )

    # Bucket voxel rows (and, on the scipy path, edges) by component ONCE. Scanning
    # `labels == c` per component is O(N) each - plus O(E) for the edge mask and an
    # O(N) `g2l` allocation on the scipy path - i.e. O(N * n_comp), which explodes
    # when a cloud fragments into many components. A stable argsort buckets the rows
    # in O(N log N); each component is then a contiguous slice, in the same ascending
    # row order `np.flatnonzero(labels == c)` produced.
    vorder = np.argsort(labels, kind="stable")
    vbounds = np.concatenate(([0], np.cumsum(np.bincount(labels, minlength=n_comp))))

    if not use_sparse:
        # Global row -> within-component local index, built once for every row.
        counts = np.diff(vbounds)
        local = np.arange(m, dtype=np.int64) - np.repeat(vbounds[:-1], counts)
        g2l = np.empty(m, dtype=np.int32)
        g2l[vorder] = local.astype(np.int32, copy=False)
        lsrc = labels[src]  # each edge's component (== labels[dst]; edges stay within)
        eorder = np.argsort(lsrc, kind="stable")
        ebounds = np.concatenate(([0], np.cumsum(np.bincount(lsrc, minlength=n_comp))))

    all_nodes, all_edges, all_radii = [], [], []
    offset = 0
    for c in range(n_comp):
        gidx = vorder[vbounds[c]:vbounds[c + 1]]  # == np.flatnonzero(labels == c)
        if use_sparse:
            # Only the component's coordinates are needed - the backend re-derives
            # adjacency - so skip the per-component edge slicing (three full ~E-sized
            # copies plus a bool mask over all edges, the loop's memory peak).
            nodes_c, edges_c, radii_c = _skeletonize_component(
                vox[gidx], None, None, None, spacing, params, use_sparse,
            )
        else:
            ec = eorder[ebounds[c]:ebounds[c + 1]]  # this component's half-edges
            nodes_c, edges_c, radii_c = _skeletonize_component(
                vox[gidx],
                g2l[src[ec]],
                g2l[dst[ec]],
                w_geom[ec],
                spacing,
                params,
                use_sparse,
            )
        all_nodes.append(nodes_c)
        all_edges.append(edges_c + offset)
        all_radii.append(radii_c)
        offset += len(nodes_c)

    nodes = np.vstack(all_nodes) if all_nodes else vox[:0]
    edges = np.vstack(all_edges) if all_edges else empty_edges
    radii = np.concatenate(all_radii) if all_radii else np.empty(0, dtype=float)
    log(f"TEASAR: {len(nodes)} nodes, {len(edges)} edges.", verbose=verbose)

    if min_branch_length and min_branch_length > 0 and len(edges):
        kept, edges = _prune_spurs(nodes, edges, float(min_branch_length), spacing)
        radii = _match_radii(nodes, radii, kept)
        nodes = kept
        log(f"After pruning: {len(nodes)} nodes, {len(edges)} edges.", verbose=verbose)

    return Skeleton(nodes.astype(dtype), edges, radii, spacing)


def _match_radii(old_nodes, old_radii, new_nodes):
    """Carry radii from `old_nodes` to `new_nodes` (a subset), matched by coordinate.

    `_prune_spurs` only removes nodes, so identity is preserved and a packed-key
    lookup recovers each surviving node's radius.
    """
    if len(new_nodes) == 0:
        return np.empty(0, dtype=float)
    shift = np.minimum(old_nodes.min(axis=0), new_nodes.min(axis=0)) - 1
    ok = pack(old_nodes - shift)
    nk = pack(new_nodes - shift)
    order = np.argsort(ok)
    pos = order[np.searchsorted(ok[order], nk)]
    return old_radii[pos]
