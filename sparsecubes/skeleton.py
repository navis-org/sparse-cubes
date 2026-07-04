"""Centerline skeleton extraction from (thinned) sparse voxels.

`skeletonize` chains `sparsecubes.thinning.thin` with `centerline`; `centerline`
turns an already one-voxel-wide voxel set into a node/edge graph. Nodes are the
voxels, edges are their 26-connections (found with the same `pack`-based lookup
as the thinner). The result is a lightweight `Skeleton` (plain numpy ``nodes``
and ``edges`` arrays, plus optional per-node ``radii``) with lazy converters to
networkx / trimesh / SWC.
"""

from dataclasses import dataclass
from collections import deque

import numpy as np

from .core import pack, log, unique, boundary_shell
from .thinning import thin, _OFF26, _check_extent, _validate

__all__ = ["Skeleton", "centerline", "skeletonize"]


def _as_spacing(spacing):
    return None if spacing is None else np.asarray(spacing)


@dataclass
class Skeleton:
    """A centerline skeleton as plain arrays.

    Attributes
    ----------
    nodes :     (M, 3) integer voxel coordinates (one node per skeleton voxel).
    edges :     (K, 2) int64 node-index pairs (undirected, 26-connectivity).
    radii :     (M,) float or None. Optional per-node radius estimate.
    spacing :   (3,) or None. Physical voxel spacing, applied by `vertices` and
                the converters (topology is always computed in index space).
    """

    nodes: np.ndarray
    edges: np.ndarray
    radii: "np.ndarray | None" = None
    spacing: "np.ndarray | None" = None

    @property
    def vertices(self):
        """Node coordinates as float, scaled by `spacing` when set."""
        v = self.nodes.astype(float)
        return v * self.spacing if self.spacing is not None else v

    def node_degrees(self):
        """Degree of every node (1=end, 2=path, >=3=branch)."""
        if len(self.edges) == 0:
            return np.zeros(len(self.nodes), dtype=np.int64)
        return np.bincount(self.edges.ravel(), minlength=len(self.nodes))

    def to_networkx(self):
        """Return a `networkx.Graph` with ``pos`` (and ``radius``) node attrs."""
        import networkx as nx

        verts = self.vertices
        g = nx.Graph()
        for i in range(len(self.nodes)):
            attrs = {"pos": tuple(verts[i])}
            if self.radii is not None:
                attrs["radius"] = float(self.radii[i])
            g.add_node(i, **attrs)
        g.add_edges_from(map(tuple, self.edges.tolist()))
        return g

    def to_path3d(self):
        """Return a `trimesh.path.Path3D` of the skeleton edges."""
        from trimesh.path import Path3D
        from trimesh.path.entities import Line

        entities = [Line(np.asarray(e)) for e in self.edges]
        return Path3D(entities=entities, vertices=self.vertices)

    def to_swc(self, filepath=None, root=None):
        """Build an SWC table ``[id, type, x, y, z, radius, parent]``.

        A rooted spanning forest is built by BFS (so any cycles are broken at an
        arbitrary edge - SWC cannot represent loops). Ids are 1-indexed and the
        root's parent is -1. If `filepath` is given the table is also written.
        """
        m = len(self.nodes)
        adj = [[] for _ in range(m)]
        for a, b in self.edges:
            adj[a].append(b)
            adj[b].append(a)

        if root is None:
            root = int(np.argmax(self.radii)) if self.radii is not None else 0

        parent = np.full(m, -1, dtype=np.int64)
        visited = np.zeros(m, dtype=bool)
        for start in [root, *range(m)]:
            if visited[start]:
                continue
            visited[start] = True
            dq = deque([start])
            while dq:
                u = dq.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        dq.append(v)

        verts = self.vertices
        radii = self.radii if self.radii is not None else np.zeros(m)
        swc = np.empty((m, 7), dtype=float)
        swc[:, 0] = np.arange(1, m + 1)  # 1-indexed id
        swc[:, 1] = 0  # type: undefined
        swc[:, 2:5] = verts
        swc[:, 5] = radii
        swc[:, 6] = np.where(parent < 0, -1, parent + 1)  # 1-indexed parent

        if filepath is not None:
            np.savetxt(
                filepath,
                swc,
                fmt="%d %d %.4f %.4f %.4f %.4f %d",
                header="id type x y z radius parent",
            )
        return swc


def _edges_26(nodes):
    """Canonicalise node order and return unique 26-connected edge index pairs.

    Nodes are reordered by their packed key so a node's index equals its
    position in the sorted key array (which `searchsorted` returns for lookups).
    """
    shift = nodes.min(axis=0) - 1
    base = nodes - shift
    keys = pack(base)
    order = np.argsort(keys, kind="stable")
    nodes = nodes[order]
    base = base[order]
    keys_sorted = keys[order]

    m = len(nodes)
    nb = base[:, None, :] + _OFF26[None, :, :]  # (M, 26, 3)
    nb_keys = pack(nb.reshape(-1, 3)).reshape(m, 26)
    pos = np.clip(np.searchsorted(keys_sorted, nb_keys), 0, m - 1)
    hit = keys_sorted[pos] == nb_keys  # (M, 26)

    if not hit.any():
        return nodes, np.empty((0, 2), dtype=np.int64)

    src = np.repeat(np.arange(m), 26)[hit.ravel()]
    dst = pos[hit]
    e = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis=1)
    return nodes, unique(e, axis=0)


def _validate_edges(edges, m):
    """Check an injected edge list: (E, 2) integer index pairs within [0, m)."""
    e = np.asarray(edges)
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError(f"edges must have shape (E, 2), got {e.shape}.")
    if not np.issubdtype(e.dtype, np.integer):
        raise ValueError(f"edges must be integer indices, got dtype {e.dtype}.")
    if len(e) and (e.min() < 0 or e.max() >= m):
        raise ValueError("edges reference node indices outside the voxel set.")
    return e


def _reduce_edges(nodes, edges):
    """Drop diagonal shortcut edges that are the long side of a filled triangle.

    A one-voxel-wide staircase produces filled triangles in 26-connectivity
    (e.g. ``(0,0,0)-(1,0,0)-(1,1,0)`` plus the ``(0,0,0)-(1,1,0)`` hypotenuse),
    which inflate node degrees and manufacture spurious branch points. An edge
    ``(a, c)`` is removed if some common neighbour ``b`` gives a *strictly
    shorter* two-hop path ``a-b-c``. Face (6-connectivity, length 1) edges are
    always kept, so connectivity - and any genuine diagonal-only link, which has
    no shorter detour - is preserved; only discretisation triangles collapse.
    """
    if len(edges) == 0:
        return edges

    coords = nodes.astype(np.int64)

    def d2(i, j):
        delta = coords[i] - coords[j]
        return int(delta @ delta)

    adj = [set() for _ in range(len(nodes))]
    for a, b in edges:
        adj[a].add(int(b))
        adj[b].add(int(a))

    keep = np.ones(len(edges), dtype=bool)
    for idx in range(len(edges)):
        a, b = int(edges[idx, 0]), int(edges[idx, 1])
        length = d2(a, b)
        if length <= 1:  # 6-connectivity edges are always kept
            continue
        for c in adj[a] & adj[b]:
            if d2(a, c) < length and d2(c, b) < length:
                keep[idx] = False
                break
    return edges[keep]


def _prune_spurs(nodes, edges, min_len, spacing):
    """Remove terminal branches (end -> junction) shorter than `min_len`.

    Length is the summed segment length along the branch, in physical units when
    `spacing` is given. Iterates until stable (removing a spur can shorten the
    junction to degree 2 and expose another).
    """
    if min_len <= 0 or len(edges) == 0:
        return nodes, edges

    changed = True
    while changed:
        changed = False
        m = len(nodes)
        adj = [[] for _ in range(m)]
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        deg = np.array([len(a) for a in adj])
        coords = nodes.astype(float)
        if spacing is not None:
            coords = coords * spacing

        remove = set()
        for end in np.flatnonzero(deg == 1):
            if end in remove:
                continue
            path = [int(end)]
            prev, cur, length = -1, int(end), 0.0
            while True:
                nbrs = [n for n in adj[cur] if n != prev]
                if len(nbrs) != 1:
                    break  # reached a junction (deg>=3) or dead end
                nxt = nbrs[0]
                length += float(np.linalg.norm(coords[nxt] - coords[cur]))
                path.append(nxt)
                prev, cur = cur, nxt
                if deg[cur] != 2:
                    break
            junction = path[-1]
            # Only prune spurs hanging off a real branch point; leave isolated
            # end-to-end paths (whole components) alone.
            if deg[junction] >= 3 and length < min_len:
                remove.update(path[:-1])
                changed = True

        if remove:
            keep = np.ones(len(nodes), dtype=bool)
            keep[list(remove)] = False
            remap = np.full(len(nodes), -1, dtype=np.int64)
            remap[keep] = np.arange(int(keep.sum()))
            edge_keep = keep[edges[:, 0]] & keep[edges[:, 1]]
            nodes = nodes[keep]
            edges = remap[edges[edge_keep]]

    return nodes, edges


def _radii(nodes, object_voxels, spacing):
    """Per-node radius ~ distance to the nearest background (empty) voxel.

    The object's exposed faces, each stepped one voxel outward, form the shell of
    background cells hugging the surface; the nearest such cell to a node is the
    nearest empty voxel, i.e. the local radius. This stays sparse (reuses
    `find_surface_voxels`) and, unlike distance to the nearest *surface voxel*,
    is non-zero for one-voxel-thick neurites whose centerline lies on the
    surface. It is an approximation (nearest voxel *centre*), not an exact EDT.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError(
            "radii=True requires scipy; install `sparse-cubes[skeleton]` "
            "or `pip install scipy`."
        ) from exc

    shell = boundary_shell(object_voxels)

    pts = nodes.astype(float)
    spts = shell.astype(float)
    if spacing is not None:
        pts = pts * spacing
        spts = spts * spacing
    dist, _ = cKDTree(spts).query(pts, k=1)
    return np.asarray(dist, dtype=float)


def centerline(
    thinned,
    *,
    spacing=None,
    min_branch_length=0,
    radii=False,
    object_voxels=None,
    edges=None,
    verbose=False,
):
    """Extract a centerline `Skeleton` from already-thinned voxels.

    Parameters
    ----------
    thinned :           (M, 3) integer voxels, expected one-voxel-wide (e.g. the
                        output of `thin`). Not re-thinned here.
    spacing :           length-3, optional. Physical voxel spacing. Affects
                        branch-length thresholds, radii and converter output;
                        never the graph topology.
    min_branch_length : float, optional. Prune terminal branches shorter than
                        this (in physical units when `spacing` is set).
    radii :             bool, optional. If True, estimate a per-node radius as
                        the distance to the nearest background (empty) voxel just
                        outside the object. Needs `object_voxels` and scipy.
    object_voxels :     (N, 3), optional. The original (un-thinned) voxels, used
                        only for radius estimation.
    edges :             (E, 2) int, optional. Pre-built undirected edges as index
                        pairs into `thinned` (e.g. from `downsample_graph`). When
                        given, 26-adjacency is *not* re-derived from coordinates
                        and `thinned` is used as supplied (must already be unique,
                        indexed by `edges`).
    verbose :           bool.

    Returns
    -------
    Skeleton

    """
    _validate(thinned)
    dtype = thinned.dtype
    spacing = _as_spacing(spacing)

    if edges is None:
        nodes = unique(thinned, axis=0).astype(np.int64)
    else:
        # Injected graph: keep `thinned` as-is so the given edges stay aligned.
        nodes = np.ascontiguousarray(thinned).astype(np.int64)
    if len(nodes) == 0:
        return Skeleton(nodes.astype(dtype), np.empty((0, 2), np.int64), None, spacing)

    _check_extent(nodes)
    if edges is None:
        nodes, edges = _edges_26(nodes)
    else:
        edges = _validate_edges(edges, len(nodes))
    edges = _reduce_edges(nodes, edges)
    log(f"Centerline: {len(nodes)} nodes, {len(edges)} edges.", verbose=verbose)

    if min_branch_length and min_branch_length > 0:
        nodes, edges = _prune_spurs(nodes, edges, float(min_branch_length), spacing)
        log(f"After pruning: {len(nodes)} nodes, {len(edges)} edges.", verbose=verbose)

    r = None
    if radii:
        if object_voxels is None:
            raise ValueError(
                "radii=True needs the original object; call skeletonize(...) or "
                "pass object_voxels=<original (N, 3) voxels>."
            )
        r = _radii(nodes, np.asarray(object_voxels).astype(np.int64), spacing)

    return Skeleton(nodes.astype(dtype), edges, r, spacing)


def skeletonize(
    voxels,
    *,
    spacing=None,
    preserve_endpoints=True,
    min_branch_length=0,
    radii=False,
    verbose=False,
):
    """Thin `voxels` and extract a centerline `Skeleton` in one call.

    Convenience wrapper: ``centerline(thin(voxels), object_voxels=voxels, ...)``.
    See `thin` and `centerline` for the parameters.
    """
    _validate(voxels)
    thinned = thin(voxels, preserve_endpoints=preserve_endpoints, verbose=verbose)
    return centerline(
        thinned,
        spacing=spacing,
        min_branch_length=min_branch_length,
        radii=radii,
        object_voxels=voxels,
        verbose=verbose,
    )
