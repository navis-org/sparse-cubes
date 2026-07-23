import warnings

import numpy as np
import trimesh as tm

try:
    from fastremap import unique
except ImportError:
    from numpy import unique

INT_DTYPES = (np.int64, np.int32, np.int16, np.uint64, np.uint32, np.uint16)

from ._sparse import sparse_aware

# `find_surface_voxels` extracts the object's exposed faces with one native pass
# over the coordinates - `dijkstra3d_sparse.exposed_faces`, a per-voxel 6-bit mask
# built off a transient spatial index - instead of three lexsorts over the whole
# voxel array. The index is built and freed inside that call, so nothing persists
# into the later mesh stages. Needs dijkstra3d_sparse >= 0.2.1; a missing or older
# install falls back to the sort-based path, which stays dependency-free.
try:
    import dijkstra3d_sparse as _d3s

    _HAVE_EXPOSED_FACES = hasattr(_d3s, "exposed_faces")
except ImportError:
    _d3s = None
    _HAVE_EXPOSED_FACES = False


# The original Trimesh automatically casts vertices to float64. Realistically,
# our meshes should be fine with uint32 most of the time so we will avoid the
# casting altogether and hope we don't break anything in the process.
# This should speed things up quite a bit and obviously also reduce the
# memory footprint
class Trimesh(tm.Trimesh):
    @property
    def vertices(self):
        # Check if vertices exist and return with appropriate dtype
        verts = self._data.get("vertices", None)
        if verts is None:
            return np.empty(shape=(0, 3), dtype=np.uint32)
        return verts

    @vertices.setter
    def vertices(self, values):
        self._data["vertices"] = np.asanyarray(values, order="C")


@sparse_aware
def mesh(voxels, spacing=None, step_size=1, verbose=False, smooth=True, simplify=False):
    """Generate a surface mesh from sparse `(N, 3)` voxel indices.

    Works directly on the sparse surface voxels (memory ~ number of surface
    voxels), so it avoids densifying to a 3D grid the way marching cubes would.

    Two vertex-placement modes are available via `smooth`:

    - Smooth (``smooth=True``, the default): naive SurfaceNets. One vertex per
      surface cell, placed at the centroid of the surface crossings around it.
      This is a *dual* method (a cousin of dual contouring) and smooths the
      staircase you would otherwise get on diagonal surfaces. Vertices are
      floats. See also `surface_nets`.
    - Blocky (``smooth=False``): each exposed voxel face becomes an axis-aligned
      quad with corners on the integer voxel grid ("culled cube faces", à la
      Minecraft). Diagonal surfaces come out as 90 degree steps, but vertices
      keep the input integer dtype. This is the historical output. See also
      `culled_faces`.

    Neither mode is true dual contouring or marching cubes: vertices land either
    on cube corners (blocky) or at edge-crossing centroids (smooth), not via the
    QEF/normal-based placement of feature-preserving dual contouring.

    Parameters
    ----------
    voxels :    (N, 3) array of integers
                XYZ voxel coordinates (indices) to find isosurfaces for. On the
                blocky path (``smooth=False``) the data type carries over to the
                mesh vertices - i.e. if `voxels` are unsigned 32-bit integers,
                the mesh vertex coordinates are too. The smooth path always
                produces float vertices.
    spacing :   length-3 tuple of floats, optional
                Voxel spacing in spatial dimensions corresponding to numpy array
                indexing dimensions as in `voxels`. Note that the data type of
                `spacing` will affect the data type of the mesh's vertices.
                Ideally, you should use the same dtype for `spacing` as for
                `voxels`.
    step_size : int, optional
                Step size in voxels. Default 1. Larger steps yield coarser
                results.
    verbose :   bool
                If True, will provide some feedback on progress.
    smooth :    bool, optional
                If True (default), use naive SurfaceNets placement: one vertex
                per active dual cell, positioned at the centroid of that cell's
                surface-crossing edges. This smooths the staircase on diagonal
                surfaces at the cost of a small constant-factor slowdown. Note
                that vertices are floats on this path (they are generally
                half-integer or finer), so the integer-dtype optimisation of
                the blocky path does not apply.
                If False, each exposed voxel face instead becomes an
                axis-aligned quad with corners on integer voxel coordinates -
                i.e. a blocky mesh with 90 degree steps on diagonal surfaces.
                This output is byte-identical to versions <= 0.2.0 and keeps
                the input integer dtype.
    simplify :  bool, optional
                Only used on the blocky path (``smooth=False``). If True, merge
                coplanar adjacent faces into maximal rectangles (greedy meshing).
                This is lossless - the covered surface is unchanged - and
                typically produces ~2x fewer triangles, which usually makes it
                *faster* than the un-simplified blocky path (fewer vertices to
                merge). See `greedy_faces`. Caveat: greedy meshing can introduce
                T-junctions, so the result may be "less watertight" than the
                per-face mesh. Ignored (with a warning) when ``smooth=True``.

    Returns
    -------
    Trimesh

    """
    if not isinstance(voxels, np.ndarray):
        raise TypeError(f'Expected numpy array, got "{type(voxels)}"')
    elif voxels.ndim != 2:
        raise TypeError(f'Expected 2d numpy array, got "{voxels.ndim}"')
    elif voxels.shape[1] != 3:
        raise TypeError(f'Expected numpy array of shape (N, 3), got "{voxels.shape}"')
    elif voxels.dtype not in INT_DTYPES:
        raise TypeError(f"Expected integer dtype, got {voxels.dtype}")

    if simplify and smooth:
        warnings.warn(
            "`simplify` has no effect on the smooth path; ignoring.",
            stacklevel=2,
        )
        simplify = False

    if not isinstance(spacing, type(None)):
        spacing = np.array(spacing)

    # Step size is implemented by simple downsampling
    if step_size and step_size > 1:
        voxels = unique(voxels // step_size, axis=0).astype(voxels.dtype)

        if not isinstance(spacing, type(None)):
            spacing = spacing * step_size

    # Find surface voxels
    log("Looking for surface voxels... ", end="", flush=True, verbose=verbose)
    (voxels_left, voxels_right, voxels_back, voxels_front, voxels_bot, voxels_top) = (
        find_surface_voxels(voxels)
    )
    log("Done.", flush=True, verbose=verbose)

    # Generate vertices + faces
    log("Generating vertices and faces... ", end="", flush=True, verbose=verbose)
    if smooth:
        verts, faces = make_surface_nets(
            voxels_left, voxels_right, voxels_back, voxels_front, voxels_bot, voxels_top
        )
    elif simplify:
        verts, faces = make_greedy_faces(
            voxels_left, voxels_right, voxels_back, voxels_front, voxels_bot, voxels_top
        )
    else:
        verts, faces = make_culled_faces(
            voxels_left, voxels_right, voxels_back, voxels_front, voxels_bot, voxels_top
        )
    log("Done.", flush=True, verbose=verbose)

    # Create mesh
    log("Making mesh... ", end="", flush=True, verbose=verbose)
    m = Trimesh(verts, faces, process=False)
    log("Done.", flush=True, verbose=verbose)

    # Collapse vertices. The smooth path already emits one vertex per active
    # dual cell (dedup comes for free via the cell -> vertex index map), so we
    # only need to merge on the blocky path.
    if not smooth:
        log("Merging vertices... ", end="", flush=True, verbose=verbose)
        tm.grouping.merge_vertices(m, digits_vertex=0, merge_tex=True, merge_norm=True)
        log("Done.", flush=True, verbose=verbose)

    # Apply spacing after we collapse duplicate vertices
    if not isinstance(spacing, type(None)):
        m.vertices = m.vertices * spacing

    log("All done!", flush=True, verbose=verbose)
    return m


@sparse_aware
def surface_nets(voxels, spacing=None, step_size=1, verbose=False):
    """Smooth (naive SurfaceNets) surface mesh from sparse voxels.

    Thin wrapper around ``mesh(..., smooth=True)``. See `mesh` for details.
    """
    return mesh(
        voxels, spacing=spacing, step_size=step_size, verbose=verbose, smooth=True
    )


@sparse_aware
def culled_faces(voxels, spacing=None, step_size=1, verbose=False, simplify=False):
    """Blocky (culled cube faces) surface mesh from sparse voxels.

    Thin wrapper around ``mesh(..., smooth=False)``. Pass ``simplify=True`` (or
    use `greedy_faces`) to merge coplanar faces. See `mesh` for details.
    """
    return mesh(
        voxels,
        spacing=spacing,
        step_size=step_size,
        verbose=verbose,
        smooth=False,
        simplify=simplify,
    )


@sparse_aware
def greedy_faces(voxels, spacing=None, step_size=1, verbose=False):
    """Simplified blocky mesh with coplanar faces merged (greedy meshing).

    Thin wrapper around ``mesh(..., smooth=False, simplify=True)``. Lossless
    (covered surface unchanged) and typically ~2x fewer triangles. See `mesh`
    and `make_greedy_faces` for details and caveats (e.g. T-junctions).
    """
    return mesh(
        voxels,
        spacing=spacing,
        step_size=step_size,
        verbose=verbose,
        smooth=False,
        simplify=True,
    )


@sparse_aware
def dual_contour(voxels, spacing=None, step_size=1, verbose=False, interpolate=True):
    """Deprecated, misnamed alias for `mesh` (this is not dual contouring).

    .. deprecated::
        Use `mesh` (or the explicit `surface_nets` / `culled_faces`) instead.
        The `interpolate` argument maps to `smooth`.
    """
    warnings.warn(
        "`dual_contour` is deprecated and misnamed - this library does not "
        "implement dual contouring. Use `mesh(..., smooth=...)` instead "
        "(or `surface_nets` / `culled_faces`).",
        DeprecationWarning,
        stacklevel=2,
    )
    return mesh(
        voxels,
        spacing=spacing,
        step_size=step_size,
        verbose=verbose,
        smooth=interpolate,
    )


@sparse_aware
def marching_cubes(voxels, spacing=None, step_size=1, verbose=False, interpolate=True):
    """Deprecated, misnamed alias for `mesh` (this is not marching cubes).

    .. deprecated::
        Use `mesh` (or the explicit `surface_nets` / `culled_faces`) instead.
        The `interpolate` argument maps to `smooth`.
    """
    warnings.warn(
        "`marching_cubes` is deprecated and misnamed - this library does not "
        "implement marching cubes. Use `mesh(..., smooth=...)` instead "
        "(or `surface_nets` / `culled_faces`).",
        DeprecationWarning,
        stacklevel=2,
    )
    return mesh(
        voxels,
        spacing=spacing,
        step_size=step_size,
        verbose=verbose,
        smooth=interpolate,
    )


def sort_cols(a, order=[0, 1, 2]):
    """Sort 2-d array by columns."""
    return a[argsort_cols(a, order=order)]


def argsort_cols(a, order=[0, 1, 2]):
    """Sort 2-d array by columns."""
    cols = a.T[order[::-1]]
    return np.lexsort(cols)


def find_surface_voxels(voxels):
    """Find surface voxels.

    Returns the six directional sets of voxels with an exposed face - one per
    face direction, in the order ``(left, right, back, front, bot, top)``. A
    voxel is in a set when its neighbour one step along that direction is absent
    from the object.

    Uses the native `exposed_faces` probe (`_surface_voxels_probe`) when
    `dijkstra3d_sparse` is available, otherwise the sort-based fallback
    (`_surface_voxels_sorted`). The two return set-identical results; only the
    order of voxels within each set differs, which nothing downstream depends on
    (each exposed face is meshed independently, and `boundary_shell` deduplicates).

    Parameters
    ----------
    voxels :    (N, 3) numpy arraay

    Returns
    -------
    (voxels_left,
     voxels_right,
     voxels_back,
     voxels_front,
     voxels_bot,
     voxels_top)

    """
    if _HAVE_EXPOSED_FACES and len(voxels):
        try:
            return _surface_voxels_probe(voxels)
        except ValueError:
            # `exposed_faces` rejects duplicate coordinates, which the sort-based
            # path tolerates; fall back so meshing a set with repeats keeps
            # working exactly as before. The probe is only ever an accelerator.
            return _surface_voxels_sorted(voxels)
    return _surface_voxels_sorted(voxels)


# Bit index in `dijkstra3d_sparse.exposed_faces`' per-voxel mask for each of the
# six returned sets, in tuple order (left, right, back, front, bot, top). The
# mask's bit k is set when the neighbour at FACE_OFFSET[k] is absent; its offset
# order (+x, -x, +y, -y, +z, -z) matches this module's `_FACE_KEY_DELTA`.
_SURFACE_FACE_BITS = (1, 0, 4, 5, 3, 2)  # -x, +x, +z, -z, -y, +y


def _surface_voxels_probe(voxels):
    """Exposed-face sets from `dijkstra3d_sparse.exposed_faces`.

    One native pass answers all six face directions per voxel as a 6-bit mask,
    off a spatial index that is built and freed inside that call (so nothing
    persists into the later mesh stages); splitting the mask into the six
    directional sets is six cheap boolean gathers. Replaces the three
    O(N log N) lexsorts of `_surface_voxels_sorted`. See the module note by
    `_HAVE_EXPOSED_FACES`.
    """
    mask = _d3s.exposed_faces(voxels)
    # Indexing back into `voxels` keeps the input integer dtype.
    return tuple(voxels[(mask & (1 << bit)) != 0] for bit in _SURFACE_FACE_BITS)


def _surface_voxels_sorted(voxels):
    """Sort-based fallback for `find_surface_voxels` (no `dijkstra3d_sparse`)."""
    # Get surface voxels from back to front
    voxels = sort_cols(voxels, order=[0, 1, 2])
    is_front = np.ones(len(voxels), dtype=bool)

    # For each voxel check if previous voxel is in same xy
    same_xy = np.all(voxels[1:, :2] == voxels[:-1, :2], axis=1)

    # For each voxel check the distance along z to previous voxel
    dist_z = voxels[1:, 2] - voxels[:-1, 2]

    # Front voxels are those where the prior voxel is either
    # not in the same xy or is more than one voxel along Z away
    is_front[1:] = ~same_xy | (dist_z > 1)
    voxels_front = voxels[is_front]

    # Do the same for the back voxels
    is_back = np.ones(len(voxels), dtype=bool)
    same_xy = np.all(voxels[:-1, :2] == voxels[1:, :2], axis=1)
    # dist_z = voxels[:-1, 2] - voxels[1:, 2]
    is_back[:-1] = ~same_xy | (dist_z > 1)
    voxels_back = voxels[is_back]

    # Now left to right
    voxels = sort_cols(voxels, order=[2, 1, 0])
    is_left = np.ones(len(voxels), dtype=bool)
    same_yz = np.all(voxels[1:, 1:] == voxels[:-1, 1:], axis=1)
    dist_x = voxels[1:, 0] - voxels[:-1, 0]
    is_left[1:] = ~same_yz | (dist_x > 1)
    voxels_left = voxels[is_left]

    is_right = np.ones(len(voxels), dtype=bool)
    same_yz = np.all(voxels[:-1, 1:] == voxels[1:, 1:], axis=1)
    # dist_x = voxels[:-1, 0] - voxels[1:, 0]
    is_right[:-1] = ~same_yz | (dist_x > 1)
    voxels_right = voxels[is_right]

    # Last but not least: top to bottom
    voxels = sort_cols(voxels, order=[0, 2, 1])
    is_bot = np.ones(len(voxels), dtype=bool)
    same_xz = np.all(voxels[1:, [0, 2]] == voxels[:-1, [0, 2]], axis=1)
    dist_y = voxels[1:, 1] - voxels[:-1, 1]
    is_bot[1:] = ~same_xz | (dist_y > 1)
    voxels_bot = voxels[is_bot]

    is_top = np.ones(len(voxels), dtype=bool)
    same_xz = np.all(voxels[:-1, [0, 2]] == voxels[1:, [0, 2]], axis=1)
    # dist_y = voxels[:-1, 1] - voxels[1:, 1]
    is_top[:-1] = ~same_xz | (dist_y > 1)
    voxels_top = voxels[is_top]

    return (
        voxels_left,
        voxels_right,
        voxels_back,
        voxels_front,
        voxels_bot,
        voxels_top,
    )


def boundary_shell(voxels):
    """The shell of background cells hugging the object's surface.

    Each exposed face (from `find_surface_voxels`) stepped one voxel outward is a
    background cell 6-adjacent to the surface; their union (deduplicated) is the
    set of empty voxels immediately outside the object. The nearest such cell to
    any object voxel is its nearest background voxel, i.e. an exact
    distance-from-boundary (Euclidean distance transform) reference set that stays
    sparse - it never densifies the bounding box. Used by both the skeleton radius
    estimate and the TEASAR distance-from-boundary field.
    """
    left, right, back, front, bot, top = find_surface_voxels(voxels)
    shell = np.vstack(
        [
            right + np.array([1, 0, 0]),
            left + np.array([-1, 0, 0]),
            top + np.array([0, 1, 0]),
            bot + np.array([0, -1, 0]),
            back + np.array([0, 0, 1]),
            front + np.array([0, 0, -1]),
        ]
    )
    return unique(shell, axis=0)


# Packed-key deltas for the six face steps under pack()'s default (21, 21, 21)
# layout: stepping one voxel along an axis just adds a constant to the key (no
# carry between the 21-bit fields as long as coordinates stay in range).
_FACE_KEY_DELTA = np.array(
    [1 << 42, -(1 << 42), 1 << 21, -(1 << 21), 1, -1], dtype=np.int64
)
_POS_FACE_KEY_DELTA = np.array([1 << 42, 1 << 21, 1], dtype=np.int64)  # +x, +y, +z
# The 13 positive-key offsets of the 26-neighbourhood (their mirrors are the
# other 13); an undirected graph only needs one direction per pair.
_POS26_KEY_DELTA = np.array(
    sorted(
        (dx << 42) + (dy << 21) + dz
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx or dy or dz) and (dx << 42) + (dy << 21) + dz > 0
    ),
    dtype=np.int64,
)
_PACK_FIELD = 1 << 21  # per-axis capacity of the packed key


def _sorted_hit(keys, ref_sorted):
    """Boolean mask: which `keys` are present in the sorted array `ref_sorted`."""
    if len(ref_sorted) == 0:
        return np.zeros(len(keys), dtype=bool)
    pos = np.clip(np.searchsorted(ref_sorted, keys), 0, len(ref_sorted) - 1)
    return ref_sorted[pos] == keys


def _component_labels(keys_sorted, connectivity):
    """Connected-component labels for a sorted array of packed cell keys.

    Returns ``(n_components, labels)`` aligned with `keys_sorted`. Prefers the
    optional `dijkstra3d_sparse` accelerator (labels straight off coordinates);
    otherwise builds the sparse graph and uses scipy. Raises if neither is
    available - the topology-safe fill genuinely needs a components routine.
    """
    try:  # accelerated path, straight off the coordinates
        import dijkstra3d_sparse as _d3s

        if hasattr(_d3s, "connected_components"):
            return _d3s.connected_components(unpack(keys_sorted), connectivity=connectivity)
    except Exception:  # pragma: no cover - fall back to scipy on any hiccup
        pass
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError as exc:
        raise ImportError(
            "fill_cavities(mode='exact') needs scipy or dijkstra3d_sparse; "
            "install `sparse-cubes[skeleton]` or `pip install scipy`."
        ) from exc
    deltas = _POS_FACE_KEY_DELTA if connectivity == 6 else _POS26_KEY_DELTA
    m = len(keys_sorted)
    rows, cols = [], []
    for delta in deltas:  # one direction per pair (graph is symmetrised below)
        nb = keys_sorted + delta
        pos = np.clip(np.searchsorted(keys_sorted, nb), 0, m - 1)
        hit = keys_sorted[pos] == nb
        rows.append(np.flatnonzero(hit))
        cols.append(pos[hit])
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r), bool), (r, c)), shape=(m, m))
    return connected_components(graph, directed=False)


def _fill_exact(vox, max_depth, max_cavity_size, verbose):
    """Fill enclosed background voids (see `fill_cavities`, ``mode='exact'``)."""
    # Shift so that a `max_depth`-step outward flood never drives a coordinate
    # (or a +-1 neighbour of it) negative, and guard pack()'s 21-bit fields.
    shift = vox.min(axis=0) - (max_depth + 1)
    v = vox - shift
    ext = int(v.max()) + max_depth + 1
    if ext >= _PACK_FIELD:
        raise ValueError(
            f"Voxel extent (+ flood margin) reaches {ext}, exceeding pack()'s "
            f"{_PACK_FIELD}-per-axis range. Downsample or lower max_depth."
        )
    obj = np.sort(pack(v))
    vmin, vmax = v.min(axis=0), v.max(axis=0)

    shell = np.unique(pack(boundary_shell(vox) - shift))
    shell = shell[~_sorted_hit(shell, obj)]  # shells are background by construction

    # Multi-source BFS over background, `max_depth` layers. Each new layer is a
    # neighbour's shortest-path frontier, so a candidate can only collide with the
    # previous two frontiers - dedup against those, never a growing visited set.
    frontiers = [shell]
    prev, cur = np.empty(0, np.int64), shell
    reached = 0
    for d in range(1, max_depth + 1):
        cand = np.unique((cur[:, None] + _FACE_KEY_DELTA[None, :]).ravel())
        cand = cand[~_sorted_hit(cand, obj)]  # object voxels are walls
        cand = cand[~_sorted_hit(cand, cur)]
        cand = cand[~_sorted_hit(cand, prev)]
        if len(cand) == 0:
            break
        frontiers.append(cand)
        prev, cur = cur, cand
        reached = d

    band = np.concatenate(frontiers)
    depth = np.concatenate(
        [np.full(len(f), i, dtype=np.int32) for i, f in enumerate(frontiers)]
    )
    order = np.argsort(band, kind="stable")
    band, depth = band[order], depth[order]

    n_comp, labels = _component_labels(band, 6)

    # A component is EXTERIOR iff it reaches the cut-off frontier (depth ==
    # max_depth: the flood was still expanding) or steps outside the object's own
    # bbox (definitionally outside). Both are cheap OR-reductions per component.
    # An enclosed void can do neither, so `~exterior` is exactly the voids -
    # crucially there are no false positives, so we never over-fill (fuse/erase
    # topology); the only miss is a void thicker than `max_depth` from its lining.
    coords = unpack(band)
    outside = ((coords < vmin) | (coords > vmax)).any(axis=1)
    at_frontier = depth >= max_depth  # only true when the flood was cut off
    flag = outside | at_frontier
    exterior = np.zeros(n_comp, dtype=bool)
    np.logical_or.at(exterior, labels, flag)
    enclosed = ~exterior
    if max_cavity_size is not None:
        sizes = np.bincount(labels, minlength=n_comp)
        enclosed &= sizes <= max_cavity_size

    # Preserve connected components: a void whose lining spans two *distinct*
    # object components is a pocket trapped *between* them, and filling it would
    # weld them into one. Skip those - leave the pocket - so the fill never
    # changes the object's b0 (never fuses separate branches).
    if enclosed.any():
        vsel = enclosed[labels]
        vcells, vcomp = band[vsel], labels[vsel]
        nb = (vcells[:, None] + _FACE_KEY_DELTA[None, :]).ravel()
        owner = np.repeat(vcomp, len(_FACE_KEY_DELTA))
        pos = np.clip(np.searchsorted(obj, nb), 0, len(obj) - 1)
        hit = obj[pos] == nb  # face-neighbours that are object (the void's lining)
        n_obj, obj_lab = _component_labels(obj, 26)
        lin_comp, lin_lab = owner[hit], obj_lab[pos[hit]]
        lo_lab = np.full(n_comp, n_obj, dtype=np.int64)
        hi_lab = np.full(n_comp, -1, dtype=np.int64)
        np.minimum.at(lo_lab, lin_comp, lin_lab)
        np.maximum.at(hi_lab, lin_comp, lin_lab)
        enclosed &= lo_lab == hi_lab  # single lining component (== also drops liningless)

    void_keys = band[enclosed[labels]]
    log(
        f"fill_cavities: depth {reached}, band {len(band)} cells, "
        f"{int(enclosed.sum())} void(s), {len(void_keys)} cells filled.",
        verbose=verbose,
    )
    if len(void_keys) == 0:
        return vox  # nothing enclosed - return the (int64) input unchanged
    voids = unpack(np.sort(void_keys)) + shift
    return np.vstack([vox, voids])


def _fill_closing(vox, verbose):
    """Fill 1-voxel voids by a radius-1 morphological closing (see `fill_cavities`)."""
    shift = vox.min(axis=0) - 2  # room for the +-1 dilation
    v = vox - shift
    if int(v.max()) + 1 >= _PACK_FIELD:
        raise ValueError("Voxel extent exceeds pack()'s range; downsample first.")
    keys = pack(v)
    dil = np.unique(
        np.concatenate([keys] + [keys + d for d in _FACE_KEY_DELTA])
    )
    keep = np.ones(len(dil), dtype=bool)  # erode: keep cells with all 6 faces set
    for d in _FACE_KEY_DELTA:
        keep &= _sorted_hit(dil + d, dil)
    out = unpack(dil[keep]) + shift
    log(
        f"fill_cavities(closing): {len(vox)} -> {len(out)} voxels.", verbose=verbose
    )
    return out


@sparse_aware(mirror=True)
def fill_cavities(voxels, *, mode="exact", max_depth=8, max_cavity_size=None, verbose=False):
    """Fill enclosed background voids in a sparse ``(N, 3)`` voxel set.

    Topological thinning (`sparsecubes.binary.thin`) preserves enclosed cavities, so a
    void inside the object leaves a thick, un-thinnable "blob" on the centerline.
    Filling the voids first removes those blobs. Everything stays sparse - no
    dense grid is allocated; work scales with the object surface (and flood depth),
    not the bounding-box volume.

    Parameters
    ----------
    voxels :        (N, 3) integer array of XYZ voxel coordinates.
    mode :          {"exact", "closing"}, optional
                    ``"exact"`` (default) fills *only* truly enclosed voids: it
                    seeds a 6-connected background flood from `boundary_shell` and
                    keeps the components walled off from the exterior. It preserves
                    components and tunnels/loops and never fuses nearby branches,
                    but needs scipy (or `dijkstra3d_sparse`) for the component
                    labelling. ``"closing"`` is a fast, dependency-free radius-1
                    morphological closing (dilate then erode); it fills only
                    ~1-voxel voids and **can fuse distinct branches that pass
                    within two voxels of each other** - use with care.
    max_depth :     int, optional
                    (Exact mode.) Flood depth in voxels; the largest void
                    *thickness from its lining* that can be filled is ``max_depth``.
                    Cost scales with `max_depth`. Voids thicker than this are left
                    unfilled (never mis-filled), so raise it only if thick voids
                    persist. Default 8.
    max_cavity_size : int, optional
                    (Exact mode.) If set, enclosed components larger than this many
                    cells are left unfilled (a safety cap). Default ``None``.
    verbose :       bool, optional. Log progress.

    Returns
    -------
    (M, 3) array of the same integer dtype as `voxels` (``M >= N``), deduplicated.
    """
    if not isinstance(voxels, np.ndarray) or voxels.ndim != 2 or voxels.shape[1] != 3:
        raise TypeError("Expected a (N, 3) numpy array of voxel coordinates.")
    if voxels.dtype not in INT_DTYPES:
        raise TypeError(f"Expected integer dtype, got {voxels.dtype}")
    if len(voxels) == 0:
        return voxels

    out_dtype = voxels.dtype
    vox = voxels.astype(np.int64)
    if mode == "closing":
        out = _fill_closing(vox, verbose)
    elif mode == "exact":
        out = _fill_exact(vox, int(max_depth), max_cavity_size, verbose)
    else:
        raise ValueError(f"mode must be 'exact' or 'closing', got {mode!r}")
    return unique(out, axis=0).astype(out_dtype)


def surface_voxel_mask(voxels):
    """Create mask for voxels.

    Parameters
    ----------
    voxels :    (N, 3) numpy arraay

    Returns
    -------
    (voxels_left,
     voxels_right,
     voxels_back,
     voxels_front,
     voxels_bot,
     voxels_top)

    """
    # Get surface voxels from back to front
    srt = argsort_cols(voxels, order=[0, 1, 2])
    vxl_srt = voxels[srt]
    is_front = np.ones(len(voxels), dtype=bool)

    # For each voxel check if previous voxel is in same xy
    same_xy = np.all(vxl_srt[1:, :2] == vxl_srt[:-1, :2], axis=1)

    # For each voxel check the distance along z to previous voxel
    dist_z = vxl_srt[1:, 2] - vxl_srt[:-1, 2]

    # Front voxels are those where the prior voxel is either
    # not in the same xy or is more than one voxel along Z away
    is_front[1:] = ~same_xy | (dist_z > 1)
    voxels_front = srt[is_front]

    # Do the same for the back voxels
    is_back = np.ones(len(voxels), dtype=bool)
    same_xy = np.all(vxl_srt[:-1, :2] == voxels[1:, :2], axis=1)
    is_back[:-1] = ~same_xy | (dist_z > 1)
    voxels_back = srt[is_back]

    # Now left to right
    srt = argsort_cols(voxels, order=[2, 1, 0])
    vxl_srt = voxels[srt]
    is_left = np.ones(len(vxl_srt), dtype=bool)
    same_yz = np.all(vxl_srt[1:, 1:] == vxl_srt[:-1, 1:], axis=1)
    dist_x = vxl_srt[1:, 0] - vxl_srt[:-1, 0]
    is_left[1:] = ~same_yz | (dist_x > 1)
    voxels_left = srt[is_left]

    is_right = np.ones(len(vxl_srt), dtype=bool)
    same_yz = np.all(vxl_srt[:-1, 1:] == vxl_srt[1:, 1:], axis=1)
    is_right[:-1] = ~same_yz | (dist_x > 1)
    voxels_right = srt[is_right]

    # Last but not least: top to bottom
    srt = argsort_cols(voxels, order=[0, 2, 1])
    vxl_srt = voxels[srt]
    is_bot = np.ones(len(vxl_srt), dtype=bool)
    same_xz = np.all(vxl_srt[1:, [0, 2]] == vxl_srt[:-1, [0, 2]], axis=1)
    dist_y = vxl_srt[1:, 1] - vxl_srt[:-1, 1]
    is_bot[1:] = ~same_xz | (dist_y > 1)
    voxels_bot = srt[is_bot]

    is_top = np.ones(len(vxl_srt), dtype=bool)
    same_xz = np.all(vxl_srt[:-1, [0, 2]] == vxl_srt[1:, [0, 2]], axis=1)
    is_top[:-1] = ~same_xz | (dist_y > 1)
    voxels_top = srt[is_top]

    return (
        voxels_left,
        voxels_right,
        voxels_back,
        voxels_front,
        voxels_bot,
        voxels_top,
    )


def make_culled_faces(
    voxels_left, voxels_right, voxels_back, voxels_front, voxels_bot, voxels_top
):
    """Create vertices and faces as culled cube faces (blocky mesh).

    Parameters
    ----------
    voxels_ :   (N, 3) numpy array

    Returns
    -------
    vertices :  (M, 3) array
    faces :     (K, 3) array

    """
    # Vertices left
    verts_left = np.repeat(voxels_left, 4, axis=0)
    verts_left[1::4, 1] += 1
    verts_left[2::4, 2] += 1
    verts_left[3::4, [1, 2]] += 1

    # Faces left
    single_face = [[0, 1, 3], [0, 3, 2]]
    faces_left = np.tile(single_face, (len(voxels_left), 1))
    offsets = np.repeat(np.arange(len(voxels_left)), 2) * 4
    faces_left += offsets.reshape((-1, 1))

    # Vertices right
    verts_right = np.repeat(voxels_right, 4, axis=0)
    verts_right[:, 0] += 1
    verts_right[1::4, 1] += 1
    verts_right[2::4, 2] += 1
    verts_right[3::4, [1, 2]] += 1

    # Faces right
    single_face = [[3, 1, 0], [2, 3, 0]]
    faces_right = np.tile(single_face, (len(voxels_right), 1))
    offsets = np.repeat(np.arange(len(voxels_right)), 2) * 4
    faces_right += offsets.reshape((-1, 1))

    # Vertices bot
    verts_bot = np.repeat(voxels_bot, 4, axis=0)
    verts_bot[1::4, 0] += 1
    verts_bot[2::4, 2] += 1
    verts_bot[3::4, [0, 2]] += 1

    # Faces bot
    single_face = [[3, 1, 0], [2, 3, 0]]
    faces_bot = np.tile(single_face, (len(voxels_bot), 1))
    offsets = np.repeat(np.arange(len(voxels_bot)), 2) * 4
    faces_bot += offsets.reshape((-1, 1))

    # Vertices top
    verts_top = np.repeat(voxels_top, 4, axis=0)
    verts_top[:, 1] += 1
    verts_top[1::4, 0] += 1
    verts_top[2::4, 2] += 1
    verts_top[3::4, [0, 2]] += 1

    # Faces top
    single_face = [[0, 1, 3], [0, 3, 2]]
    faces_top = np.tile(single_face, (len(voxels_top), 1))
    offsets = np.repeat(np.arange(len(voxels_top)), 2) * 4
    faces_top += offsets.reshape((-1, 1))

    # Vertices front
    verts_front = np.repeat(voxels_front, 4, axis=0)
    verts_front[1::4, 0] += 1
    verts_front[2::4, 1] += 1
    verts_front[3::4, [0, 1]] += 1

    # Faces front
    single_face = [[0, 1, 3], [0, 3, 2]]
    faces_front = np.tile(single_face, (len(voxels_front), 1))
    offsets = np.repeat(np.arange(len(voxels_front)), 2) * 4
    faces_front += offsets.reshape((-1, 1))

    # Vertices back
    verts_back = np.repeat(voxels_back, 4, axis=0)
    verts_back[:, 2] += 1
    verts_back[1::4, 0] += 1
    verts_back[2::4, 1] += 1
    verts_back[3::4, [0, 1]] += 1

    # Faces back
    single_face = [[3, 1, 0], [2, 3, 0]]
    faces_back = np.tile(single_face, (len(voxels_back), 1))
    offsets = np.repeat(np.arange(len(voxels_back)), 2) * 4
    faces_back += offsets.reshape((-1, 1))

    # Combine vertices and faces
    faces = [faces_left, faces_right, faces_bot, faces_top, faces_front, faces_back]
    verts = [verts_left, verts_right, verts_bot, verts_top, verts_front, verts_back]

    # Note we need to add another offset to faces
    num_verts = 0
    for v, f in zip(verts, faces):
        f[:] += num_verts
        num_verts += len(v)

    verts = np.vstack(verts)
    faces = np.vstack(faces)

    return verts, faces


# Per-direction geometry for greedy meshing, indexed by dir_id in the order
# (left, right, bot, top, front, back). For each direction:
#   - plane axis: the axis held constant across the face,
#   - uax / vax: the two in-plane axes; uax carries the "corner 1" offset and
#     vax the "corner 2" offset - matching make_culled_faces exactly,
#   - plane offset: +1 for the high-side faces (right / top / back),
#   - tri CCW: whether the quad winds CCW ([[0,1,3],[0,3,2]]) or CW
#     ([[3,1,0],[2,3,0]]).
# A merged rectangle is just a unit quad with its two spanning offsets scaled,
# so reusing this recipe keeps the winding/normals identical to the blocky mesh.
_GREEDY_PLANE_AXIS = np.array([0, 0, 1, 1, 2, 2])
_GREEDY_UAX = np.array([1, 1, 0, 0, 0, 0])
_GREEDY_VAX = np.array([2, 2, 2, 2, 1, 1])
_GREEDY_PLANE_OFFSET = np.array([0, 1, 0, 1, 0, 1])
_GREEDY_TRI_CCW = np.array([True, False, False, True, True, False])
_FACE_CCW = np.array([[0, 1, 3], [0, 3, 2]])
_FACE_CW = np.array([[3, 1, 0], [2, 3, 0]])


def make_greedy_faces(
    voxels_left, voxels_right, voxels_back, voxels_front, voxels_bot, voxels_top
):
    """Create vertices and faces as merged coplanar rectangles (greedy meshing).

    Lossless simplification of the blocky (culled cube faces) mesh: coplanar
    adjacent faces are merged into maximal rectangles, so a flat W x H region
    becomes a single quad instead of W*H quads. Reuses the six directional
    exposed-face sets, stays fully vectorised (no per-voxel loop, no dense grid),
    and keeps integer rectangle corners so the vertex dtype is preserved.

    Each plane's cells are decomposed into rectangles in two passes: a horizontal
    run-length encode (merge along `u` within a row) followed by merging vertically
    adjacent strips of identical `[u0, u1]` extent (merge along `v`). Both passes
    are exact partitions, so the result covers exactly the same surface.

    Note: like all greedy meshing this can introduce T-junctions where a large
    rectangle borders a differently subdivided region, so the result may be
    "less watertight" than the per-face mesh even though the covered area is
    identical.

    Parameters
    ----------
    voxels_ :   (N, 3) numpy array

    Returns
    -------
    vertices :  (M, 3) array
    faces :     (K, 3) array

    """
    orig_dtype = voxels_left.dtype

    # (voxel set, plane_axis, uax, vax) in dir_id order.
    specs = [
        (voxels_left, 0, 1, 2),
        (voxels_right, 0, 1, 2),
        (voxels_bot, 1, 0, 2),
        (voxels_top, 1, 0, 2),
        (voxels_front, 2, 0, 1),
        (voxels_back, 2, 0, 1),
    ]

    d_list, plane_list, u_list, v_list = [], [], [], []
    for dir_id, (V, pax, uax, vax) in enumerate(specs):
        if len(V) == 0:
            continue
        V = V.astype(np.int64, copy=False)
        d_list.append(np.full(len(V), dir_id, dtype=np.int64))
        plane_list.append(V[:, pax])
        u_list.append(V[:, uax])
        v_list.append(V[:, vax])

    if not d_list:
        return (
            np.empty((0, 3), dtype=orig_dtype),
            np.empty((0, 3), dtype=np.int64),
        )

    d = np.concatenate(d_list)
    plane = np.concatenate(plane_list)
    u = np.concatenate(u_list)
    v = np.concatenate(v_list)

    # Pass 1: horizontal run-length encode. A strip breaks where the direction,
    # plane or row (v) changes, or where u is no longer contiguous.
    order = np.lexsort((u, v, plane, d))
    d, plane, u, v = d[order], plane[order], u[order], v[order]
    new_strip = np.ones(len(u), dtype=bool)
    new_strip[1:] = (
        (d[1:] != d[:-1])
        | (plane[1:] != plane[:-1])
        | (v[1:] != v[:-1])
        | (u[1:] != u[:-1] + 1)
    )
    firsts = np.flatnonzero(new_strip)
    lasts = np.r_[firsts[1:] - 1, len(u) - 1]
    s_d, s_plane, s_v = d[firsts], plane[firsts], v[firsts]
    s_u0, s_u1 = u[firsts], u[lasts]

    # Pass 2: merge vertically adjacent strips of identical [u0, u1] extent. A
    # rectangle breaks where direction, plane or the u-extent changes, or where v
    # is no longer contiguous.
    order2 = np.lexsort((s_v, s_u1, s_u0, s_plane, s_d))
    s_d, s_plane, s_u0, s_u1, s_v = (
        s_d[order2],
        s_plane[order2],
        s_u0[order2],
        s_u1[order2],
        s_v[order2],
    )
    new_rect = np.ones(len(s_v), dtype=bool)
    new_rect[1:] = (
        (s_d[1:] != s_d[:-1])
        | (s_plane[1:] != s_plane[:-1])
        | (s_u0[1:] != s_u0[:-1])
        | (s_u1[1:] != s_u1[:-1])
        | (s_v[1:] != s_v[:-1] + 1)
    )
    rf = np.flatnonzero(new_rect)
    rl = np.r_[rf[1:] - 1, len(s_v) - 1]
    r_d = s_d[rf]
    r_plane = s_plane[rf]
    r_u0, r_u1 = s_u0[rf], s_u1[rf]
    r_v0, r_v1 = s_v[rf], s_v[rl]
    R = len(r_d)

    # Emit 4 integer corners + 2 triangles per rectangle, all vectorised.
    pax = _GREEDY_PLANE_AXIS[r_d]
    uax = _GREEDY_UAX[r_d]
    vax = _GREEDY_VAX[r_d]
    poff = _GREEDY_PLANE_OFFSET[r_d]
    rows = np.arange(R)

    base = np.zeros((R, 3), dtype=np.int64)
    base[rows, pax] = r_plane + poff
    base[rows, uax] = r_u0
    base[rows, vax] = r_v0

    uvec = np.zeros((R, 3), dtype=np.int64)
    uvec[rows, uax] = r_u1 - r_u0 + 1  # rectangle width along uax
    vvec = np.zeros((R, 3), dtype=np.int64)
    vvec[rows, vax] = r_v1 - r_v0 + 1  # rectangle height along vax

    verts = np.empty((R * 4, 3), dtype=np.int64)
    verts[0::4] = base
    verts[1::4] = base + uvec
    verts[2::4] = base + vvec
    verts[3::4] = base + uvec + vvec

    single_face = np.where(_GREEDY_TRI_CCW[r_d][:, None, None], _FACE_CCW, _FACE_CW)
    faces = (single_face + (rows * 4)[:, None, None]).reshape(-1, 3)

    return verts.astype(orig_dtype, copy=False), faces


# Ordered lower-corner offsets of the 4 dual cells that share a surface-crossing
# edge, keyed by the edge's axis (0=X, 1=Y, 2=Z). The order is CCW as seen
# looking down the +axis, so a crossing whose filled voxel sits on the low side
# of the edge (positive normal) winds outwards. Derived once per axis via the
# right-hand rule: for axis `a` the in-plane axes (u, v) are the cyclic pair with
# u x v = a, i.e. (Y, Z), (Z, X) and (X, Y).
_CELL_OFFSETS = np.array(
    [
        [[0, -1, -1], [0, 0, -1], [0, 0, 0], [0, -1, 0]],  # X-edge, plane (Y, Z)
        [[-1, 0, -1], [-1, 0, 0], [0, 0, 0], [0, 0, -1]],  # Y-edge, plane (Z, X)
        [[-1, -1, 0], [0, -1, 0], [0, 0, 0], [-1, 0, 0]],  # Z-edge, plane (X, Y)
    ],
    dtype=np.int64,
)


def pack(xyz, bits=(21, 21, 21)):
    """Pack non-negative integer XYZ coordinates into a single int64 key.

    Each axis must be non-negative and fit in its allotted number of bits.
    Used as an exact hash for membership / dedup so we never rely on float
    vertex equality.

    Parameters
    ----------
    xyz :   (N, 3) array of non-negative integers
    bits :  length-3 tuple, optional
            Bits allotted to X, Y and Z respectively. The default ``(21, 21,
            21)`` is byte-identical to the historical ``(x << 42) | (y << 21) |
            z`` layout and keeps the key within a signed 63-bit range. The sum
            must not exceed 63. Widen individual axes (keeping the sum <= 63) to
            support coordinate ranges beyond 2**21 along a given axis.
    """
    _bx, by, bz = bits
    x, y, z = xyz.T.astype(np.int64)
    return (x << (by + bz)) | (y << bz) | z


def unpack(keys, bits=(21, 21, 21)):
    """Inverse of `pack`: recover (N, 3) integer XYZ from packed int64 keys.

    `bits` must match the value used to pack. See `pack` for details.
    """
    _bx, by, bz = bits
    keys = np.asarray(keys, dtype=np.int64)
    z = keys & ((1 << bz) - 1)
    y = (keys >> bz) & ((1 << by) - 1)
    x = keys >> (by + bz)
    return np.stack([x, y, z], axis=1)


def make_surface_nets(
    voxels_left, voxels_right, voxels_back, voxels_front, voxels_bot, voxels_top
):
    """Create vertices and faces using naive SurfaceNets placement.

    Treats voxel occupancy as the samples (outside the set = empty). Each
    exposed voxel face is a surface-crossing edge ``(p, p + e)`` between a
    filled and an empty sample. Every crossing edge is shared by exactly four
    dual cells; each such cell is active (mixed corners) and gets one vertex at
    the centroid of its crossing-edge midpoints. The four cells around an edge
    are joined into a quad, wound by the crossing's sign.

    This reuses the six directional exposed-face sets from
    ``find_surface_voxels`` and stays sparse (memory ~ surface voxels): no dense
    grid and no per-voxel Python loop.

    Parameters
    ----------
    voxels_ :   (N, 3) numpy array

    Returns
    -------
    vertices :  (M, 3) float array
    faces :     (K, 3) int array

    """
    # Unit basis vectors (int64 so subtraction on unsigned inputs can't wrap).
    basis = np.eye(3, dtype=np.int64)

    # Each exposed face is a crossing edge (p, p + e_axis) with `p` the endpoint
    # of lower coordinate along `axis`. sign = +1 when the filled voxel sits at
    # `p` (outward normal points +axis), -1 when it sits at `p + e_axis`.
    specs = [
        (voxels_right, 0, +1),  # +X face of a voxel  -> p = v
        (voxels_left, 0, -1),   # -X face of a voxel  -> p = v - e_x
        (voxels_top, 1, +1),    # +Y face of a voxel  -> p = v
        (voxels_bot, 1, -1),    # -Y face of a voxel  -> p = v - e_y
        (voxels_back, 2, +1),   # +Z face of a voxel  -> p = v
        (voxels_front, 2, -1),  # -Z face of a voxel  -> p = v - e_z
    ]

    p_list, axis_list, sign_list, mid2_list = [], [], [], []
    for V, a, s in specs:
        if len(V) == 0:
            continue
        V = V.astype(np.int64, copy=False)
        e = basis[a]
        p = V if s > 0 else V - e
        p_list.append(p)
        # Midpoint in x2 integer units: 2 * p + e (integer, exact).
        mid2_list.append(2 * p + e)
        axis_list.append(np.full(len(V), a, dtype=np.int64))
        sign_list.append(np.full(len(V), s, dtype=np.int64))

    if not p_list:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
        )

    P = np.vstack(p_list)          # (E, 3) lower point of each crossing edge
    mid2 = np.vstack(mid2_list)    # (E, 3) edge midpoint x2
    axis = np.concatenate(axis_list)  # (E,)
    sign = np.concatenate(sign_list)  # (E,)
    n_edges = len(P)

    # Expand each edge into its 4 sharing dual cells (lower corners), keeping the
    # CCW-for-+normal ordering so we can also use it for winding below.
    cells = P[:, None, :] + _CELL_OFFSETS[axis]  # (E, 4, 3)
    flat_cells = cells.reshape(-1, 3)            # (E*4, 3)

    # Dedup cells -> vertex index via an exact integer key. Shift to keep every
    # coordinate non-negative (stencil corners can reach one below the minimum).
    shift = flat_cells.min(axis=0)
    keys = pack(flat_cells - shift)
    # `cells`/`flat_cells` are the biggest arrays here (~E*4*3 int64) and are done
    # once `keys` exists; free them so the dedup below reuses their footprint
    # rather than adding to the peak.
    del cells, flat_cells
    # Label each incidence by its cell's first appearance in sorted-key order -
    # the same vertex index `np.unique(return_inverse=True)` assigns, but built by
    # hand so each large temporary is freed the moment it is spent. That both
    # avoids the ~E*4 cold binary searches of the old `unique` + `searchsorted`
    # (the searches were 1.4 s of a 2.5 s call) and keeps fewer E*4 arrays live at
    # once than `return_inverse` does, so the dedup peaks *below* the old path.
    order = np.argsort(keys)
    srt = keys[order]
    del keys
    is_new = np.empty(len(srt), dtype=bool)
    is_new[0] = True
    np.not_equal(srt[1:], srt[:-1], out=is_new[1:])  # run boundary = new cell
    del srt
    vidx = np.cumsum(is_new)          # 1..n_cells along the sorted order
    n_cells = int(vidx[-1])
    vidx -= 1
    del is_new
    labels = np.empty(len(vidx), dtype=np.int64)
    labels[order] = vidx             # scatter labels back to incidence order
    del order
    vidx = labels

    # Vertex = mean of the midpoints of the cell's crossing edges. Accumulate the
    # x2-integer midpoints per cell (bincount is faster than np.add.at), then
    # divide by the count and scale by 0.5 to leave x2 units -> float verts.
    mid2_rep = np.repeat(mid2, 4, axis=0)  # (E*4, 3) edge midpoint for each cell
    counts = np.bincount(vidx, minlength=n_cells)
    sums = np.stack(
        [
            np.bincount(vidx, weights=mid2_rep[:, k], minlength=n_cells)
            for k in range(3)
        ],
        axis=1,
    )
    verts = sums / counts[:, None] * 0.5

    # Faces: the 4 cells per edge in winding order (reversed for -normal), split
    # into two triangles.
    quad = vidx.reshape(n_edges, 4)
    neg = sign < 0
    quad[neg] = quad[neg][:, [0, 3, 2, 1]]
    faces = np.empty((n_edges * 2, 3), dtype=np.int64)
    faces[0::2] = quad[:, [0, 1, 2]]
    faces[1::2] = quad[:, [0, 2, 3]]

    return verts, faces


def log(statement, verbose=False, **kwargs):
    """Print statement if verbose is True."""
    if verbose:
        print(statement, **kwargs)
