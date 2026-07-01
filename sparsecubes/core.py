import warnings

import numpy as np
import trimesh as tm

try:
    from fastremap import unique
except ImportError:
    from numpy import unique

INT_DTYPES = (np.int64, np.int32, np.int16, np.uint64, np.uint32, np.uint16)


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


def mesh(voxels, spacing=None, step_size=1, verbose=False, smooth=True):
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


def surface_nets(voxels, spacing=None, step_size=1, verbose=False):
    """Smooth (naive SurfaceNets) surface mesh from sparse voxels.

    Thin wrapper around ``mesh(..., smooth=True)``. See `mesh` for details.
    """
    return mesh(
        voxels, spacing=spacing, step_size=step_size, verbose=verbose, smooth=True
    )


def culled_faces(voxels, spacing=None, step_size=1, verbose=False):
    """Blocky (culled cube faces) surface mesh from sparse voxels.

    Thin wrapper around ``mesh(..., smooth=False)``. See `mesh` for details.
    """
    return mesh(
        voxels, spacing=spacing, step_size=step_size, verbose=verbose, smooth=False
    )


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


def pack(xyz):
    """Pack non-negative integer XYZ coordinates into a single int64 key.

    Each axis must be non-negative and < 2**21. Used as an exact hash for
    membership / dedup so we never rely on float vertex equality.
    """
    x, y, z = xyz.T.astype(np.int64)
    return (x << 42) | (y << 21) | z


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
    cell_keys = np.unique(keys)              # sorted unique cell keys
    n_cells = len(cell_keys)
    vidx = np.searchsorted(cell_keys, keys)  # (E*4,) vertex index per incidence

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
