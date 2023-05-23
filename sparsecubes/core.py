import numpy as np
import trimesh as tm

try:
    from fastremap import unique
except ImportError:
    from numpy import unique


# The original Trimesh automatically casts vertices to float64. Realistically,
# our meshes should be fine with uint32 most of the time so we will avoid the
# casting altogether and hope we don't break anything in the process.
# This should speed things up quite a bit and obviously also reduce the
# memory footprint
class Trimesh(tm.Trimesh):
    @property
    def vertices(self):
        return self._data.get('vertices', np.empty(shape=(0, 3), dtype=np.uint32))

    @vertices.setter
    def vertices(self, values):
        self._data['vertices'] = np.asanyarray(values, order='C')


def marching_cubes(voxels, spacing=None, step_size=1, verbose=False):
    """Marching cubes algorithm to find surfaces in 2d voxel data.

    Parameters
    ----------
    voxels :    (N, 3) array of integers
                XYZ voxel coordinates (indices) to find isosurfaces for. The
                data type should carry over to the mesh vertices - i.e. if
                `voxels` are unsigned 32-bit integers, the mesh vertex
                coordinates should also end up being unsigned 32-bit integers.
    spacing :   length-3 tuple of floats, optional
                Voxel spacing in spatial dimensions corresponding to numpy array
                indexing dimensions as in `voxels`.
    step_size : int, optional
                Step size in voxels. Default 1. Larger steps yield coarser
                results.
    verbose :   bool
                If True, will provide some feedback on progress.

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

    if not isinstance(spacing, type(None)):
        spacing = np.array(spacing)

    # Step size is implemented by simple downsampling
    if step_size and step_size > 1:
        voxels = unique(voxels // step_size, axis=0)

        if not isinstance(spacing, type(None)):
            spacing = spacing * step_size

    # Find surface voxels
    log('Looking for surface voxels... ', end='', flush=True, verbose=verbose)
    (voxels_left,
     voxels_right,
     voxels_back,
     voxels_front,
     voxels_bot,
     voxels_top) = find_surface_voxels(voxels)
    log('Done.', flush=True, verbose=verbose)

    # Generate vertices + faces
    log('Generating vertices and faces... ', end='', flush=True, verbose=verbose)
    verts, faces = make_verts_faces(voxels_left,
                                    voxels_right,
                                    voxels_back,
                                    voxels_front,
                                    voxels_bot,
                                    voxels_top)
    log('Done.', flush=True, verbose=verbose)

    # Create mesh
    log('Making mesh... ', end='', flush=True, verbose=verbose)
    m = Trimesh(verts, faces, process=False)
    log('Done.', flush=True, verbose=verbose)

    # Collapse vertices
    log('Merging vertices... ', end='', flush=True, verbose=verbose)
    merge_vertices(m)
    log('Done.', flush=True, verbose=verbose)

    # Apply spacing after we collapse duplicate vertices
    if not isinstance(spacing, type(None)):
        m.vertices = m.vertices * spacing

    log('All done!', flush=True, verbose=verbose)
    return m


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
    #dist_z = voxels[:-1, 2] - voxels[1:, 2]
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
    #dist_x = voxels[:-1, 0] - voxels[1:, 0]
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
    #dist_y = voxels[:-1, 1] - voxels[1:, 1]
    is_top[:-1] = ~same_xz | (dist_y > 1)
    voxels_top = voxels[is_top]

    return (voxels_left,
            voxels_right,
            voxels_back,
            voxels_front,
            voxels_bot,
            voxels_top)


def make_verts_faces(voxels_left,
                     voxels_right,
                     voxels_back,
                     voxels_front,
                     voxels_bot,
                     voxels_top):
    """Create vertices and faces from surface voxels.

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


def log(statement, verbose=False, **kwargs):
    """Print statement if verbose is True."""
    if verbose:
        print(statement, **kwargs)
