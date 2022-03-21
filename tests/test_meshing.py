import numpy as np
import sparsecubes as sc


def test_meshing():
    # A single voxel
    voxels = np.array([[1,1,1]])

    m = sc.marching_cubes(voxels)

    assert m.vertices.shape[0] == 8
    assert m.faces.shape[0] == 6 * 2

    # 3 L-shaped voxels
    voxels = np.array([[1,1,1],
                       [1,2,1],
                       [2,1,1]])

    m = sc.marching_cubes(voxels)

    assert m.vertices.shape[0] == 16
    assert m.faces.shape[0] == 14 * 2

    # A simple 2 x 2 x 2 cube
    voxels = np.array([[1,1,1],
                       [2,1,1],
                       [1,2,1],
                       [2,2,1],
                       [1,1,2],
                       [2,1,2],
                       [1,2,2],
                       [2,2,2],
                       ])

    m = sc.marching_cubes(voxels)

    assert m.vertices.shape[0] == 26
    assert m.faces.shape[0] == 24 * 2
