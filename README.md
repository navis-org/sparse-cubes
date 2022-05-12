# sparse-cubes
Marching cubes for `(N, 3)` voxel indices - i.e. the equivalent of a 3D
sparse matrix in COOrdinate format.

Running marching cubes directly on sparse voxels is faster and importantly much
more memory efficient than converting to a 3d matrix and using the implementation
in e.g. `sklearn`.

The only dependencies are `numpy` and `trimesh`. Will use `fastremap` if present.

## Install

Install latest version from PyPI:

```
pip3 install sparse-cubes -U
```

To install developer version from Github:

```
pip3 install git+https://github.com/navis-org/sparse-cubes.git
```

## Usage

```python
>>> import sparsecubes as sc
>>> import numpy as np
>>> # Indices for two adjacent voxels
>>> voxel_xyz = np.array([[0, 0, 0],
...                       [0, 0, 1]],
...                      dtype='uint32')
>>> m = sc.marching_cubes(voxel_xyz)
>>> m
<trimesh.Trimesh(vertices.shape=(12, 3), faces.shape=(20, 3))>
>>> m.is_winding_consistent
True
```

## Notes
- The mesh might have non-manifold edges. Trimesh will report these
  meshes as not watertight but in the very literal definition they do hold water.
- Currently only full edges.
