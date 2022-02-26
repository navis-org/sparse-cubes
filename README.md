# sparse-cubes
Marching cubes for sparse matrices - i.e. `(N, 3)` voxel data.

Running marching cubes directly on sparse voxels is faster and importantly much
more memory efficient than converting to a 3d matrix and using the implementation
in e.g. `sklearn`.

The only dependencies are `numpy` and `trimesh`. Will use `fastremap` if present.

## Install

```
pip3 install git+https://github.com/navis-org/sparse-cubes.git
```

## Usage

```python
>>> import sparsecubes as sc
>>> import numpy as np
>>> voxels = np.array([[0, 0, 0], [0, 0, 1]])
>>> m = sc.marching_cubes(voxels)
>>> m
<trimesh.Trimesh(vertices.shape=(12, 3), faces.shape=(20, 3))>
>>> m.is_winding_consistent
True
```

## Notes
- Currently the mesh might have non-manifold edges. Trimesh will report these
  meshes as not watertight but in the very literal definition they do hold water.
