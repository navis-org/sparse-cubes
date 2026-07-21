# sparse-cubes

Fast, memory-efficient operations on sparse voxel data:
`(N, 3)` arrays of voxel indices - i.e. the 3D equivalent of a sparse matrix in
COOrdinate (COO) format.

Everything works *directly* on the sparse voxel coordinates - no dense 3D grid is
ever allocated. Memory scales with the number of (surface) voxels rather than the
volume's bounding box, so `sparse-cubes` handles large, thin, low-occupancy
objects (e.g. neurons spanning a huge bounding box) that would be wasteful to
densify for `scikit-image` (marching cubes / thinning) or `kimimaro`.

## Features

- **Meshing** - turn surface voxels into a mesh, either **smooth** (SurfaceNets)
  or **blocky** (culled cube faces à la Minecraft).
- **Voxelization** - the inverse: turn a triangle mesh into sparse voxels,
  **solid** (filled interior) or surface-only.
- **Lossless simplification** - merge coplanar blocky faces into maximal
  rectangles (greedy meshing), typically ~2x fewer triangles.
- **Thinning** - peel voxels down to a 1-voxel-wide, topology-preserving medial
  curve.
- **Centerline skeletons** - extract a node/edge graph (with radii) from thinned voxels; export to
  SWC / networkx / trimesh.
- **TEASAR skeletons** - well-centered medial-axis skeletons with radii, a sparse
  reimplementation of [`kimimaro`](https://github.com/seung-lab/kimimaro).
- **Primitives** - morphology (dilate/erode/open/close), set algebra, connected
  components and measurements, in `sparsecubes.binary` and `sparsecubes.measure`.
- **Adjacency & downsampling** - the voxel graph as an explicit edge list, and
  pooling onto a coarser lattice (optionally connectivity-safe).
- **Sparse-array interop** - every voxel-taking function also accepts a 3-D
  `scipy.sparse.coo_array` and hands one back where that makes sense.


![example mesh](./_static/example_mesh.png)
*Example using a set of 789M voxels, meshed in 8:40mins on an M3 MacBook with 32GB memory. The resulting mesh has 177M faces.*

## Install

Install latest version from PyPI:

```bash
pip3 install sparse-cubes -U
```

To install the developer version from Github:

```bash
pip3 install git+https://github.com/navis-org/sparse-cubes.git
```

The only required dependencies are `numpy` and `trimesh`. Will use `fastremap` if
present. Optional extras:

- `pip install sparse-cubes[recommended]` - the
  [`dijkstra3d-sparse`](https://pypi.org/project/dijkstra3d-sparse/) accelerator,
  which considerably speeds up `teasar_skeletonize`.
- `pip install sparse-cubes[skeleton]` - scipy (for `teasar_skeletonize` and
  `radii=True`) plus the recommended `dijkstra3d-sparse` accelerator.
- `pip install sparse-cubes[graph]` - networkx (for `to_networkx`).

## Quickstart

Meshing:

```python
>>> import sparsecubes as sc
>>> import numpy as np
>>> # Indices for two adjacent voxels
>>> voxel_xyz = np.array([[0, 0, 0],
...                       [0, 0, 1]],
...                      dtype='uint32')
>>> # Smooth (SurfaceNets) mesh by default; vertices are floats
>>> m = sc.mesh(voxel_xyz)
>>> m
<trimesh.Trimesh(vertices.shape=(12, 3), faces.shape=(20, 3))>
>>> m.is_winding_consistent
True
>>> # Pass smooth=False (or call sc.culled_faces) for the blocky, integer mesh
>>> m_blocky = sc.mesh(voxel_xyz, smooth=False)
>>> # ...and simplify=True (or sc.greedy_faces) to merge coplanar faces losslessly
>>> m_small = sc.mesh(voxel_xyz, smooth=False, simplify=True)
```

Voxelization (the inverse of `sc.mesh`):

```python
>>> import trimesh as tm
>>> m = tm.creation.icosphere(subdivisions=3, radius=10)
>>> # Solid by default: surface + filled interior
>>> vox = sc.voxelize(m, spacing=1.0)
>>> vox.shape
(4169, 3)
>>> # ...or just the surface shell
>>> shell = sc.voxelize(m, spacing=1.0, solid=False)
>>> # Anisotropic voxels are fine, and the result feeds straight back in
>>> vox = sc.voxelize(m, spacing=(1.0, 1.0, 2.0))
>>> skel = sc.thin_skeletonize(sc.voxelize(m, 1.0))
```

Primitives:

```python
>>> # Morphology and set algebra (voxels in -> voxels out)
>>> grown = sc.binary.dilate(voxels, iterations=2)
>>> clean = sc.binary.opening(voxels)          # strip specks and thin spurs
>>> both = sc.binary.intersection(voxels, other)
>>> # Labelling and measurements (voxels in -> numbers/labels out)
>>> n, labels = sc.measure.connected_components(voxels)
>>> body = sc.measure.largest_component(voxels)
>>> sc.measure.volume(voxels, spacing=(1, 1, 2))
>>> sc.measure.distance_transform(voxels)      # exact, sparse EDT
```

Skeletonization:

```python
>>> # `thin` peels the object to a 1-voxel medial curve (a subset of the input)
>>> thinned = sc.binary.thin(voxels)
>>> # `thin_skeletonize` thins and extracts the centerline graph in one step
>>> skel = sc.thin_skeletonize(voxels, min_branch_length=3, radii=True)
>>> # ...or trace a well-centered TEASAR medial-axis skeleton
>>> skel = sc.teasar_skeletonize(voxels, spacing=(1, 1, 1), min_branch_length=3)
>>> skel.nodes            # (M, 3) voxel coordinates
>>> skel.edges            # (K, 2) undirected node-index pairs
>>> skel.radii            # (M,) distance-to-boundary per node (needs scipy)
>>> skel.to_swc("cell.swc")          # SWC table (navis/NEURON-friendly)
```

---

## Meshing modes

`sparse-cubes` finds the exposed faces of your voxels and turns them into a
mesh. There are two ways to place the vertices, selected with the `smooth`
flag on `mesh()` (or via the explicit `surface_nets()` / `culled_faces()`
functions):

- **Smooth (`sc.mesh(voxels)` / `sc.surface_nets(voxels)`, the default).** A
  naive [SurfaceNets](https://0fps.net/2012/07/12/smooth-voxel-terrain-part-2/)
  pass: one vertex per surface cell, placed at the centroid of the surface
  crossings around it. This is a *dual* method (a cousin of dual contouring) and
  smooths the staircase you would otherwise get on diagonal surfaces. Vertices
  are floats.
- **Blocky (`sc.mesh(voxels, smooth=False)` / `sc.culled_faces(voxels)`).** Each
  exposed voxel face becomes an axis-aligned quad with corners on the integer
  voxel grid ("culled cube faces", à la Minecraft). Fast and keeps the input
  integer dtype, but diagonal surfaces come out as 90° steps. This is the
  historical output.

### Optional simplification (blocky only)

Pass `simplify=True` (or use `sc.greedy_faces(voxels)`) to merge coplanar faces
of the blocky mesh into maximal rectangles
([greedy meshing](https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/)):

```python
>>> full = sc.mesh(voxels, smooth=False)
>>> small = sc.mesh(voxels, smooth=False, simplify=True)  # ~2x fewer triangles
```

This is **lossless** - the covered surface is identical - and keeps the integer
vertex dtype. It typically roughly halves the triangle count (a flat W×H wall
becomes a single quad instead of W·H quads) at little to no extra cost. Caveat:
like all greedy meshing it can introduce T-junctions, so the simplified mesh may
be "less watertight" than the per-face mesh; it is opt-in for that reason.

Please see [this blog](https://www.boristhebrave.com/2018/04/15/dual-contouring-tutorial/) for an excellent introduction to dual contouring and SurfaceNets.
See also notes at the end of the README.

`sc.dual_contour` and `sc.marching_cubes` still exist as **deprecated aliases**
of `sc.mesh` (their old `interpolate` argument maps to `smooth`) but emit a
`DeprecationWarning` - neither name ever described what this library actually
does.

## Voxelization

`sc.voxelize` is the inverse of `sc.mesh`: it rasterizes a `trimesh.Trimesh` (or a
`(vertices, faces)` pair) into the same `(N, 3)` integer representation, in two
stages that both stay sparse.

The **surface** stage is an exact conservative rasterization - a voxel is emitted
iff the triangle genuinely intersects its cube, decided by a separating-axis test
rather than by point sampling. The **interior** stage is a scanline parity fill:
each triangle is rasterized in the XY projection over voxel column centres, the
resulting Z crossings are sorted per column and paired up, and the cells between
a pair are emitted as runs. Memory is proportional to the crossings plus the
output, and the even-odd rule means face winding is irrelevant (meshes with
inconsistent normals still work) and enclosed cavities are correctly left empty.

```python
>>> vox = sc.voxelize(mesh, spacing=1.0)               # solid
>>> vox = sc.voxelize(mesh, spacing=1.0, solid=False)  # surface shell only
>>> vox = sc.voxelize(mesh, spacing=(0.5, 0.5, 1.0))   # anisotropic
```

Voxel `i` along an axis covers `[(i - 0.5) * spacing, (i + 0.5) * spacing)`, so
its *centre* is at `i * spacing`. This matches trimesh's `VoxelGrid` convention
and makes the round trip line up: `sc.mesh(sc.voxelize(m, s), spacing=s)` lands
back on top of the original mesh. Indices are absolute and may be negative.

**Why not just use trimesh?** `mesh.voxelized(pitch)` already returns sparse
*surface* voxels without densifying, though it approximates - it subdivides faces
and keeps the cells containing the resulting vertices, so it misses cells a
triangle only clips through a corner. The gap is solid voxelization: every fill
path in trimesh materializes the full bounding box (`fill('holes')` runs
`scipy.ndimage.binary_fill_holes` on a dense array, and `fill('base')` allocates a
cube of the largest coordinate), which is exactly what this library exists to
avoid. `sc.voxelize` fills sparsely, so peak memory tracks the object rather than
its bounding box.

If a mesh is not watertight some columns cannot be paired up. Those are left
unfilled and a warning names how many; either repair the mesh first
(`trimesh`'s `fill_holes`) or pass `solid=False`.

## Thinning, centerline & TEASAR skeletons

The same sparse machinery can *thin* voxels down to a one-voxel-wide medial
curve and extract a **centerline skeleton** (a node/edge graph), or trace a
**TEASAR** medial-axis skeleton (the algorithm behind
[`kimimaro`](https://github.com/seung-lab/kimimaro)). Like the meshing, both run
directly on the `(N, 3)` coordinates - no dense grid is ever allocated - so they
work on large, sparse objects (e.g. neurons spanning a huge bounding box at low
occupancy) that would be wasteful to densify for `scikit-image`'s thinning or
`kimimaro`'s dense distance transform.

```python
>>> import sparsecubes as sc
>>> # `thin` peels the object to a 1-voxel medial curve (a subset of the input)
>>> thinned = sc.binary.thin(voxels)
>>> # `thin_skeletonize` thins and extracts the centerline graph in one step
>>> skel = sc.thin_skeletonize(voxels, min_branch_length=3, radii=True)
>>> skel.nodes            # (M, 3) voxel coordinates
>>> skel.edges            # (K, 2) undirected node-index pairs
>>> skel.radii            # (M,) distance-to-boundary per node (needs scipy)
>>> skel.node_degrees()   # 1 = tip, 2 = along a path, >=3 = branch point
>>> skel.to_swc("cell.swc")          # SWC table (navis/NEURON-friendly)
>>> skel.to_networkx()               # networkx.Graph (needs networkx)
>>> skel.to_path3d()                 # trimesh.path.Path3D for visualisation
```

`thin` uses topological thinning (Lee/Palágyi-style simple-point removal with
sub-field-parallel deletion) and **preserves topology** - connected components
and loops are kept, endpoints are not eroded. It matches
`skimage.morphology.skeletonize(..., method="lee")` topologically but stays
sparse.

For a well-centered medial-axis skeleton with clean radii, use `teasar_skeletonize`
(a sparse reimplementation of TEASAR / `kimimaro`). It roots the object at its
geodesically furthest point and traces shortest paths - through a penalty field
that hugs the centerline - to the most distant remaining voxel, invalidating a
distance-scaled tube around each path. Every stage (distance-from-boundary field,
geodesic distances, path finding, invalidation) runs on the sparse voxels via
`scipy` KD-trees and `scipy.sparse.csgraph`, so memory scales with the voxel count
- never the bounding-box volume `kimimaro`'s dense EDT would need.

```python
>>> skel = sc.teasar_skeletonize(voxels, spacing=(1, 1, 1), min_branch_length=3)
>>> skel.radii            # (M,) distance-from-boundary (medial radius) per node
>>> skel.to_swc("cell.swc")
```

The output is the same `Skeleton` object. Note TEASAR always returns an **acyclic
tree/forest** - loops are broken (an annulus becomes an open curve), matching SWC
conventions - whereas `thin` preserves loops. The invalidation ball radius is
`scale * DBF + const`; `const` is in physical units (defaults to ~4 voxels), so
unlike `kimimaro`'s nanometre-scale default of 300 it is sensible in index space.

The `branching` parameter dials the speed/fidelity tradeoff (all yield an acyclic
tree):

- `branching="exact"` (default) - one shortest-path search per path, grafting each
  branch onto the skeleton (`kimimaro`'s `fix_branching`). Most faithful, but
  `O(paths)` Dijkstra runs, so it gets slow on very large objects.
- `branching="tree"` - reuse a single root Dijkstra tree. Fastest, but junctions
  are coarser.
- `branching="fast"` - a multi-source variant that grafts a batch of paths per
  search: a middle ground, roughly an order of magnitude faster than `"exact"` on
  large objects and slightly coarser. Pass an int to set the batch size explicitly
  (larger is faster and coarser).

```python
>>> skel = sc.teasar_skeletonize(big_voxels, branching="tree")   # fastest
>>> skel = sc.teasar_skeletonize(big_voxels, branching="fast")   # middle ground
```

**Scope / when to use something else.** Topological thinning (`thin`) preserves
loops but is sensitive to surface noise and sprouts spurs (prune with
`min_branch_length`); TEASAR (`teasar_skeletonize`) gives smoother, well-centered
paths with radii but breaks loops and is slower on very large objects (pure-`scipy`
Dijkstra). Both shine on large, thin, sparse structures - the same regime as the
rest of `sparse-cubes`. For small/fat solids, densifying and calling
`scikit-image` / `kimimaro` directly is simpler and faster.

The distance-from-boundary KD-tree query is threaded by default (`workers=-1`).
It is purely a speed knob - the skeleton is identical either way - and the same
parameter is on `sc.measure.distance_transform`. Set `workers=1` when you are
already parallelizing over objects yourself, e.g. inside a `multiprocessing`
pool, where the default would oversubscribe the CPUs:

```python
>>> skel = sc.teasar_skeletonize(voxels)             # all cores (default)
>>> skel = sc.teasar_skeletonize(voxels, workers=1)  # single-threaded
```

`teasar_skeletonize` transparently uses [`dijkstra3d-sparse`](https://pypi.org/project/dijkstra3d-sparse/)
when it is installed to run Dijkstra straight over the voxel coordinates, which is
markedly faster than the pure-`scipy` `csgraph` fallback. It is **optional but highly
recommended** - `sparse-cubes` falls back to scipy without it. It ships with the
`skeleton` extra, or install it on its own with `pip install sparse-cubes[recommended]`.

## Primitives: `binary` and `measure`

The top level carries the end-to-end pipelines (`mesh`, `voxelize`,
`*_skeletonize`). The primitives they are built from live in two submodules, split
by what they return:

- **`sparsecubes.binary`** - voxel set(s) in, voxel set out.
- **`sparsecubes.measure`** - voxel set in, numbers or labels out.

| `sc.binary` | | `sc.measure` | |
| --- | --- | --- | --- |
| `dilate` / `erode` | grow / shrink by a neighbourhood | `connected_components` | `(n, labels)`, row-aligned |
| `opening` / `closing` | strip specks / bridge gaps | `largest_component` | biggest blob only |
| `union` / `intersection` | set algebra over clouds | `remove_small_objects` | despeckle by voxel count |
| `difference` / `symmetric_difference` | subtraction / XOR | `volume` / `surface_area` | with optional `spacing` |
| `isin` / `index_of` | per-row membership / row lookup | `bounding_box` / `centroid` | index bounds / centre of mass |
| `thin` / `fill_cavities` | topological thinning / void fill | `distance_transform` | exact sparse EDT |
| | | `iou` / `dice` | set similarity of two clouds |

```python
>>> import sparsecubes as sc
>>> clean = sc.binary.opening(voxels)                 # drop surface noise
>>> body = sc.measure.largest_component(clean)        # keep the main object
>>> skel = sc.teasar_skeletonize(body)                # then the usual pipeline
```

All of it stays sparse. `dilate`/`erode` accept `connectivity=6|18|26` and
`iterations=n` with the same semantics as `scipy.ndimage`, and the morphology is
tested to agree with it voxel-for-voxel - the difference is that `scipy` needs the
bounding box densified first and these do not. Likewise `measure.distance_transform`
returns exactly what `scipy.ndimage.distance_transform_edt` would, computed from
the sparse background shell instead of a dense grid.

Two caveats worth knowing. `closing` can **fuse** structures that pass within
`2 * iterations` voxels of each other; when you specifically want enclosed voids
filled without that risk, `fill_cavities(mode="exact")` is topology-safe.
And `connected_components` supports `connectivity=6` or `26` only (not 18), since
the underlying routine does not distinguish 18 from 26.

> **Moved in 0.4.0.** `sc.thin` and `sc.fill_cavities` are now `sc.binary.thin`
> and `sc.binary.fill_cavities` - they are primitives, not pipelines. The old
> names raise an `AttributeError` naming the new spelling.

### Carrying per-voxel values

The primitives map coordinates to coordinates; real image data carries a value
per voxel. `isin` re-aligns values through the *shrinking* operations (`erode`,
`thin`, `difference`), where every output row came from the input:

```python
>>> small = sc.binary.erode(voxels)
>>> small_values = values[sc.binary.isin(voxels, small)]
```

For the *growing* ones (`dilate`, `union`, `fill_cavities`) some output rows are
new, so you need to know where each one came from - that is `index_of`, which
returns the row index in the source or `-1`:

```python
>>> grown = sc.binary.dilate(voxels)
>>> src = sc.binary.index_of(grown, voxels)      # -1 for the newly added voxels
>>> grown_values = np.where(src >= 0, values[src], fill)
```

## Adjacency and downsampling

Three operations change what the voxel set *is* - its graph, or its lattice -
rather than which voxels are in it, so they sit at the top level:

```python
>>> nodes, edges = sc.edges(voxels, connectivity=26)  # the voxel graph
>>> coarse = sc.downsample(voxels, 2)                 # pool onto a coarser grid
>>> coarse, coarse_edges = sc.downsample_graph(voxels, 2)   # connectivity-safe
```

`edges` returns the deduplicated, sorted `nodes` plus `(E, 2)` index pairs into
them, canonical (`lo < hi`) and deduplicated - one entry per undirected edge.
It is the primitive underneath `centerline`, exposed for handing to networkx /
igraph or injecting via `teasar_skeletonize(edges=...)`. It walks *positive*
packed-key deltas only, so each undirected edge is found exactly once, and costs
one `searchsorted` per delta - no KD-tree, no dense neighbour block.

`downsample` pools voxels into `factor`-sized cells (`v // factor`,
deduplicated) - the sparse counterpart of `scipy.ndimage.zoom` on a dense grid.
`factor` may be a length-3 tuple for anisotropic pooling. Because several fine
voxels collapse into one coarse cell, per-voxel data has to be *reduced* rather
than re-indexed, which it will do for you:

```python
>>> coarse, coarse_values = sc.downsample(voxels, 2, values=intensity, agg="max")
```

`agg` is `"max"` (default; preserves peaks), `"min"`, `"mean"` or `"sum"`.
Integer values accumulate in `int64` for `sum`/`mean`, so pooling `uint8`
intensities does not wrap around. Remember to scale any `spacing` you carry
alongside by `factor`.

The catch with plain pooling is that it can **fuse** structures less than
`2 * factor` apart - adjacency is implicit in the coarse coordinates, so cells at
`(0,0,0)` and `(1,0,0)` read as connected whether or not anything joined them.
When that matters (skeletonizing, counting components), use `downsample_graph`,
which returns the coarse cells *plus an explicit edge list* lifted from the fine
26-connectivity graph. No connection is introduced that did not exist, and the
connected-component partition is preserved exactly. Feed the edges on rather than
re-deriving them from the coarse geometry - re-deriving would reintroduce the
very links it avoided:

```python
>>> coarse, coarse_edges = sc.downsample_graph(voxels, 2)
>>> skel = sc.teasar_skeletonize(coarse, edges=coarse_edges, spacing=spacing * 2)
```

## Working with `scipy.sparse` arrays

An `(N, 3)` index array and a 3-D sparse array are the same thing in different
clothing - a COO volume *is* a list of occupied coordinates - so every
voxel-taking function accepts either:

```python
>>> from scipy.sparse import coo_array
>>> vol = coo_array((data, (xs, ys, zs)), shape=(512, 512, 128))
>>> sc.mesh(vol)                        # -> Trimesh
>>> sc.measure.volume(vol)              # -> float
>>> sc.binary.dilate(vol)               # -> coo_array, modelled on the input
>>> sc.teasar_skeletonize(vol)          # -> Skeleton
```

Operations that return a **voxel set** (`binary.*`, `measure.largest_component`,
`measure.remove_small_objects`) give you a `coo_array` back, with the input's
dtype. Those returning something with no sparse form - a mesh, a `Skeleton`,
labels, a scalar - return it unchanged. Mixing is fine: one sparse argument is
enough, so `sc.binary.union(sparse_a, ndarray_b)` returns sparse.

Three things worth knowing:

- **scipy stays optional.** `sparse-cubes` never imports it to make this work.
  Detection is duck-typing on the argument's type, and the module is only fetched
  from `sys.modules` - which a sparse argument proves is already populated. The
  overhead on the ordinary ndarray path is about 1.5 µs per call.
- **3-D COO only.** scipy supports 3-D in the COO format alone (CSR/DOK/LIL are
  still 2-D as of scipy 1.15). Passing a 2-D matrix raises - it cannot represent
  a volume. Note that n-D `coo_array` itself needs scipy >= 1.15, i.e. Python
  >= 3.10; on 3.9 there is no 3-D sparse array to pass in the first place. The
  rest of `sparse-cubes` is unaffected.
- **The shape is a floor, not a clamp.** An operation that grows the object past
  the array's bounds widens the shape to fit rather than truncating, so no voxel
  is ever silently dropped. The exception is growth *below* index 0, which a
  sparse array simply cannot represent and which raises:

  ```python
  >>> vol = coo_array(..., shape=(4, 4, 4))   # object touching index 0
  >>> sc.binary.dilate(vol)
  ValueError: Result contains negative coordinates (min (-1, -1, -1)) ...
  ```

  Pad the array first, or pass an `(N, 3)` index array - those are unbounded and
  handle negative coordinates natively.

## Notes
- The mesh might have non-manifold edges. Trimesh will report these
  meshes as not watertight but in the very literal definition they do hold water.
- The names `dual_contour` / `marching_cubes` were misnomers: the blocky path is
  really culled cube faces (vertices only ever land on cube corners) and the
  smooth default is naive dual/SurfaceNets placement. Full feature-preserving
  dual contouring (QEF-based placement using surface normals) is not
  implemented.
