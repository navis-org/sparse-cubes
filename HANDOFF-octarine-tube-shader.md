# Handoff: a parametric-tube shader for `octarine`

> **STATUS: proposed, not implemented.** The producer side is done and shipped:
> `sparsecubes.tube` (`sc.tube_coefficients`, `TubeProfile.to_gpu_buffer()`, added
> 2026-08-04) emits a buffer that is ready to bind and evaluate as-is — no
> conversion, no repacking, no reordering on the octarine side. Nothing in
> sparse-cubes is blocked on this. What is missing is the consumer: a pygfx
> `WorldObject`/`Material`/`Shader` triple in `octarine` that draws the tube
> **without ever materializing a mesh**.
>
> | | mesh of the same arbor | **parametric tube (this document)** |
> |---|---|---|
> | GPU bytes | O(triangles) | **1.93 MB** (30,226 nodes x 64 B) |
> | angular LOD | remesh / swap buffers | a uniform + a smaller draw call |
> | axial LOD | remesh / swap buffers | swap a 240 KB index buffer |
> | rebuild after a proofread | remesh | re-extract coefficients (~7 s) |
>
> Numbers from `tests/10075_scale3.npy` (5.56 M voxels -> 30,226 nodes /
> 30,188 edges), `benchmarks/bench_tube.py --neuron`.

**Audience:** whoever implements this in `~/Github/octarine`.
**Requested by:** `sparse-cubes`, specifically `sparsecubes/tube.py::TubeProfile`.

The ask is a **vertex-pulling** surface shader: bind the coefficients as a storage
buffer, generate every vertex position from `vertex_index` alone. No vertex
buffer, no index buffer, no geometry upload, and — the point of the whole exercise
— **LOD selection becomes a uniform change rather than a buffer swap.**

---

## 1. Why vertex pulling, and why now

`octarine` already has the harder version of this. `octarine/shaders/sparse_volume.py`
draws a whole sparse volume from two textures and `{"indices": (36, 1)}` — 36
vertices, no geometry, positions synthesized in `vs_main` from an implicit cube.
This is the same trick applied to a different generator function, and it is
*simpler*: no raycasting, no brick map, no atlas.

The constraint that shapes the design: **WebGPU has no tessellation stage and mesh
shaders are not in core**, so the NeuroTessMesh-style hardware-tessellation
approach is unavailable under pygfx/wgpu. Vertex pulling is the way to get
resolution-independent geometry there.

What that buys, concretely:

- `N_theta` is a **uniform** plus a draw-call size. Going from 32 angular samples
  to 8 is one `uniform_buffer` write and a smaller `get_render_info` — no upload,
  no reallocation, no LOD bookkeeping touching memory at all.
- The coefficient buffer is uploaded **once** and never moves, at any level of
  detail. On the neuron that is 1.93 MB total.
- Regenerating after proofreading is a re-extract, not a remesh.

This is explicitly a **coarse-LOD** representation. The intended end state is
hybrid: parametric tube for coarse levels (tiny, streams instantly, correct
silhouette), a real chunked mesh at the finest level where measurements are taken.
Do not scope this as a replacement for the mesh path.

---

## 2. The data contract

Everything comes from a `sparsecubes.TubeProfile`. Two arrays are needed.

### 2a. Per-node coefficients — `profile.to_gpu_buffer()`

```python
buf, header = profile.to_gpu_buffer()          # form="cartesian" is the default
# buf    : (M, 8 + 2K) float32, C-contiguous
# header : {"K": 4, "n_theta": 64, "n_nodes": 30226, "stride_floats": 16,
#           "spacing": (16.0, 16.0, 16.0), "form": "cartesian"}
```

Row layout, in floats:

| offset | count | meaning |
|---|---|---|
| `0` | 3 | `pos.xyz` — node position, **already in physical units** (spacing applied) |
| `3` | 4 | `quat.xyzw` — frame rotation, `w >= 0`, unit |
| `7` | 1 | `a0` — mean radius, physical units |
| `8` | `K` | `a_1 .. a_K` — cosine coefficients |
| `8 + K` | `K` | `b_1 .. b_K` — sine coefficients |

At `K = 4` that is 16 floats = **64 bytes a node**, so

```
r(theta) = a0 + sum_k [ a_k cos(k*theta) + b_k sin(k*theta) ]
```

**Assert on `header["form"]`** rather than trusting the default — `form="polar"`
puts `m_1..m_K` / `phi_1..phi_K` in the same slots, and the two are
indistinguishable by shape.

**Positions are physical**, so the visual needs no spacing scale — set only the
world offset. This differs from `SparseVolume`, which sets `local.scale_*` from
the voxel spacing; do not copy that part.

**Frame unpack.** The quaternion is the rotation whose *columns* are `(u, v, t)`:

```wgsl
fn frame_uv(q: vec4<f32>) -> mat2x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    let u = vec3<f32>(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + z*w), 2.0*(x*z - y*w));
    let v = vec3<f32>(2.0*(x*y - z*w), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z + x*w));
    return mat2x3<f32>(u, v);   // t = cross(u, v) if ever needed
}
```

`u` and `v` span the cross-section plane; `t` is the skeleton tangent.

### 2b. Topology — `profile.edges`

```python
profile.edges   # (E, 2) int32, index pairs into the nodes
```

**This is not optional and it is not a node-index range.** A skeleton is a tree,
not a chain — the neuron has 30,226 nodes in **6,295 branches**. Sweeping quads
between consecutive *node indices* would stitch unrelated branches together. Quads
must be built per **edge**.

### 2c. Why the buffer is Cartesian and the storage is not

Worth understanding, because the two forms look interchangeable and are not.

`TubeProfile` *stores* magnitude/phase (`profile.mag`, `profile.phase`,
`save_npz`) for two reasons that both matter and neither of which is about
rendering. `m_k` is **frame-independent** — the rotation-minimizing frame is only
defined up to a constant rotation about the tangent, which mixes `a_k` and `b_k`
into each other but leaves `m_k` alone — so it is the only form a truncation
threshold can legitimately be applied to. And `phi_k` varies smoothly along a
branch, so it delta-codes.

`to_gpu_buffer()` *emits* Cartesian, because a shader wants the other one:

```
sum_k [ a_k cos(k*theta) + b_k sin(k*theta) ]
```

walks `cos(k*theta)` and `sin(k*theta)` upward by angle addition, so the whole
series costs **one `cos` and one `sin` for any `K`** — and `dr/dtheta`, i.e. the
surface normal, falls out of the same loop for free (§3b). The polar form would
need a `cos(phi_k)` and a `sin(phi_k)` per harmonic per vertex, which is the
entire cost of the loop, paid again, at every vertex.

So: **nothing to decide and nothing for octarine to convert.** Call
`to_gpu_buffer()` and index `a_k` at offset `8 + (k-1)`, `b_k` at
`8 + K + (k-1)`. `form="polar"` exists for a consumer that genuinely needs
magnitudes on the GPU — thresholding or fading harmonics per node, say — and is
not what this shader wants.

---

## 3. The shader

### 3a. Vertex-index decode

Draw `6 * E * N_theta` vertices, 1 instance. Each quad is one (edge, angular
sector) cell of the tube surface:

```wgsl
let vid   = i32(in.vertex_index);
let quad  = vid / 6;
let corner = vid % 6;
let e = quad / N_THETA;          // which skeleton edge
let j = quad % N_THETA;          // which angular sector

// Two triangles: (0,0) (1,0) (0,1) and (1,0) (1,1) (0,1)
var ends = array<i32,6>(0, 1, 0,  1, 1, 0);
var offs = array<i32,6>(0, 0, 1,  0, 1, 1);

let edge = load_s_edges(e);                    // vec2<i32>
let node = select(edge.x, edge.y, ends[corner] == 1);
let jj   = (j + offs[corner]) % N_THETA;
let theta = 2.0 * PI * f32(jj) / f32(N_THETA);
```

`N_THETA` and `K_MAX` come from the material uniform, **not** from a template
constant — that is what makes angular LOD a uniform change. (`K_MAX` may be
lowered below the buffer's `K` for angular truncation; the buffer stride stays
`{{ stride }}`, which *is* a template constant since it is fixed at upload.)

### 3b. Profile evaluation — `r` and `dr/dtheta` from one loop

```wgsl
const STRIDE: i32 = {{ stride }};   // 8 + 2*K of the uploaded buffer
const KBUF:   i32 = {{ k_buf }};    // K of the uploaded buffer

// Returns (r, dr/dtheta) at `theta` for node `i`.
fn eval_profile(i: i32, theta: f32, kmax: i32) -> vec2<f32> {
    let base = i * STRIDE;
    var r  = load_s_coefs(base + 7);   // a0
    var dr = 0.0;

    let c = cos(theta);
    let s = sin(theta);
    var ck = 1.0;      // cos(0*theta)
    var sk = 0.0;      // sin(0*theta)

    for (var k = 1; k <= kmax; k = k + 1) {
        // Angle addition: advance to k*theta with no further trig calls.
        let t = ck * c - sk * s;
        sk = sk * c + ck * s;
        ck = t;

        let a = load_s_coefs(base + 7 + k);
        let b = load_s_coefs(base + 7 + KBUF + k);
        r  = r  + a * ck + b * sk;
        dr = dr + f32(k) * (b * ck - a * sk);
    }
    return vec2<f32>(r, dr);
}
```

`dr` falls straight out of the same loop — `d/dtheta [a cos(k t) + b sin(k t)] =
k (b cos(k t) - a sin(k t))` — which is what makes analytic normals free.

### 3c. Surface point and normal

```wgsl
struct SurfacePoint { pos: vec3<f32>, dp_dtheta: vec3<f32>, e_r: vec3<f32> };

fn surface_point(i: i32, theta: f32, kmax: i32) -> SurfacePoint {
    let base = i * STRIDE;
    let p = vec3<f32>(load_s_coefs(base), load_s_coefs(base+1), load_s_coefs(base+2));
    let q = vec4<f32>(load_s_coefs(base+3), load_s_coefs(base+4),
                      load_s_coefs(base+5), load_s_coefs(base+6));
    let uv = frame_uv(q);
    let c = cos(theta); let s = sin(theta);
    let e_r = c * uv[0] + s * uv[1];       // radial
    let e_t = -s * uv[0] + c * uv[1];      // tangential (d e_r / d theta)

    let rr = eval_profile(i, theta, kmax);
    var out: SurfacePoint;
    out.pos       = p + rr.x * e_r;
    out.dp_dtheta = rr.y * e_r + rr.x * e_t;
    out.e_r       = e_r;
    return out;
}
```

For the normal, take the axial direction from the *other* end of the same edge, at
the same `theta`. That is exact for the quad being drawn and, unlike using the
stored tangent `t`, it accounts for the radius changing along the tube (boutons
and varicosities are exactly this — a local bump in `a0` along the axis, not an
angular phenomenon):

```wgsl
let here  = surface_point(node, theta, kmax);
let other = surface_point(select(edge.y, edge.x, ends[corner] == 1), theta, kmax);
var n = normalize(cross(here.dp_dtheta, other.pos - here.pos));
n = select(-n, n, dot(n, here.e_r) > 0.0);    // point outward
```

That is two profile evaluations per vertex. At `K = 4` it is ~8 fused multiply-adds
each — cheaper than the memory traffic a real mesh would need.

#### Truncate the normal harder than the surface — this one is not optional

`kmax` above is deliberately *not* the same uniform for the position and the
normal. Add a second, `kmax_normal`, and use it for the `eval_profile` call that
feeds `dp_dtheta` (and for both `surface_point`s in the axial difference; the
normal must describe one consistent surface). The position keeps the full `kmax`.

The reason is in the derivative: `dr/dtheta` weights harmonic `k` by `k`, so once
the `m_k` flatten out at the rasterization floor — which they do, there is no knee
in this data — every extra harmonic contributes more slope than shape. Measured on
the 16 nm arbor, the normal's median tilt away from radial:

```
kmax_normal      0      1      2      4      6
median tilt   12.6   23.9   32.2   35.4   37.0   deg
p95           63.2   64.3   67.0   69.1   70.8   deg
```

The silhouette keeps improving with `kmax` while the shading gets worse. This is
honest geometry rather than a bug — a neurite of ~5-voxel radius really is that
bumpy — but shaded at `kmax_normal = kmax = 4` it looks like sandpaper, and the
first read is "my normals are broken". They are not. Default `kmax_normal` to 1
and expose it; `kmax_normal = 0` is the smooth-tube floor.

The residual 12.6° at `kmax_normal = 0` is the axial term, and it is mostly real:
smoothing `a0` three times along the branch only takes it to 10.0°. Do not chase
it.

`TubeProfile.evaluate(k=..., k_normal=..., return_normals=True)` implements
exactly this and is the reference to difference against (§10).

Then the standard pygfx varyings, exactly as `sparse_volume.wgsl` does it:

```wgsl
let world_pos = u_wobject.world_transform * vec4<f32>(here.pos, 1.0);
let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
varyings.position = vec4<f32>(ndc_pos);
varyings.world_pos = vec3<f32>(world_pos.xyz);
varyings.normal = vec3<f32>(normalize((u_wobject.world_transform * vec4<f32>(n, 0.0)).xyz));
```

Fragment stage: `{$ include 'pygfx.light_phong_simple.wgsl' $}` and call
`lighting_phong(...)`, the same include pygfx's own simple-lit paths use. Nothing
custom is needed there.

---

## 4. The pygfx plumbing

Follow `octarine/shaders/sparse_volume.py` structurally; it is the closest
precedent and it already solves the "no geometry" problem.

```python
# octarine/shaders/tubes.py
class TubeVisual(gfx.WorldObject): ...
class TubeMaterial(gfx.MeshPhongMaterial): ...

@register_wgpu_render_function(TubeVisual, TubeMaterial)
class TubeShader(BaseShader):
    type = "render"
```

**Bindings.** Both buffers are read-only storage, visible to `VERTEX`.
`octarine/shaders/lines.py:124` is the existing example of appending one:

```python
bindings = [
    Binding("u_stdinfo",  "buffer/uniform", shared.uniform_buffer),
    Binding("u_wobject",  "buffer/uniform", wobject.uniform_buffer),
    Binding("u_material", "buffer/uniform", material.uniform_buffer),
    Binding("s_coefs", "buffer/read_only_storage", geometry.coefs, "VERTEX"),
    Binding("s_edges", "buffer/read_only_storage", geometry.tube_edges, "VERTEX"),
]
```

**Buffer shapes matter, and this is the one place it is easy to get wrong.**
`_define_buffer` in `pygfx/renderers/wgpu/shader/bindings.py` supports only 1, 2,
3 or 4 channels and generates a `load_<name>(i: i32)` accessor from the buffer's
inferred format. Verified against pygfx 0.16.0 / wgpu 0.31.0:

| you pass | inferred format | generated WGSL |
|---|---|---|
| `(M*STRIDE,)` f32 | `f4` | `array<f32>`, `load_s_coefs(i) -> f32` |
| `(M, 16)` f32 | `16xf4` | **`ValueError: Unexpected vertex format '16xf4'`** |
| `(E, 2)` i32 | `2xi4` | `array<vec2<i32>>`, `load_s_edges(i) -> vec2<i32>` |
| `(M*4, 4)` f32 | `4xf4` | `array<vec4<f32>>`, `load_s_coefs(i) -> vec4<f32>` |

So `s_coefs` must be **flattened** — `gfx.Buffer(buf.ravel())`. Handing pygfx the
`(M, 16)` array that `to_gpu_buffer()` returns raises at pipeline build time, not
at bind time, so the traceback points somewhere unhelpful.

(An optimisation once it works: when `(8 + 2K) % 4 == 0` — true at `K = 4` — bind
as `(M * STRIDE/4, 4)` float32 and read a `vec4<f32>` at a time, which is the
efficient path on most GPUs. Start flat; it is the version that works for every
`K`.)

**Uniforms.** Extend the material's uniform struct so LOD is a uniform:

```python
uniform_type = dict(gfx.MeshPhongMaterial.uniform_type, n_theta="i4", k_max="i4")
```

with `n_theta` / `k_max` properties writing `self.uniform_buffer.data[...]` and
calling `update_full()`, exactly like `SparseVolumeMaterial.step_size`
(`sparse_volume.py:100-111`).

**Draw size.**

```python
def get_render_info(self, wobject, shared):
    return {"indices": (6 * wobject.n_edges * wobject.material.n_theta, 1)}
```

**Pipeline.** Start with `"cull_mode": wgpu.CullMode.none` — the quad winding
depends on the frame handedness and is easy to get backwards, and `none` renders
correctly either way. Switch to `back` only after verifying the winding, as an
optimisation.

**Bounding box.** pygfx derives it from `geometry.positions`, so give it two
corner points covering node positions expanded by the largest possible radius
(`a0 + sum(mag)` bounds `r(theta)` from above):

```python
reach = (profile.a0 + profile.mag.sum(axis=1)).max()
corners = np.array([verts.min(0) - reach, verts.max(0) + reach], dtype=np.float32)
geometry = gfx.Geometry(positions=corners, coefs=..., tube_edges=...)
```

Without this the viewer's auto-centering will frame the wrong volume.

**Import lazily.** `octarine/shaders/__init__.py` already guards `pygfx>=0.16` and
is imported on first use, not at package import. Add `TubeVisual`/`TubeMaterial`
to its `__all__` and keep the lazy import in the converter, as
`visuals.py:501-509` does for `SparseVolume`.

---

## 5. The octarine-side API

Mirror the sparse-volume plumbing, which is four touch points:

1. **`octarine/shaders/tubes.py`** — the three classes above.
2. **`octarine/shaders/wgsl/tube.wgsl`** — the shader.
   `MANIFEST.in` already ships `octarine/shaders/wgsl/*.wgsl`; nothing to change.
3. **`octarine/visuals.py::tubes2gfx(profile, color=..., n_theta=..., k=...)`** —
   builds the geometry/material/visual, sets `vis._object_type = "tubes"` and
   `vis._object_id`.
4. **`octarine/viewer.py::Viewer.add_tubes(...)`** — the usual `name`/`group`/
   `center` handling and `self._add_to_scene(vis, center)`.

**Do not import sparse-cubes.** octarine must not gain that dependency. Duck-type
it the way `utils.is_volume` / `is_points` already do:

```python
def is_tube_profile(x):
    return all(hasattr(x, a) for a in ("a0", "mag", "phase", "frame", "edges",
                                       "to_gpu_buffer"))
```

and register it in `conversion.CONVERTERS` so `Viewer.add(profile)` just works.
`add_tubes` should also accept the raw arrays, for callers who have coefficients
from somewhere else.

---

## 6. LOD: what is actually free, and what is not

Be precise about this, because the two axes are **not** symmetric.

**Angular (`n_theta`, `k_max`) — genuinely free.** A uniform write plus a
different `get_render_info` count. No upload, no reallocation. This is the claim
the whole design rests on and it should be demonstrated in a test.

**Axial — a small buffer swap, not free.** Subsampling nodes means a *different
edge list*, since edges connect specific node pairs. The honest framing: the
1.93 MB coefficient buffer never moves; only a ~240 KB (E x 2 int32) index buffer
does. Precomputing three or four levels costs ~1 MB total. That is still orders of
magnitude better than swapping meshes, but it is a buffer swap and should not be
described otherwise.

(A swap-free variant exists — a per-node `level` byte plus a "next surviving node"
pointer, collapsing quads to zero area above the threshold — but it needs another
buffer and careful handling of gaps. Not worth it for a first implementation.)

**Which axis matters more.** Boutons and varicosities are *not* an angular
phenomenon — they are a local bump in `a0` along the axis. Measured on the neuron,
the angular spectrum has a `k=2` spike (ellipticity) and then decays smoothly with
no knee, and at `K=4` the truncation residual is already down at the rasterization
floor (~0.54 voxels). So **axial resampling dominates the error budget for most
arbors**; angular truncation matters mainly for large-calibre proximal dendrites.
Do not over-invest in the angular axis.

---

## 7. Known artifacts — expected, not bugs

**Junction seams.** Rotation-minimizing frames are propagated per branch and
seeded independently at each branch's proximal end, so the frames on the two sides
of a bifurcation are related by an arbitrary rotation about the tangent. An edge
crossing a branch boundary will therefore have a twisted quad ring.

*This is mostly invisible and should not be fixed here.* Interpenetrating opaque
tubes from incident branches look fine under a z-buffer — correct depth, no
visible seam at typical zoom. It only bites for transparency, watertight export,
or volume/area measurement, none of which is this shader's job. Every
generalized-cylinder method fights this; the extractor deliberately flags it
rather than solving it:

```python
profile.flag("junction")    # at, or adjacent to, a degree>=3 node
```

Worth exposing as an option to *dim* or *hide* junction-adjacent quads for
debugging, not to correct them.

**Non-star-shaped cross-sections.** `r(theta)` cannot represent a cross-section
that is not star-shaped about its skeleton point. Measured: 63.5% of interior
non-junction nodes have at least one violating ray, but only **4.4% of rays**
violate — so the failure is widespread but shallow, and shows up as a locally
flattened silhouette rather than as anything structural. `profile.non_star` gives
the per-node severity (fraction of rays), which is a natural thing to offer as a
debug colormap.

**`a0` is a mean, not a max.** A node whose rays escaped (`profile.flag("ray_escaped")`,
12.4% on the neuron) has an over-estimated radius. Also worth a debug colormap.

---

## 8. Tests

`octarine/tests/test_visuals.py` already has the pattern: offscreen viewer,
`v.screenshot(filename=None, size=(300, 300))`, assert on pixel statistics
(`test_mesh_silhouette_render:83` is the model).

Synthetic fixtures with known answers, so the test does not depend on sparse-cubes:

- **A straight circular tube** — `a0 = R`, every `a_k`/`b_k` zero, positions along
  +z, frames constant. Rendered head-on it must be a filled disc of the right
  radius; side-on, a band of width `2R`. This catches the frame unpack, the `e_r`
  construction and the position scale in one go.
- **`a_2` only** — `r = a0 + a_2 cos(2*theta)`, an elliptic tube. Side-on width
  must differ by `2*a_2` between two camera rolls 90 degrees apart. Catches the
  harmonic loop and, because `a_2` alone puts the long axis at `theta = 0`, the
  `u` half of the frame unpack. Repeat with `b_2` only, which rotates it 45
  degrees, to catch the `v` half — swapping `u` and `v` is otherwise invisible on
  a circular tube.
- **Normals** — a lit sphere-like tube must be bright facing the camera and dark
  at the silhouette. Catches the `cross` order and the outward flip; getting the
  normal inverted is the most likely single bug and is invisible in a wireframe.
- **A tapered tube** — `a0` growing linearly along a straight centreline, every
  harmonic zero. The normal must tilt off radial by exactly the taper angle. This
  is the one fixture where the shader's one-sided axial difference and
  `evaluate`'s centred one agree exactly (a cone differenced either way is the
  same cone), so it can be asserted to tolerance rather than by eye — and it is
  the test that fails if anyone "simplifies" the normal to use the stored tangent.
- **`kmax_normal` is separate from `kmax`** — render the same fixture at
  `(kmax=4, kmax_normal=4)` and `(kmax=4, kmax_normal=1)`: the silhouettes must be
  pixel-identical and the shading must not be. Assert it, or the two uniforms will
  get quietly collapsed back into one.
- **`n_theta` is a uniform** — render at `n_theta=8` and `n_theta=64`, assert the
  silhouettes converge and that **the coefficient buffer object is unchanged**
  (`vis.geometry.coefs is buf_before`). This is the property the design exists for;
  assert it explicitly or it will regress silently.
- **`k_max` truncation** — `k_max=0` must render a circular tube of radius `a0`
  whatever the harmonics say.
- **Branch topology** — a Y-shaped skeleton must render three arms; a test that
  quads are built per *edge* and not per node-index gap. Build the fixture with a
  deliberately non-consecutive node ordering so an index-range implementation
  fails it.
- **Empty / single node / single edge** — must not crash and must not divide by
  zero in `get_render_info`.

---

## 9. Non-goals

- **No transparency work.** The junction artifact is invisible only under opaque
  z-buffered rendering; making tubes transparent exposes it and is out of scope.
- **No watertight export.** This is a renderer, not a mesher. If someone wants a
  mesh, `TubeProfile.evaluate()` already returns ring points on the CPU — that is
  the CPU mirror of this shader's loop and the right place to build one.
- **No soma handling.** Somata are not tube-like and are excluded upstream.
- **No picking / selection** in the first pass. Vertex-pulled geometry needs its
  own pick-id path; add it only if `octarine.selection` actually needs to hit
  tubes.
- **No dependency on sparse-cubes.** Duck-type the profile (§5).
- **Not a replacement for the mesh path.** Coarse LOD only (§1).

---

## 10. Cross-cutting note

`TubeProfile.evaluate(n_theta=..., k=..., nodes=...)` in
`sparsecubes/tube.py` is deliberately the **CPU mirror of `vs_main`** — same
recurrence, same two LOD knobs, same `p = pos + r * (cos(theta)*u + sin(theta)*v)`.
Use it as the reference while bringing the shader up: render a small fixture both
ways and difference the point sets. It is much faster than debugging in WGSL, and
if the two ever disagree, one of them is wrong.

**The normal is mirrored too**: `evaluate(return_normals=True)` returns
`(points, normals)`, unit length and outward, with the same `k_normal` knob §3c
asks you to add. Use it rather than reverse-engineering the definition in WGSL —
an inverted or mis-truncated normal is the single most likely bug here and the
hardest to see.

It is not a bit-exact mirror, and the difference is worth knowing before you write
the comparison:

- **`dp/ds` is centred on the CPU, one-sided in the shader.** `evaluate` differences
  the node's branch predecessor against its successor; a shader pulling one quad
  per edge only has that edge's two ends. The two disagree by O(curvature × node
  spacing), and the shader's version is discontinuous between the two quads
  meeting at a ring — that faceting is expected. A disagreement *larger than the
  local turn angle*, or one that does not shrink as the fixture's centreline
  straightens, is a real bug.
- **Junctions are meaningless in both.** Frames are seeded per branch, so `theta`
  denotes different directions on either side of a junction and the axial
  difference across one is noise. Those nodes carry `FLAG_JUNCTION`; exclude them
  from any comparison, and expect the artifact §9 already declines to fix.

A practical bring-up order: check positions first with normals off, then check
`dot(n, e_r) > 0` everywhere (catches the outward flip), then compare directions
against `evaluate` on a *straight, tapered* fixture — where the centred and
one-sided differences coincide exactly, so any disagreement at all is yours.
