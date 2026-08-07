# Handoff: a `ray_exits` primitive for `dijkstra3d_sparse`

> **STATUS: implemented, wired in and measured — 2026-08-04.** `ray_exits` and
> `Graph.ray_exits` shipped in **dijkstra3d-sparse 0.3.0**, which is now
> sparse-cubes' minimum, and `sparsecubes/tube.py` uses them unconditionally.
> §4 records what was measured rather than what was projected; everything above it
> is kept as the specification that was built to.
>
> **It delivered.** On the 5.56 M-voxel neuron (73,671 TEASAR nodes, 4.71 M rays):
>
> | `tube_profile` | diagnostics=False | diagnostics=True |
> |---|---|---|
> | `backend="keys"` (searchsorted) | 9.42 s | 14.47 s |
> | `backend="sparse"` (numpy DDA + `index_of`) | 3.48 s | 5.26 s |
> | **`backend="exits"`** | **0.93 s** | **1.42 s** |
>
> All three are pinned bit-for-bit against each other by
> `tests/test_tube.py::test_ray_exits_backend_matches_the_numpy_walk`.
>
> Two findings worth carrying back, both covered below: the walk itself now runs at
> the bulk-`index_of` probe floor (§4), and once it did, the caller's *own*
> `unique(axis=0)` dedup became four times the cost of the raycast (§3).

**Audience:** whoever implements this in `~/Github/dijkstra3d-sparse`.
**Requested by:** `sparse-cubes`, specifically `sparsecubes/tube.py::_raycast`.

---

## 1. Why

`tube_profile` stores, per skeleton node, the cross-section profile `r(θ)` as a
truncated Fourier series. Getting `r(θ)` means firing `n_theta` rays in the plane
normal to the skeleton tangent and taking each one's first exit from the object.

The walk is inherently **serial per ray** and inherently **ragged**: every ray
finishes after a different number of cells. numpy can only express that by
marching all rays in lockstep and compacting the survivors each step.

### Measured, on `tests/10075_scale3.npy` (5,560,148 voxels → 30,226 nodes)

`n_theta=64`, so 1,934,464 rays:

```
                       rays        probes  steps/ray    wall   ns/probe
diagnostics=False  1,934,464   12,343,316       6.38   5.73 s      464
diagnostics=True   1,934,464   21,300,854      11.01   6.50 s      305
```

Two things in there are the whole argument.

**The per-probe cost is 305 ns against a 22 ns floor.** That floor is measured, on
the same index, same machine: 21,300,000 probes issued as one bulk
`Graph.index_of` call take 0.468 s. So ~93% of the raycast is *not* the lookups —
it is the `argmin` over an `(R, 3)`, the fancy-index gathers and scatters to
advance one axis, the boolean masks, the six-array compaction, and one Python↔Rust
crossing, all of which touch the whole live set to advance each ray by **one
cell**.

**The loop length is set by the longest ray, not the typical one.** Per 1 M-ray
block:

```
outer-loop iterations                                    131
mean live set                                         87,826
after 50% of iterations, still live                        981  (0.10% of block)
after 75% of iterations, still live                         66  (0.01%)
after 90% of iterations, still live                         10  (0.00%)
iterations running with <1% of the block live    88 of 131 (67%), carrying 0.8% of probes
```

The median ray is done in ~11 steps; two thirds of the iterations exist to finish
a handful of stragglers. A scalar loop pays for a 131-step ray with 131 steps. The
lockstep version pays with 131 whole-block iterations.

### Why not fix it caller-side

There is nothing left to vectorise; the lockstep-plus-compaction shape is already
the best numpy can do.

- **Bigger or smaller blocks don't help.** The cost is per *live ray*, and the
  block size only trades peak memory against the same total work.
- **Fewer steps aren't available.** Amanatides-Woo visits exactly the cells the
  ray passes through, which is the minimum for an exact first exit. Fixed-step
  sampling would be *more* work for *less* accuracy — `tests/_tube_oracle.py` is
  precisely that, kept as an independent oracle, and it is orders of magnitude
  slower.
- **The probe is already Rust.** `Graph.index_of` is the fast part. Swapping it
  for `np.searchsorted` (the pure-numpy fallback backend) only takes 6.50 s to
  11.93 s — a 1.8x spread on a cost that is 93% bookkeeping. That ratio is the
  cleanest evidence that the probe is not the problem.

### Reproduce

```bash
python benchmarks/bench_tube.py --neuron --backends sparse,keys
```

For the probe floor and the loop-tail statistics, instrument `tube._raycast`'s
`inside` callback and count calls / live-set sizes; both measurements are three
lines each.

---

## 2. The primitive — `ray_exits`

Walk each ray from its origin and report the parametric distances at which it
crosses the object's boundary.

### Interface

```python
def ray_exits(
    voxels: npt.ArrayLike,        # (N, 3) integer voxel coordinates
    origins: npt.ArrayLike,       # (R, 3) float64, index space
    directions: npt.ArrayLike,    # (R, 3) float64, index space, need not be unit
    *,
    max_dist: npt.ArrayLike,      # scalar or (R,) float64
    max_crossings: int = 1,
    index_kind: str = "hash",
) -> tuple[np.ndarray, np.ndarray]: ...
    # t       (R, max_crossings) float64, +inf padded
    # n_hits  (R,) int32

class Graph:
    def ray_exits(self, origins, directions, *, max_dist, max_crossings=1): ...
```

**Semantics.** Voxel centres sit on integer coordinates, so cell `c` occupies
`[c-0.5, c+0.5)` on each axis. The ray is `p(t) = origins[r] + t * directions[r]`
for `t >= 0`. A **crossing** is a value of `t` at which `p` moves from an occupied
cell to an empty one, or back. Crossings are reported in increasing `t`, strictly
alternating:

- `t[r, 0]` — the first **exit**. This is the radius the caller wants.
- `t[r, 1]` — the first **re-entry** after it, if any.
- `t[r, 2]` — the next exit. And so on.

`n_hits[r]` is how many entries of `t[r]` are valid; the rest are `+inf`.

**The `Graph` method is the primary API here** — note the inversion relative to
`exposed_faces`, where the *free* function was the win because it could free the
index inside the call. A caller casting rays casts them in chunks against one
voxel set, so the index must persist across calls; rebuilding it per chunk would
dominate everything. Ship both, but `Graph.ray_exits` is the one `sparse-cubes`
will call.

**`index_kind` genuinely matters here**, unlike in `exposed_faces` and `factorize`
where it is accepted for signature parity and documented as inert. A ray walk is a
stream of unpredictable point probes with no exploitable ordering — exactly the
case `Hash` exists for. `Sorted` must work and must give identical output, but
expect it to be measurably slower and say so in the docstring rather than implying
parity.

**Contract details to honour:**

- **The origin cell is assumed occupied and is never reported.** If it is empty,
  return `n_hits[r] = 0`. The caller distinguishes that case with one `index_of`
  probe per *node* (not per ray) — it already does, to set its `FLAG_SEED_OUTSIDE`
  — so the primitive does not need a separate signal, and adding one would cost a
  branch in the hot loop for a case the caller has already handled.
- **`max_dist` bounds `t`, not the cell count.** A ray reaching it without
  crossing gets `n_hits[r] = 0`; the caller reads that as "escaped". **Per-ray
  `max_dist` is required, not a convenience:** the caller scales it by each node's
  own radius, and a single global value would either clip fat dendrites or let
  thin ones run for thousands of cells.
- **`directions` are index-space and need not be normalised**, deliberately. The
  caller passes a physically-unit direction divided by the voxel spacing, so `t`
  comes back as a *physical* distance with no rescaling — and the primitive stays
  purely geometric, with no `anisotropy` parameter. Spacing is the caller's
  metric, not the library's. (Contrast `dijkstra_field`, which does own an
  `anisotropy`; that is a different question — edge weights in a graph — and the
  two should not be made to look alike.)
- **Zero direction components** never cross that axis (`t_delta = inf`). An
  all-zero direction gives `n_hits[r] = 0` rather than an error or a hang.
- `max_crossings >= 1`. Walking stops at `max_crossings` **or** `max_dist`,
  whichever comes first.
- **Non-finite input** (`NaN`/`inf` in `origins` or `directions`) must be rejected
  in the Python wrapper, not allowed to reach the loop — a `NaN` in `t_max` makes
  every comparison false and the termination test would never fire.
- Duplicate coordinates in `voxels`: follow the existing index constructor's
  policy (raise, via `SpatialIndex::build`). The caller deduplicates with
  `core.unique` first.
- `R == 0` or `N == 0` returns well-formed empty arrays, not an error.
- Determinism: a pure function of the inputs.
- **Coordinates fit `i32`.** A ray is *expected* to leave the occupied region, so
  stepping off the end of the coordinate range is a reachable path here rather
  than a theoretical one. Range-check before casting back, exactly as
  `dijkstra.rs:133-145` already does for neighbour offsets.

### Implementation

Amanatides & Woo (1987), one ray at a time, straight against `SpatialIndex::get`.
New module `src/raycast.rs`:

```rust
use crate::index::{key_of, SpatialIndex};

#[inline(always)]
fn argmin3(v: &[f64; 3]) -> usize {
    let a = if v[0] <= v[1] { 0 } else { 1 };
    if v[a] <= v[2] { a } else { 2 }
}

/// Walk one ray; write crossings into `out` and return how many there were.
pub fn ray_exits_one(
    origin: [f64; 3],
    dir: [f64; 3],
    max_dist: f64,
    index: &SpatialIndex,
    out: &mut [f64],
) -> i32 {
    let mut cur = [0i32; 3];
    let mut step = [0i32; 3];
    let mut t_max = [f64::INFINITY; 3];
    let mut t_delta = [f64::INFINITY; 3];

    for a in 0..3 {
        cur[a] = origin[a].round() as i32;
        if dir[a] != 0.0 {
            step[a] = if dir[a] > 0.0 { 1 } else { -1 };
            t_delta[a] = (1.0 / dir[a]).abs();
            // Next face crossing on this axis. Clamped at 0 so an origin sitting
            // exactly on a cell boundary cannot step backwards.
            t_max[a] = ((cur[a] as f64 + 0.5 * step[a] as f64) - origin[a]) / dir[a];
            if t_max[a] < 0.0 {
                t_max[a] = 0.0;
            }
        }
    }

    let mut inside = true; // origin cell assumed occupied (see semantics)
    let mut hits = 0usize;
    loop {
        let a = argmin3(&t_max);
        let t = t_max[a];
        // `!(t <= max_dist)` rather than `t > max_dist`, so an all-infinite
        // t_max (a direction that never crosses anything) terminates instead of
        // looping forever.
        if !(t <= max_dist) {
            break;
        }
        // i64 first: a ray is expected to walk off the end of the object.
        let next = cur[a] as i64 + step[a] as i64;
        if next < i32::MIN as i64 || next > i32::MAX as i64 {
            break;
        }
        cur[a] = next as i32;
        t_max[a] += t_delta[a];

        let occupied = index.get(key_of(cur[0], cur[1], cur[2])).is_some();
        if occupied != inside {
            out[hits] = t;
            hits += 1;
            inside = occupied;
            if hits == out.len() {
                break;
            }
        }
    }
    hits as i32
}
```

Notes:

- **No allocation in the hot loop.** The entire per-ray state is 13 scalars that
  live in registers; `out` is a slice of the caller's output buffer. This is the
  whole reason it wins — the numpy version allocates several `(R, 3)` temporaries
  per *step*.
- `key_of` and `SpatialIndex::get` are unchanged (`src/index.rs:20`, `:102`). This
  needs no new data structure and touches no existing one.
- `argmin3` written out by hand rather than via an iterator: it is the innermost
  thing in the library after `get`.
- **Expect to be probe-bound.** At 22 ns per `SpatialIndex::get` on a 5.5 M-entry
  `FxHashMap`, the DDA arithmetic is almost free by comparison and the loop is
  memory-latency-bound. Do not micro-optimise the arithmetic before measuring.
- **`py.allow_threads` around the whole batch**, as every other entry point does
  (`lib.rs:283, :314, :343, :377, :404, :437, :472, ...`).
- **This is the natural first `rayon` target.** Rays are completely independent:
  no shared mutable state, no ordering, a fixed output slot each, and a perfectly
  balanced `par_chunks_mut(max_crossings)` over `out` zipped with the inputs.
  `handoff.md:147` already names per-component Dijkstra as the intended first use
  of rayon and notes it was never added; this is strictly simpler (Dijkstra's
  parallelism is ragged across components, this one is per-row). Being
  latency-bound, it should scale close to linearly. **Land it single-threaded
  first**, as `handoff.md` says.

### Binding

Follows the five-file pattern the last two primitives used, purely additively:

| # | file | what |
|---|---|---|
| 1 | `src/raycast.rs` | the loop above + `#[cfg(test)] mod tests` |
| 2 | `src/lib.rs` | `mod raycast;`, the `#[pyfunction]`, the `#[pymodule]` registration, a `Graph` method |
| 3 | `python/dijkstra3d_sparse/__init__.py` | `__all__` entry, wrapper, numpydoc |
| 4 | `README.md` | signature in the API block + a prose section |
| 5 | `tests/test_raycast.py` | ~15 tests against an independent oracle |

`src/lib.rs` needs one new type alias and one new borrow helper alongside the
existing `coords_slice` (`lib.rs:52`):

```rust
/// `(t, n_hits)` output of `ray_exits`; `t` is flattened, reshaped in Python.
type RayExitArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i32>>);

/// Extract `(slice, r)` from an `(R, 3)` C-contiguous f64 array.
fn rays_slice<'a>(name: &str, a: &'a PyReadonlyArray2<'a, f64>) -> PyResult<(&'a [f64], usize)>;
```

and the function itself, positional per the house convention (defaults live in
Python, `lib.rs:363`):

```rust
#[pyfunction]
#[pyo3(signature = (voxels, origins, directions, max_dist, max_crossings, index_kind))]
fn ray_exits<'py>(
    py: Python<'py>,
    voxels: PyReadonlyArray2<'py, i32>,
    origins: PyReadonlyArray2<'py, f64>,
    directions: PyReadonlyArray2<'py, f64>,
    max_dist: PyReadonlyArray1<'py, f64>,
    max_crossings: usize,
    index_kind: &str,
) -> PyResult<RayExitArrays<'py>> { ... }
```

Borrows must be bound **outside** the `allow_threads` closure so they outlive it —
the existing explicit comment at `lib.rs:310` is about exactly this.

The Python wrapper mirrors `factorize` (`__init__.py:840`), reusing `_as_voxels`
and adding a `_as_rays` coercion in the same style, and reshaping the flat `t`
with `.reshape(-1, max_crossings)` — the same trick `label_adjacency` uses at
`__init__.py:397`.

**And please update `python/dijkstra3d_sparse/_native.pyi` in the same pass.** It
is currently stale: it never gained `exposed_faces`, `factorize` or the `Graph`
class from the last two primitives, even though `py.typed` is shipped. Adding a
third omission would make it actively misleading rather than merely incomplete.

### Tests to add

Against independent oracles, per the house standard (`tests/helpers.py` builds an
explicit CSR graph rather than reusing the library's own machinery; do the same
here).

- **Analytic**: rays from the centre of a rasterized ball of radius `R`, along the
  six axes — the exit is exactly `R + 0.5`, no tolerance needed. Along a body
  diagonal it is the outermost occupied cell's far face, computable in the test.
- **Brute force**: `sparse-cubes`' `tests/_tube_oracle.py` is exactly the oracle
  wanted — fine fixed-step sampling against a Python `set`, sharing no arithmetic,
  no packing and no data structure with the DDA. Port it (~30 lines) and assert
  the DDA never *under*shoots it and never overshoots by more than one sampling
  step. This is a one-sided comparison by construction; assert it that way.
- **Every cell is visited**: on a solid block, the cells stepped through between
  entry and exit must equal the supercover of the segment. This is what separates
  a real DDA from a sampler, and nothing else in the suite would catch a
  regression to sampling.
- **Alternation**: two parallel slabs with a gap, `max_crossings=4` — exit /
  entry / exit in strictly increasing `t`, `n_hits` matching, `+inf` padding
  beyond it.
- **`max_dist`**: a ray down an infinite corridor returns `n_hits = 0` and
  terminates; per-ray `max_dist` clips different rays at different points in one
  batch.
- **Backend parity**: `index_kind="sorted"` and `"hash"` must be **bit-identical**
  (`assert_array_equal`, not `allclose`) — they differ only in how a coordinate is
  looked up. `tests/test_graph.py` is the model - its module docstring already states the
  rule, and every parity test there follows it.
- **Degenerate**: zero direction component, all-zero direction, origin in an empty
  cell, `R == 0`, `N == 0`, `max_crossings = 1` vs large, non-finite input
  rejected.
- **i32 extremes**: a ray aimed off the end of the coordinate range must stop, not
  wrap. `components.rs:458` and `dedup.rs:324` already have this shape of trap.
- **Rust-side**: `#[cfg(test)] mod tests` in `src/raycast.rs` with a brute-force
  `reference_exit` and the `scattered_cloud` generator at `components.rs:425`
  already uses (`reference_mask` at `:399` is the shape to copy).

---

## 3. What the caller becomes

`sparsecubes/tube.py::_raycast` — ~60 lines of lockstep bookkeeping and
compaction — collapses to one call, and the node-chunking that exists purely to
bound the ray block goes with it:

```python
# before: chunk over nodes so (chunk * n_theta) rays fit in ~70 MB, then march
#         them in lockstep, compacting the survivors every step.
# after:
t, n_hits = graph.ray_exits(
    origins, d_idx, max_dist=caps, max_crossings=2 if diagnostics else 1,
)
radius    = np.where(n_hits > 0, t[:, 0], caps)
escaped   = n_hits == 0
reentered = n_hits > 1
```

The `star_window` policy — a re-entry only counts as a star-shapedness violation
within `(1 + w) * t_exit` of the exit — stays caller-side, applied to `t[:, 1]`
after the fact. It is a statement about what a crossing *means*, and baking it in
would make the primitive answer a neuroscience question. (It matters: widening
that window takes the reported non-star rate from 31% to 69% on the neuron, which
is why it is a caller policy and not a default.)

The pure-numpy `_KeyMembership` path stays as the fallback and as the parity
oracle (`tests/test_tube.py::test_backends_agree`), exactly as `_ScipyGraph` does
for `wavefront`.

---

## 4. Measured payoff

**The projection held.** On a 1 M-ray block from the neuron (mean cap 219 nm,
6.9% of rays escaping, ~9 cells traversed per ray):

| | wall | ns/ray |
|---|---|---|
| numpy DDA + `Graph.index_of` | 0.787 s | 787 |
| **`Graph.ray_exits`, `max_crossings=1`** | **0.152 s** | **152** |
| `Graph.ray_exits`, `max_crossings=2` | 0.280 s | 280 |

That is **~24 ns a probe against the 17-22 ns bulk-`index_of` floor measured on
the same index** — i.e. the scalar walk is running at memory latency with the DDA
arithmetic essentially free, which is exactly what §1 argued and is as good as
this can get single-threaded. The Python wrapper costs 1 ms of the 152, so
`_ray_args` validation is not worth optimising.

End to end, `tube_profile` on the full 73,671-node TEASAR skeleton (4.71 M rays):

| | diagnostics=False | diagnostics=True |
|---|---|---|
| `backend="keys"` | 9.42 s | 14.47 s |
| `backend="sparse"` | 3.48 s | 5.26 s |
| **`backend="exits"`** | **0.93 s** | **1.42 s** |

**3.7x over the previous default, 10.1x over pure numpy**, and the same at
`diagnostics=True` — and the raycast is now ~0.65 s of a 0.93 s call, so it has
stopped being the thing worth optimising.

### The dedup, which this exposed

Getting there needed one fix on the caller's side, recorded here because it is the
kind of thing only a faster primitive reveals. `tube_profile` opened with
`unique(voxels, axis=0)`, since `Graph` rejects duplicate coordinates. On the
5.56 M-voxel arbor that call cost **4.30 s and found zero duplicates**, against
0.07 s to build the index itself — so once the walk dropped to 0.65 s, the dedup
was four times the raycast. It is now a `try: Graph(...) except ValueError:`
fallback, so only a caller that actually has duplicates pays for the discovery.

Nothing is being asked of `dijkstra3d_sparse` here — rejecting duplicates is the
right contract, and `Graph` reports the offending row pair, which is what made the
fallback easy to write.

### Still open

- **rayon.** Untouched, and still the natural first target: rays are
  embarrassingly parallel and the walk is now provably latency-bound, so
  thread-scaling should be close to linear.
- **`max_crossings=2` costs 1.8x `max_crossings=1`**, not 2x. That is what makes
  the single-call form in §3 the right one: asking for two crossings and applying
  `star_window` to `t[:, 1]` afterwards beat a two-pass version (one `mc=1` call,
  then a tighter-capped `mc=2` over the survivors) by ~20% — 1.42 s against
  1.85 s — because the second pass re-walks the first segment. The §3 sketch was
  right and the two-pass implementation it replaced was not.

Memory: the `(R, 3)` ray-state block disappears (one ray at a time in Rust), so
`_RAY_BLOCK` chunking now bounds only the input/output arrays, not the working
set. Peak RSS on the neuron is unchanged, because skeletonization dominates it.

---

## 5. Non-goals

- **No frames, no FFT, no LOD, no star-shapedness verdict.** Rotation-minimizing
  frames, the rFFT, the storage format and the "is this cross-section
  star-shaped?" policy all stay in `sparsecubes/tube.py`. The primitive answers
  exactly one geometric question.
- **No `anisotropy` parameter.** The caller pre-divides `directions` by the
  spacing, so `t` is already in the caller's units (§2).
- **No `stop_mask` / cross-ray early termination.** Rays are independent; there is
  no global state to terminate on.
- **Not a general ray query.** No triangles, no SDF, no sub-voxel occupancy, no
  interpolation. "First boundary crossing of a sparse binary voxel set", nothing
  wider.
- **Not `euclidean_distance_field`.** `handoff.md:52-54` reserves that name for a
  separate deferred primitive; this is unrelated and does not fill that slot.
- **No amortisation across rays.** No shared frustum, no coherent packet tracing.
  Those are worth real speedups in offline renderers and would complicate the
  contract for a workload that is already going to be latency-bound.

---

## 6. Cross-cutting note

A second potential caller exists, but needs a *different* primitive rather than
this one: `sparsecubes/voxelization.py` fills mesh interiors by casting a ray down
each column and applying the even-odd rule. Same shape of question, but against
triangles rather than an occupied voxel set. Noted only so that the naming
(`ray_exits`, not `raycast`) leaves room for it.

Closer to home, `dijkstra3d_sparse` would after this have three primitives that
are all "one pass over rows, probing the index, no graph" — `exposed_faces`,
`index_of` and `ray_exits`. If a fourth appears it may be worth a shared
`probe`-style module rather than three near-identical borrow/validate/allow_threads
preambles in `lib.rs`. Not yet.
