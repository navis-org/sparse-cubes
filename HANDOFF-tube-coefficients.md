# Tube coefficients: what the extraction measured

> **STATUS: implemented and shipped, 2026-08-04.** `sparsecubes/tube.py`
> (`sc.measure.tube_profile`, `sc.tube_coefficients`, `sc.TubeProfile`) extracts a
> truncated Fourier cross-section profile per skeleton node. This document is the
> **findings record** — the characterization experiment the design was contingent
> on, run on real data, including the two places the data contradicted what was
> assumed going in. Nothing here is outstanding work.
>
> Two things were spun out of it into their own documents, so that neither is a
> second copy of the other:
>
> - **`HANDOFF-ray-exits.md`** — the `dijkstra3d_sparse` primitive that would
>   replace the numpy raycaster. *Proposed, not blocking.*
> - **`HANDOFF-octarine-tube-shader.md`** — the pygfx vertex-pulling shader that
>   renders the result. *Proposed, not blocking.*
>
> Measurements below are from `benchmarks/bench_tube.py --neuron` on
> `tests/10075_scale3.npy` (5,560,148 voxels → 30,226 nodes / 30,188 edges,
> `spacing=(16, 16, 16)`, `n_theta=64`, `diagnostics=True`).

---

## 1. Cost

```
                         t_skel   t_tube  us/ray  probes  steps/ray  peak RSS
wavefront_skeletonize     10.92        -       -       -          -   ~1.4 GB
tube_profile (sparse)         -     6.64    3.43   21.3M      11.01   1.53 GB
tube_profile (keys)           -    11.93    6.17   21.3M      11.01   1.52 GB
```

Yardsticks the rest of the library is held to on this fixture: one `thin()` pass
is ~17 s, `wavefront_skeletonize` end to end is 10.6 s / 1.67 GB. So the raycast
lands inside budget on both axes and is the *cheaper* half of the pipeline. Peak
RSS is dominated by the skeletonization that precedes it — the raycaster's own
live set is capped at `_RAY_BLOCK = 1e6` rays (~70 MB) by chunking over nodes,
independent of node count.

At `K = 4` the stored representation is **16 floats = 64 bytes a node**, so the
whole arbor is **1.93 MB**.

---

## 2. The spectrum: a `k=2` spike, then no knee

Mean `m_k` over interior non-junction nodes, physical units (mean `a0` = 67.1 nm
≈ 4.2 voxels):

```
   a0      m1      m2      m3      m4      m5      m6      m8     m10
67.08    7.64   16.69    7.45    6.23    4.50    3.71    2.62    2.02
```

`m2` is 2.2x its neighbours — the predicted ellipticity signature, and the one
harmonic that clearly earns its bytes. After it the decay is smooth and roughly
`1/k`.

**There is no knee, so `K` is a budget decision rather than a natural cut.**
Residual as a fraction of `a0` if truncated at `k`:

```
 k=1     k=2     k=3     k=4     k=5     k=6     k=8    k=10
0.264   0.243   0.156   0.129   0.108   0.094   0.076   0.064
```

At the shipped `K = 4` that is 12.9% of the mean radius = 8.7 nm ≈ **0.54 voxels**
— already at the rasterization floor. (A synthetic disc of radius 8 voxels floors
at 4.3%; the neuron's median radius is about half that, so its floor is
proportionally higher.) Past `k ≈ 6` the extra harmonics are mostly buying
quantization noise.

**Consequence for the LOD design.** The angular resolution that actually matters
is low, which matches the prediction that the two LOD axes are not symmetric:
boutons and varicosities are a local bump in `a0` *along the axis*, not an angular
phenomenon, so **axial resampling dominates the error budget** and angular
truncation matters mainly for large-calibre proximal dendrites.

---

## 3. Star-shapedness: widespread but shallow — and the metric needed fixing

`r(θ)` can only represent a cross-section that is star-shaped about its skeleton
point. This is the hard ceiling on the whole approach, so it was measured
directly.

**63.5% of interior, non-junction nodes have at least one ray that re-enters — but
only 4.4% of rays do.**

Two things worth separating in that sentence. The original design note guessed
violations would be dominated by branch-point neighbourhoods; **they are not** —
junction-adjacent nodes are *excluded* from that 63.5%, so the violations are
spread along shafts. That is the pessimistic reading, and it is real. But the
per-ray rate says the shafts are still overwhelmingly star-shaped: the failure is
a few directions per cross-section, not a wholesale breakdown. The approach
survives with a bounded, localised error rather than a fatal one.

**The number only became meaningful after separating two windows that started as
one.** Letting a ray hunt for a re-entry all the way out to its *escape* cap
conflates "this cross-section folds back on itself" with "another neurite passes
nearby", and the reported rate then simply tracks how far the ray was allowed to
keep looking:

```
re-entry window   1.25x   1.5x   2.0x   3.0x   4.0x radius
non-star nodes    0.315   0.394  0.521  0.676  0.692
```

A statistic that moves by a factor of two with a parameter that exists for an
unrelated reason is not a measurement. Hence `star_window` (default 1.0) as a
parameter distinct from `max_radius`, and `TubeProfile.non_star` reporting the
*fraction* of a node's rays rather than a bare flag — one grazing ray out of 64
and half the ring folding back are very different situations.

---

## 4. Anisotropy is worse than a flat noise floor

The control sweep (`bench_tube.py`, the `tilt`/`aniso` rows) takes **one** voxel
cylinder at 4×4×40 and reads it with its axis along z versus in-plane:

```
tangent along z    a0=31.46   m1=0.00   m2= 0.00   m4= 0.41
tangent in-plane   a0=41.53   m1=3.30   m2=32.86   m4=17.51
```

Identical geometry; `m2` goes from 0 to 33. Both readings match the continuum
polar-radius expansion of the ellipse the anisotropy makes of the disc, so the
scaling is exact rather than merely large.

**The noise floor on `m_k` is a strong function of the neurite's direction**, and
any truncation threshold has to account for the local tangent —
`TubeProfile.frame_vectors()` returns it for that reason. It is easy to mistake
this artifact for real structure.

On *isotropic* data the effect is absent: the neuron fixture at 16×16×16 shows
`m2` of 15.5 / 17.4 / 17.3 across the three tangent bins. So this is specifically
an anisotropic-acquisition problem, not an intrinsic property of the method.

This is also why the proposed `ray_exits` primitive takes `directions` in **index
space** rather than owning an `anisotropy` parameter: the caller keeps control of
the metric the quantization is measured in.

---

## 5. Skeleton centring shows up in `m_1`, as designed

`m_1` is the offset between the skeleton node and the cross-section's centroid, so
a non-zero `m_1` is a diagnostic that the skeleton is not centred — not a shape.
It works as intended, and the two skeletonizers differ measurably
(neuron //4, interior non-junction nodes):

```
             nodes   m1/a0   m2/a0   resid/a0   non-star
wavefront     5431   0.109   0.202      0.103      0.520
teasar        7850   0.325   0.244      0.137      0.750
```

`wavefront_skeletonize`'s ring centroids are better centred here than TEASAR's
medial-axis paths, which is worth knowing given TEASAR is `tube_coefficients`'
default (chosen for its tree topology and per-node radius). Not acted on; recorded
because `m_1` is the only cheap way to see it.

---

## 6. Reproduce

```bash
python benchmarks/bench_tube.py --neuron --backends sparse,keys
python benchmarks/bench_tube.py --quick          # synthetic oracles only
```

The benchmark reports all of the above — timings and peak RSS per shape, the
spectrum, the per-`k` residual budget, the failure-mode rates, and `m_k` binned by
tangent direction — because they come out of the same runs and separating them
into their own script would only let them drift apart.
