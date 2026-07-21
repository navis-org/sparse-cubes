#!/usr/bin/env python
"""Where does sparse Gaussian smoothing beat densifying the bounding box?

`sc.filters.smooth` and `scipy.ndimage.gaussian_filter` compute the same thing
(the test suite pins them to floating-point round-off), so the only question
worth asking is which one to reach for. Unlike the rest of `sparse-cubes` the
answer is genuinely "it depends": smoothing *grows* the voxel set, by the kernel
radius along every axis, so the sparsity that makes the library fast is partly
spent by the operation itself. Past some occupancy the dense grid wins outright.

This script measures that crossover instead of guessing at it, sweeping

* **occupancy** - what fraction of the bounding box is filled, and
* **sigma** - which sets the radius (``r = int(truncate * sigma + 0.5)``) and so
  how far the support grows,

for two cloud shapes with very different growth behaviour:

* **uniform** - voxels scattered at random. The worst case: every voxel grows its
  own ``(2r + 1) ** 3`` halo and almost none of them overlap.
* **clustered** - random-walk tubes, i.e. neurite-like. Growth is a surface
  effect, so the support grows far more slowly. This is what real data looks
  like, and the reason the honest answer is better than the uniform one.

It also reports `epsilon`, the pruning threshold that discards the Gaussian's
negligible tail after each pass - the lever that moves the crossover back in
sparse's favour at a bounded cost in exactness.

Usage
-----
    python benchmarks/bench_smooth.py
    python benchmarks/bench_smooth.py --quick
    python benchmarks/bench_smooth.py --neuron      # add the real fixture
"""

import argparse
import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sparsecubes as sc  # noqa: E402

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover
    sys.exit("This benchmark needs scipy: pip install scipy")


# --- cloud generators -------------------------------------------------------


def uniform_cloud(side, occupancy, seed=0):
    """`occupancy` of a `side`-cubed box, filled at random. Worst case for sparse."""
    rng = np.random.RandomState(seed)
    n = max(1, int(side**3 * occupancy))
    return np.unique(rng.randint(0, side, (n, 3)).astype(np.int64), axis=0)


def clustered_cloud(side, occupancy, seed=0):
    """Neurite-like random-walk tubes, grown until `occupancy` is reached.

    Compact structure, so the smoothing halo is a surface effect rather than a
    per-voxel cost - the regime real image data actually lives in.
    """
    rng = np.random.RandomState(seed)
    target = max(1, int(side**3 * occupancy))
    out, total = [], 0
    while total < target:
        pos = rng.randint(side // 4, 3 * side // 4, 3).astype(np.int64)
        steps = rng.randint(-1, 2, (max(64, target // 8), 3))
        walk = np.clip(pos + np.cumsum(steps, axis=0), 0, side - 1)
        # Thicken the walk into a tube so the cloud is solid, not a wire.
        tube = (walk[:, None, :] + _BALL[None, :, :]).reshape(-1, 3)
        tube = np.clip(tube, 0, side - 1)
        out.append(tube)
        total = len(np.unique(np.concatenate(out), axis=0))
    return np.unique(np.concatenate(out), axis=0)


_BALL = np.array(
    [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)],
    dtype=np.int64,
)


# --- measurement ------------------------------------------------------------


def timed(fn):
    """Run `fn`, returning (result, seconds, python-side peak bytes)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return out, dt, peak


def dense_smooth(voxels, sigma, truncate=4.0):
    """The densifying reference: allocate the padded box and let scipy filter it."""
    r = int(truncate * sigma + 0.5)
    pad = r + 1
    lo = voxels.min(axis=0) - pad
    shape = tuple(voxels.max(axis=0) - lo + pad + 1)
    grid = np.zeros(shape, dtype=float)
    grid[tuple((voxels - lo).T)] = 1.0
    return ndimage.gaussian_filter(grid, sigma, mode="constant", truncate=truncate)


def dense_bytes(voxels, sigma, truncate=4.0):
    """Bytes the dense path must allocate for its float64 grid - a hard floor."""
    r = int(truncate * sigma + 0.5)
    extent = voxels.max(axis=0) - voxels.min(axis=0) + 2 * (r + 1) + 1
    return int(np.prod(extent.astype(float)) * 8)


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


# --- the sweep --------------------------------------------------------------


def sweep(side, occupancies, sigmas, generator, name, dense_cap_gb=2.0):
    print(f"\n{'=' * 78}")
    print(f"{name}  (box {side}^3 = {side ** 3:,} cells)")
    print("=" * 78)
    print(
        f"{'occup.':>8} {'voxels':>9} {'sigma':>6} {'out vox':>10} "
        f"{'sparse':>9} {'dense':>9} {'speedup':>8} {'sparse mem':>11} {'dense mem':>10}"
    )
    print("-" * 78)

    for occ in occupancies:
        vox = generator(side, occ)
        real_occ = len(vox) / side**3
        for sigma in sigmas:
            need = dense_bytes(vox, sigma)
            if need > dense_cap_gb * 1024**3:
                print(
                    f"{real_occ:>8.4%} {len(vox):>9,} {sigma:>6.1f} "
                    f"{'-':>10} {'-':>9} {'skipped':>9} {'-':>8} {'-':>11} "
                    f"{fmt_bytes(need):>10}"
                )
                continue

            (sv, _), t_sparse, m_sparse = timed(
                lambda: sc.filters.smooth(vox, sigma=sigma)
            )
            _, t_dense, _ = timed(lambda: dense_smooth(vox, sigma))

            ratio = t_dense / t_sparse
            mark = "" if ratio >= 1 else "  <- dense wins"
            print(
                f"{real_occ:>8.4%} {len(vox):>9,} {sigma:>6.1f} {len(sv):>10,} "
                f"{t_sparse:>8.3f}s {t_dense:>8.3f}s {ratio:>7.2f}x "
                f"{fmt_bytes(m_sparse):>11} {fmt_bytes(need):>10}{mark}"
            )


def epsilon_sweep(side, occupancy, sigma, generator, name):
    """What pruning the Gaussian's tail buys, and what it costs in accuracy."""
    print(f"\n{'=' * 78}")
    print(f"epsilon: pruning the tail  ({name}, occupancy {occupancy:.2%}, sigma {sigma})")
    print("=" * 78)

    vox = generator(side, occupancy)
    (base_v, base_val), t0, _ = timed(lambda: sc.filters.smooth(vox, sigma=sigma))
    peak = float(base_val.max())
    base = dict(zip(map(tuple, base_v.tolist()), base_val))
    print(f"{'epsilon':>12} {'out vox':>10} {'vs exact':>9} {'time':>9} {'speedup':>8} {'max err':>10}")
    print("-" * 78)
    print(f"{'0 (exact)':>12} {len(base_v):>10,} {'100.0%':>9} {t0:>8.3f}s {'1.00x':>8} {'-':>10}")

    for rel in (1e-8, 1e-6, 1e-4, 1e-2):
        eps = peak * rel
        (v, val), t, _ = timed(
            lambda: sc.filters.smooth(vox, sigma=sigma, epsilon=eps)
        )
        got = dict(zip(map(tuple, v.tolist()), val))
        err = max(abs(got.get(k, 0.0) - base.get(k, 0.0)) for k in set(base) | set(got))
        print(
            f"{rel:>10.0e}*pk {len(v):>10,} {len(v) / len(base_v):>8.1%} "
            f"{t:>8.3f}s {t0 / t:>7.2f}x {err / peak:>9.2e}*pk"
        )


def neuron_case(sigmas):
    """The case the library exists for: a real neuron, far too big to densify."""
    path = os.path.join(os.path.dirname(__file__), "..", "tests", "10075_coarse.npy")
    if not os.path.exists(path):
        print("\n(neuron fixture not found, skipping)")
        return
    vox = np.load(path).astype(np.int64)
    extent = vox.max(axis=0) - vox.min(axis=0) + 1
    occ = len(vox) / np.prod(extent.astype(float))

    print(f"\n{'=' * 78}")
    print(f"real neuron  ({len(vox):,} voxels, extent {tuple(extent)}, occupancy {occ:.4%})")
    print("=" * 78)
    print(f"{'sigma':>6} {'out vox':>12} {'growth':>8} {'sparse':>9} {'dense':>9} {'speedup':>8} {'dense mem':>10}")
    print("-" * 78)
    for sigma in sigmas:
        (sv, _), t_sparse, _ = timed(lambda: sc.filters.smooth(vox, sigma=sigma))
        need = dense_bytes(vox, sigma)
        if need <= 2 * 1024**3:
            _, t_dense, _ = timed(lambda: dense_smooth(vox, sigma))
            dense_col, speed = f"{t_dense:>8.3f}s", f"{t_dense / t_sparse:>7.2f}x"
        else:
            dense_col, speed = f"{'skipped':>9}", f"{'-':>8}"
        print(
            f"{sigma:>6.1f} {len(sv):>12,} {len(sv) / len(vox):>7.1f}x "
            f"{t_sparse:>8.3f}s {dense_col} {speed} {fmt_bytes(need):>10}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer points")
    ap.add_argument("--neuron", action="store_true", help="add the real fixture")
    args = ap.parse_args()

    side = 48 if args.quick else 64
    occupancies = (
        [0.001, 0.02, 0.2] if args.quick else [0.0005, 0.002, 0.01, 0.05, 0.2, 0.5]
    )
    sigmas = [1.0, 2.0] if args.quick else [0.5, 1.0, 2.0, 4.0]

    sweep(side, occupancies, sigmas, uniform_cloud, "uniform (worst case for sparse)")
    sweep(side, occupancies, sigmas, clustered_cloud, "clustered / neurite-like (realistic)")
    epsilon_sweep(side, 0.02, 2.0, clustered_cloud, "clustered")

    if args.neuron:
        neuron_case(sigmas)

    print(
        "\nRule of thumb: sparse wins while the *grown* support stays well under the\n"
        "bounding box. Growth is what matters, not the input's own occupancy - which\n"
        "is why the clustered numbers are so much better than the uniform ones, and\n"
        "why `epsilon` (which caps the growth) moves the crossover so effectively."
    )


if __name__ == "__main__":
    main()
