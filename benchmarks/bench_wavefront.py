#!/usr/bin/env python
"""Benchmark the prototype wavefront skeletonizer against the existing backends.

``wavefront_skeletonize`` collapses geodesic level sets ("rings") of the voxel
graph to their centroids - the Reeb-graph construction behind `skeletor`'s
``by_wavefront``, ported onto sparse voxels. This script times it against
``thin_skeletonize`` (topological thinning) and ``teasar_skeletonize`` on the
same inputs and scores the skeletons on the same metrics.

Metrics
-------
cc / b1     Skeleton components and first Betti number (loops). ``cc`` should
            match the object's component count; a mismatch means the skeleton
            fragmented. ``b1 > 0`` means cycles survived.
cover95     95th percentile of the distance from an object voxel to the nearest
            skeleton node, in voxels. How well the skeleton "explains" the
            object - lower is better, and it should sit near the local radius.
centre      Mean distance from a skeleton node to the nearest background voxel.
            This *is* the local radius at a perfectly centred node, so higher is
            better: it says the nodes sit on the medial axis rather than
            hugging the wall.
rad/centre  Mean reported radius over that mean true radius. 1.0 = unbiased *on
            average* - but over- and under-estimates cancel here, so read it
            together with radRMSE.
radRMSE     Root-mean-square per-node error of the reported radius against the
            same reference. This is the column that separates radius estimators:
            a shape-dependent aggregation can look unbiased above while being
            wrong node by node.
            CAVEAT: `thin` and `teasar` define their radii *as* the distance to
            the nearest background voxel, which is the reference used here, so
            their 0.00 is a tautology and not a measure of quality. The column is
            only meaningful for comparing the `wave-r*` estimators to each other.

Usage
-----
    python benchmarks/bench_wavefront.py              # default sweep
    python benchmarks/bench_wavefront.py --quick      # small shapes only
    python benchmarks/bench_wavefront.py --neuron     # add the real fixtures
    python benchmarks/bench_wavefront.py --methods wave,teasar-exact
"""

import argparse
import os
import sys
import time

import numpy as np

# `_shapes` and the neuron fixtures live under tests/.
_TESTS = os.path.join(os.path.dirname(__file__), os.pardir, "tests")
sys.path.insert(0, _TESTS)

import sparsecubes as sc
from sparsecubes.core import boundary_shell, unique
from sparsecubes.wavefront import wavefront_skeletonize
from _shapes import betti, n_components, self_touch_hairpin, solid_cube, solid_cylinder


# Each entry: label -> callable(voxels) -> Skeleton. Radii are requested
# everywhere so the radius column is comparable.
METHODS = {
    "thin": lambda v: sc.thin_skeletonize(v, radii=True),
    "teasar-exact": lambda v: sc.teasar_skeletonize(v, branching="exact"),
    "teasar-fast": lambda v: sc.teasar_skeletonize(v, branching="fast"),
    "wave": lambda v: wavefront_skeletonize(v),
    "wave-step3": lambda v: wavefront_skeletonize(v, step_size=3.0),
    "wave-surface": lambda v: wavefront_skeletonize(v, surface=True),
    "wave-raw": lambda v: wavefront_skeletonize(v, tree=False),
    "wave-rmean": lambda v: wavefront_skeletonize(v, radius_agg="mean"),
    "wave-rmax": lambda v: wavefront_skeletonize(v, radius_agg="max"),
}
DEFAULT_METHODS = ["thin", "teasar-exact", "teasar-fast", "wave", "wave-step3", "wave-surface", "wave-raw"]


def _score(sk, voxels, shell_tree):
    """Topology + geometry metrics for one skeleton (see the module docstring)."""
    if len(sk.nodes) == 0:
        return dict(cc=0, b1=0, cover95=float("nan"), centre=float("nan"), bias=float("nan"))
    cc, b1 = betti(sk.nodes, sk.edges)
    verts = sk.vertices

    # How far an object voxel is from the skeleton, and how deep in the object
    # the skeleton nodes sit. Both are KD-tree queries on point sets we already
    # have - nothing here allocates a dense volume.
    d_obj, _ = _tree(verts).query(voxels.astype(float), k=1)
    d_shell, _ = shell_tree.query(verts, k=1)

    bias = rmse = float("nan")
    if sk.radii is not None and len(sk.radii) and d_shell.mean() > 0:
        bias = float(np.mean(sk.radii) / d_shell.mean())
        rmse = float(np.sqrt(np.mean((sk.radii - d_shell) ** 2)))
    return dict(
        cc=cc,
        b1=b1,
        cover95=float(np.percentile(d_obj, 95)),
        centre=float(d_shell.mean()),
        bias=bias,
        rmse=rmse,
    )


def _tree(points):
    from scipy.spatial import cKDTree

    return cKDTree(points)


def _time(fn, voxels, repeats):
    """Best-of-`repeats` wall time (seconds) plus one sample skeleton."""
    sk = fn(voxels)  # warmup (and the skeleton we score)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(voxels)
        best = min(best, time.perf_counter() - t0)
    return best, sk


def _shapes(quick, neuron):
    """(label, voxels) sweep, ordered by voxel count."""
    shapes = [
        ("cylinder r4 L20", solid_cylinder(4, 20)),
        ("hairpin", self_touch_hairpin()),
    ]
    if not quick:
        shapes += [
            ("cylinder r8 L60", solid_cylinder(8, 60)),
            ("cube 20", solid_cube(20)),
        ]
    if neuron:
        for name, div in [("10075_coarse.npy", 1), ("10075_scale3.npy", 8), ("10075_scale3.npy", 4)]:
            path = os.path.join(_TESTS, name)
            if not os.path.exists(path):
                print(f"(fixture not found: {path}; skipping)")
                continue
            v = unique((np.load(path) // div).astype(np.int64), axis=0)
            shapes.append((f"{name.split('_')[0]} //{div} ({len(v)})", v))
    return shapes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3, help="timed runs per cell (best-of)")
    ap.add_argument("--quick", action="store_true", help="small shapes only")
    ap.add_argument("--neuron", action="store_true", help="include the real neuron fixtures")
    ap.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=f"comma-separated subset of: {', '.join(METHODS)}",
    )
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        ap.error(f"unknown method(s): {unknown}. Choose from {list(METHODS)}")

    header = (
        f"{'method':<13} {'t(s)':>8} {'nodes':>7} {'edges':>7} {'cc':>3} {'b1':>4} "
        f"{'cover95':>8} {'centre':>7} {'rad/centre':>10} {'radRMSE':>8}"
    )
    for label, vox in _shapes(args.quick, args.neuron):
        n_obj = n_components(vox)
        print(f"\n=== {label}   {len(vox)} voxels, {n_obj} component(s)")
        print(header)
        print("-" * len(header))
        shell_tree = _tree(boundary_shell(vox).astype(float))
        for name in methods:
            try:
                t, sk = _time(METHODS[name], vox, args.repeats)
            except Exception as exc:  # a backend that cannot handle this shape
                print(f"{name:<13} FAILED: {type(exc).__name__}: {exc}")
                continue
            s = _score(sk, vox, shell_tree)
            flag = "" if s["cc"] == n_obj else f"  <- cc != {n_obj}"
            print(
                f"{name:<13} {t:>8.3f} {len(sk.nodes):>7} {len(sk.edges):>7} "
                f"{s['cc']:>3} {s['b1']:>4} {s['cover95']:>8.2f} {s['centre']:>7.2f} "
                f"{s['bias']:>10.2f} {s['rmse']:>8.2f}{flag}"
            )


if __name__ == "__main__":
    main()
