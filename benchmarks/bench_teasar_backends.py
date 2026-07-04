#!/usr/bin/env python
"""Benchmark the Dijkstra backends behind ``teasar_skeletonize``.

``teasar_skeletonize`` runs its shortest-path passes through ``dijkstra3d_sparse``
(straight over the voxel coordinates, no CSR graph) when it is installed, and
falls back to ``scipy.sparse.csgraph.dijkstra`` otherwise. This script times the
two end-to-end across a sweep of shapes, sizes and branching modes, and checks
they agree on skeleton topology.

Only the Dijkstra passes differ between backends - the shared DBF /
connected-components / invalidation work is identical - so these are honest
end-to-end numbers (the isolated Dijkstra speedup is larger). The accelerator's
big win is ``branching="exact"`` on large objects: it grafts each path onto the
skeleton with an early-terminating ``shortest_path_to_set`` (the search shrinks
as the tree fills) instead of a full root Dijkstra per path. The ``tree`` / ``fast``
modes need the full field, so there the two backends are near par.

Usage
-----
    python benchmarks/bench_teasar_backends.py             # default sweep
    python benchmarks/bench_teasar_backends.py --quick     # smaller / faster
    python benchmarks/bench_teasar_backends.py --repeats 5
    python benchmarks/bench_teasar_backends.py --neuron    # add the real fixture
"""

import argparse
import os
import sys
import time

import numpy as np

# `_shapes` and the neuron fixture live under tests/.
_TESTS = os.path.join(os.path.dirname(__file__), os.pardir, "tests")
sys.path.insert(0, _TESTS)

import sparsecubes as sc
from sparsecubes import teasar
from sparsecubes.core import unique
from _shapes import solid_cube, solid_cylinder, betti


def _signature(sk):
    """Topology fingerprint: (nodes, edges, components, first_betti)."""
    ncomp, first_betti = betti(sk.nodes, sk.edges)
    return len(sk.nodes), len(sk.edges), ncomp, first_betti


def _time(voxels, use_lib, repeats, **kw):
    """Best-of-`repeats` wall time (seconds) for one backend, plus a sample skeleton.

    Toggles the module-level `dijkstra3d_sparse` handle to force the backend, and
    always restores it. One untimed warmup precedes the timed runs.
    """
    saved = teasar._dijkstra3d_sparse
    if not use_lib:
        teasar._dijkstra3d_sparse = None
    try:
        sk = sc.teasar_skeletonize(voxels, **kw)  # warmup (and returned sample)
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            sc.teasar_skeletonize(voxels, **kw)
            best = min(best, time.perf_counter() - t0)
    finally:
        teasar._dijkstra3d_sparse = saved
    return best, sk


def _shapes(quick, neuron):
    """(label, voxels) sweep, ordered by graph size."""
    if quick:
        shapes = [
            ("cylinder r4 L20", solid_cylinder(4, 20)),
            ("cube 10", solid_cube(10)),
        ]
    else:
        shapes = [
            ("cylinder r4 L20", solid_cylinder(4, 20)),
            ("cylinder r8 L80", solid_cylinder(8, 80)),
            ("cube 12", solid_cube(12)),
            ("cube 20", solid_cube(20)),
        ]
    if neuron:
        path = os.path.join(_TESTS, "10075_scale3.npy")
        if os.path.exists(path):
            # //8 keeps ~36k voxels of real data - large enough to show the
            # accelerator's edge, small enough that exact mode stays tractable.
            v = unique((np.load(path) // 8).astype(np.int64), axis=0)
            shapes.append((f"neuron //8 ({len(v)})", v))
        else:
            print(f"(neuron fixture not found at {path}; skipping)\n")
    return shapes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3, help="timed runs per cell (best-of)")
    ap.add_argument("--quick", action="store_true", help="smaller shapes, fewer of them")
    ap.add_argument("--neuron", action="store_true", help="include the real neuron fixture")
    ap.add_argument(
        "--modes", default="exact,tree,fast",
        help="comma-separated branching modes to sweep",
    )
    args = ap.parse_args()

    have_lib = teasar._dijkstra3d_sparse is not None
    ver = getattr(teasar._dijkstra3d_sparse, "__version__", "?") if have_lib else "n/a"
    print(f"dijkstra3d_sparse: {'installed v' + ver if have_lib else 'NOT installed'}")
    if not have_lib:
        print("  -> nothing to compare; install the library to benchmark the accelerator.")
        return
    print(f"repeats={args.repeats} (best-of)  modes={args.modes}\n")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    shapes = _shapes(args.quick, args.neuron)

    header = f"{'shape':<20} {'voxels':>7} {'mode':<6} {'scipy(s)':>9} {'ds(s)':>9} {'speedup':>8}  {'topology'}"
    print(header)
    print("-" * len(header))

    speedups = []
    for label, vox in shapes:
        for mode in modes:
            t_sp, sk_sp = _time(vox, use_lib=False, repeats=args.repeats, branching=mode)
            t_ds, sk_ds = _time(vox, use_lib=True, repeats=args.repeats, branching=mode)
            sig_sp, sig_ds = _signature(sk_sp), _signature(sk_ds)
            # betti (components, first_betti) must match; node/edge counts may
            # differ by a tie-break, so report them when they do.
            if sig_sp[2:] == sig_ds[2:] and sig_sp[:2] == sig_ds[:2]:
                topo = "match"
            elif sig_sp[2:] == sig_ds[2:]:
                topo = f"~match (n {sig_sp[0]}/{sig_ds[0]})"
            else:
                topo = f"DIFF {sig_sp} vs {sig_ds}"
            speedup = t_sp / t_ds if t_ds > 0 else float("inf")
            speedups.append(speedup)
            print(
                f"{label:<20} {len(vox):>7} {mode:<6} "
                f"{t_sp:>9.4f} {t_ds:>9.4f} {speedup:>7.2f}x  {topo}"
            )

    if speedups:
        gm = float(np.exp(np.mean(np.log(speedups))))  # geometric mean of ratios
        print("-" * len(header))
        print(f"geometric-mean speedup (scipy / dijkstra3d_sparse): {gm:.2f}x")
        print("(>1 means the accelerator is faster; end-to-end, Dijkstra is one stage)")


if __name__ == "__main__":
    main()
