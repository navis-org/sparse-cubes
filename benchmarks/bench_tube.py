#!/usr/bin/env python
"""Benchmark and characterize ``measure.tube_profile``.

Two jobs in one script, because they want the same runs.

**Cost.** Wall time and peak RSS for the raycast, across shapes and both
membership backends. Each cell runs in its own subprocess: ``ru_maxrss`` is a
process-wide high-water mark, so back-to-back runs in one process contaminate
each other and every number after the first is really the largest one so far.

**Characterization.** The parametric-tube idea rests on four measurements, and
they fall out of the same runs rather than needing their own scaffolding:

1. ``m_k`` against ``k`` - where the spectrum knees is where ``K`` should be cut.
2. residual after truncating at each ``k``, in physical units - the actual error
   budget, not a proxy for it.
3. the star-shapedness failure rate - the hard ceiling on the whole approach,
   since ``r(theta)`` cannot represent a cross-section that is not star-shaped
   about its skeleton point.
4. ``m_k`` binned by the skeleton tangent's direction - at 4x4x40 nm the radial
   quantization is direction-dependent, so the noise floor on ``m_k`` is *not*
   flat and it is easy to mistake that artifact for real structure. The synthetic
   ``aniso`` rows sweep one known tube through a range of orientations, which is
   the control that separates the two.

Usage
-----
    python benchmarks/bench_tube.py                  # default sweep
    python benchmarks/bench_tube.py --quick          # small shapes only
    python benchmarks/bench_tube.py --neuron         # add the real fixture
    python benchmarks/bench_tube.py --n-theta 128 --K 8
    python benchmarks/bench_tube.py --in-process     # skip the RSS subprocesses
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

# `_shapes` and the neuron fixture live under tests/.
_TESTS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "tests"))
sys.path.insert(0, _TESTS)

import sparsecubes as sc  # noqa: E402
from sparsecubes import tube as _tube  # noqa: E402
from sparsecubes.core import unique  # noqa: E402

from _shapes import (  # noqa: E402
    _ball,
    axial_skeleton,
    elliptic_cylinder,
    lobed_cylinder,
    self_touch_hairpin,
    solid_cylinder,
)

# The number of harmonics reported in the spectrum table, independent of the `K`
# actually stored - the whole question is where the spectrum stops being worth
# storing, which cannot be answered from a truncated one.
_PROBE_K = 16


# ---------------------------------------------------------------------------
# workloads
# ---------------------------------------------------------------------------


def _tilted_tube(radius=6, length=60, tilt=0.0):
    """A solid cylinder of known radius, rotated `tilt` radians in the x-z plane.

    The anisotropy control: identical geometry at every tilt, so any change in the
    measured spectrum is quantization aliasing into ``k``, not shape.
    """
    z = np.arange(length) - length / 2.0
    axis = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    centres = z[:, None] * axis[None, :]

    disc = _ball((0, 0, 0), radius)
    # Keep only the slab of the ball perpendicular to the axis, so stacking it
    # along the axis sweeps a cylinder rather than a string of spheres.
    disc = disc[np.abs(disc @ axis) <= 0.5]

    pts = np.rint(centres[:, None, :] + disc[None, :, :]).reshape(-1, 3).astype(np.int64)
    return unique(pts - pts.min(axis=0), axis=0)


def _workloads(quick, neuron):
    """``(label, voxels, skeleton_or_None, spacing)`` rows, ordered by size."""
    rows = [
        ("disc r8 L40", lobed_cylinder(8, 4, 0.0, 40), "axial", None),
        ("ellipse 10x5", elliptic_cylinder(10, 5, 40), "axial", None),
        ("lobed k3 r8", lobed_cylinder(8, 3, 1.5, 40), "axial", None),
    ]
    if not quick:
        rows += [
            ("cylinder r8 L80", solid_cylinder(8, 80), None, None),
            ("hairpin", self_touch_hairpin(), None, None),
            ("disc r8 aniso z", lobed_cylinder(8, 4, 0.0, 40), "axial", (4, 4, 40)),
        ]
        for tilt in (0.0, 0.4, 0.8, 1.2, 1.57):
            rows.append((f"tilt {tilt:.2f} aniso", _tilted_tube(6, 60, tilt), None,
                         (4, 4, 40)))
    if neuron:
        path = os.path.join(_TESTS, "10075_scale3.npy")
        if not os.path.exists(path):
            print(f"(fixture not found: {path}; skipping)")
        else:
            full = np.load(path).astype(np.int64)
            rows.append((f"neuron //4 ({len(unique(full // 4, axis=0))})",
                         unique(full // 4, axis=0), None, (64, 64, 64)))
            rows.append((f"neuron ({len(full)})", full, None, (16, 16, 16)))
    return rows


def _skeleton(voxels, kind, spacing):
    if kind == "axial":
        return axial_skeleton(voxels, spacing=spacing)
    return sc.wavefront_skeletonize(voxels, spacing=spacing)


# ---------------------------------------------------------------------------
# one measured cell
# ---------------------------------------------------------------------------


def _run_one(label, quick, neuron, backend, n_theta, diagnostics):
    """Profile one workload and return a JSON-able record. Runs in the subprocess."""
    row = next(r for r in _workloads(quick, neuron) if r[0] == label)
    _, voxels, kind, spacing = row

    t0 = time.perf_counter()
    skel = _skeleton(voxels, kind, spacing)
    t_skel = time.perf_counter() - t0

    t0 = time.perf_counter()
    p = _tube.tube_profile(voxels, skel, K=_PROBE_K, n_theta=n_theta,
                           backend=backend, diagnostics=diagnostics)
    t_tube = time.perf_counter() - t0

    # Interior nodes only: an end cap's "cross-section" is not one, and a junction's
    # is undefined. Both are flagged, which is what makes this filter honest rather
    # than cherry-picked.
    keep = ~(p.flag("branch_end") | p.flag("junction") | p.flag("seed_outside"))
    if keep.sum() < 8:
        keep = np.ones(len(p.nodes), dtype=bool)
    mag = p.mag[keep].astype(float)

    # Residual if we had truncated at each k: Parseval on the harmonics dropped,
    # added in quadrature to what `_PROBE_K` itself discarded.
    tail = np.sqrt(np.cumsum((mag**2)[:, ::-1], axis=1)[:, ::-1] / 2.0)
    resid = np.sqrt(tail**2 + (p.residual[keep].astype(float) ** 2)[:, None])

    _, _, tangent = p.frame_vectors()
    return {
        "label": label,
        "n_voxels": int(len(voxels)),
        "n_nodes": int(len(p.nodes)),
        "n_scored": int(keep.sum()),
        "t_skel": t_skel,
        "t_tube": t_tube,
        "rays": int(len(p.nodes) * n_theta),
        "a0": float(p.a0[keep].mean()),
        "mag": mag.mean(axis=0).tolist(),
        "resid_at_k": resid.mean(axis=0).tolist(),
        "resid_frac": (resid.mean(axis=0) / max(float(p.a0[keep].mean()), 1e-9)).tolist(),
        "non_star": float(p.flag("non_star")[keep].mean()),
        "non_star_rays": float(p.non_star[keep].mean()),
        "escaped": float(p.flag("ray_escaped").mean()),
        "junction": float(p.flag("junction").mean()),
        # |tangent . z|: 1 = running along the anisotropic axis, 0 = in-plane.
        "tilt": float(np.abs(tangent[keep, 2]).mean()),
        "mag_by_tilt": _bin_by_tilt(np.abs(tangent[keep, 2]), mag),
        "peak_rss_mb": _peak_rss_mb(),
    }


def _bin_by_tilt(tilt, mag):
    """Mean ``m_k`` in three tangent-direction bins - the anisotropy diagnostic."""
    out = {}
    for name, lo, hi in [("in-plane", 0.0, 0.34), ("oblique", 0.34, 0.67),
                         ("along-z", 0.67, 1.01)]:
        sel = (tilt >= lo) & (tilt < hi)
        out[name] = mag[sel].mean(axis=0).tolist() if sel.any() else None
    return out


def _peak_rss_mb():
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kB, macOS bytes.
    return peak / (1024.0 if sys.platform.startswith("linux") else 1024.0**2)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _spawn(label, args, backend):
    """Run one cell in a fresh process so its peak RSS is attributable."""
    cmd = [sys.executable, os.path.abspath(__file__), "--child", label,
           "--backend", backend, "--n-theta", str(args.n_theta)]
    if args.quick:
        cmd.append("--quick")
    if args.neuron:
        cmd.append("--neuron")
    if args.diagnostics:
        cmd.append("--diagnostics")
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        return {"label": label, "error": (out.stderr or "").strip().splitlines()[-1:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


def _print_cost(records):
    head = (f"{'shape':<24} {'voxels':>9} {'nodes':>7} {'rays':>9} {'t_skel':>7} "
            f"{'t_tube':>7} {'us/ray':>7} {'peakMB':>7}")
    print(head)
    print("-" * len(head))
    for r in records:
        if "error" in r:
            print(f"{r['label']:<24} FAILED: {r['error']}")
            continue
        print(f"{r['label']:<24} {r['n_voxels']:>9} {r['n_nodes']:>7} {r['rays']:>9} "
              f"{r['t_skel']:>7.2f} {r['t_tube']:>7.2f} "
              f"{1e6 * r['t_tube'] / max(r['rays'], 1):>7.2f} {r['peak_rss_mb']:>7.0f}")


def _print_spectrum(records, K):
    print("\n=== mean m_k by harmonic (physical units; a0 for scale) ===")
    ks = list(range(1, _PROBE_K + 1))
    print(f"{'shape':<24} {'a0':>8} " + " ".join(f"{'m' + str(k):>8}" for k in ks[:10]))
    print("-" * (33 + 9 * 10))
    for r in records:
        if "error" in r:
            continue
        print(f"{r['label']:<24} {r['a0']:>8.3f} "
              + " ".join(f"{m:>8.4f}" for m in r["mag"][:10]))

    print(f"\n=== residual if truncated at k, as a fraction of a0 "
          f"(K={K} is the shipped default) ===")
    print(f"{'shape':<24} " + " ".join(f"{'k=' + str(k):>8}" for k in ks[:10]))
    print("-" * (25 + 9 * 10))
    for r in records:
        if "error" in r:
            continue
        print(f"{r['label']:<24} "
              + " ".join(f"{v:>8.4f}" for v in r["resid_frac"][:10]))


def _print_limits(records):
    print("\n=== failure modes - the ceiling on the approach ===")
    print("(non-star measured with the default star_window; it is a statement about")
    print(" the cross-section, so it must not be read as a function of max_radius)")
    head = (f"{'shape':<24} {'nonstar/node':>13} {'nonstar/ray':>12} {'escaped':>9} "
            f"{'junction':>9} {'|t.z|':>7}")
    print(head)
    print("-" * len(head))
    for r in records:
        if "error" in r:
            continue
        print(f"{r['label']:<24} {r['non_star']:>13.3f} {r['non_star_rays']:>12.3f} "
              f"{r['escaped']:>9.3f} {r['junction']:>9.3f} {r['tilt']:>7.3f}")

    print("\n=== m_k by tangent direction - the anisotropy artifact ===")
    print("(identical geometry must give identical rows; any spread is quantization)")
    print(f"{'shape':<24} {'bin':<10} " + " ".join(f"{'m' + str(k):>8}" for k in range(1, 7)))
    print("-" * (35 + 9 * 6))
    for r in records:
        if "error" in r:
            continue
        for name, vals in r["mag_by_tilt"].items():
            if vals is None:
                continue
            print(f"{r['label']:<24} {name:<10} "
                  + " ".join(f"{v:>8.4f}" for v in vals[:6]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small shapes only")
    ap.add_argument("--neuron", action="store_true", help="include the real fixture")
    ap.add_argument("--n-theta", type=int, default=64, help="rays cast per node")
    ap.add_argument("--K", type=int, default=4, help="the truncation to highlight")
    ap.add_argument("--diagnostics", action="store_true", default=True,
                    help="measure the star-shapedness rate (on by default here)")
    ap.add_argument("--no-diagnostics", dest="diagnostics", action="store_false")
    ap.add_argument("--backends", default="exits",
                    help="comma-separated: exits, sparse, keys")
    ap.add_argument("--in-process", action="store_true",
                    help="skip the per-cell subprocess (peak RSS becomes meaningless)")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--backend", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:  # subprocess mode: one cell, one JSON line on stdout
        print(json.dumps(_run_one(args.child, args.quick, args.neuron, args.backend,
                                  args.n_theta, args.diagnostics)))
        return

    labels = [r[0] for r in _workloads(args.quick, args.neuron)]
    for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
        print(f"\n########## backend={backend}  n_theta={args.n_theta} "
              f"diagnostics={args.diagnostics} ##########\n")
        records = [
            _run_one(lab, args.quick, args.neuron, backend, args.n_theta,
                     args.diagnostics)
            if args.in_process
            else _spawn(lab, args, backend)
            for lab in labels
        ]
        _print_cost(records)
        _print_spectrum(records, args.K)
        _print_limits(records)


if __name__ == "__main__":
    main()
