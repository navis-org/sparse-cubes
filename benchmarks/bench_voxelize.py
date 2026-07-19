#!/usr/bin/env python
"""Benchmark ``sc.voxelize`` against trimesh's own voxelizer.

Trimesh can already produce sparse *surface* voxels
(``mesh.voxelized(pitch, method="subdivide")`` never allocates a dense array), so
on the surface path the two are directly comparable - though trimesh's is an
approximation: it subdivides faces until the edges are sub-voxel and keeps the
cells containing the resulting vertices, which misses cells a triangle only clips
through a corner. ``sc.voxelize`` runs an exact separating-axis test instead, so
it is a strict superset; the reported "extra" column is that difference.

For *solid* voxelization there is no sparse alternative in trimesh at all: every
fill path materializes the bounding box (``fill('holes')`` calls
``scipy.ndimage.binary_fill_holes`` on a dense array; ``fill('base')`` allocates a
cube of the largest coordinate). The dense-grid size a given run would have needed
is reported alongside the sparse peak so the gap is visible.

Usage
-----
    python benchmarks/bench_voxelize.py
    python benchmarks/bench_voxelize.py --quick
    python benchmarks/bench_voxelize.py --neuron      # add the real fixture
"""

import argparse
import os
import sys
import time
import tracemalloc
import warnings

import numpy as np
import trimesh as tm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sparsecubes as sc  # noqa: E402


def timed(fn):
    """Run `fn`, returning (result, seconds, peak_bytes)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return out, dt, peak


def as_set(a):
    return set(map(tuple, np.asarray(a, np.int64).tolist()))


def bench(name, mesh, spacing, do_trimesh=True):
    bbox = np.asarray(mesh.bounds)
    dense_cells = np.prod(np.ceil((bbox[1] - bbox[0]) / spacing) + 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solid, t_solid, m_solid = timed(lambda: sc.voxelize(mesh, spacing))
        shell, t_shell, m_shell = timed(
            lambda: sc.voxelize(mesh, spacing, solid=False)
        )

    print(f"\n{name}  (spacing {spacing}, {len(mesh.faces)} faces)")
    print(f"  dense bool grid would be {dense_cells / 1e6:>10.1f} MB")
    print(
        f"  sc.voxelize solid   {len(solid):>9d} vox  "
        f"{t_solid:>7.2f}s  peak {m_solid / 1e6:>7.1f} MB"
    )
    print(
        f"  sc.voxelize surface {len(shell):>9d} vox  "
        f"{t_shell:>7.2f}s  peak {m_shell / 1e6:>7.1f} MB"
    )

    if not do_trimesh:
        return
    try:
        vg, t_tm, m_tm = timed(lambda: mesh.voxelized(pitch=spacing))
        origin = np.round(np.asarray(vg.transform)[:3, 3] / spacing).astype(int)
        theirs = as_set(vg.sparse_indices + origin)
    except Exception as exc:  # pragma: no cover - depends on optional deps
        print(f"  trimesh surface     failed: {exc}")
        return
    mine = as_set(shell)
    print(
        f"  trimesh surface     {len(theirs):>9d} vox  "
        f"{t_tm:>7.2f}s  peak {m_tm / 1e6:>7.1f} MB"
    )
    print(
        f"    -> {len(theirs - mine)} cell(s) trimesh has that we lack, "
        f"{len(mine - theirs)} that only the exact test finds"
    )

    try:
        _, t_fill, m_fill = timed(lambda: mesh.voxelized(pitch=spacing).fill("holes"))
        print(
            f"  trimesh solid       (dense fill)  {t_fill:>7.2f}s  "
            f"peak {m_fill / 1e6:>7.1f} MB"
        )
    except Exception as exc:  # pragma: no cover
        print(f"  trimesh solid       failed: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer / coarser runs")
    ap.add_argument("--neuron", action="store_true", help="add the real fixture")
    args = ap.parse_args()

    cases = [
        ("icosphere r=20", tm.creation.icosphere(subdivisions=4, radius=20.0), 1.0),
        ("torus", tm.creation.torus(major_radius=20, minor_radius=6), 0.5),
    ]
    if not args.quick:
        cases += [
            ("icosphere r=20 fine", tm.creation.icosphere(subdivisions=4, radius=20.0), 0.25),
            ("box 40^3", tm.creation.box(extents=[40, 40, 40]), 0.5),
        ]
    for name, mesh, spacing in cases:
        bench(name, mesh, spacing)

    if args.neuron:
        data = os.path.join(os.path.dirname(__file__), "..", "tests", "10075_coarse.npy")
        if not os.path.exists(data):
            print("\n(neuron fixture not found)")
            return
        vox = np.load(data)
        mesh = sc.mesh(vox, smooth=False)
        for spacing in (1.0, 0.5):
            # trimesh's subdivide path is very slow on a 130k-face mesh at these
            # pitches, so only the sparse numbers are reported here.
            bench(f"neuron ({len(vox)} vox)", mesh, spacing, do_trimesh=False)


if __name__ == "__main__":
    main()
