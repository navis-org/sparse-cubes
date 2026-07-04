"""Tests for the sparse TEASAR / kimimaro skeletonizer (`teasar_skeletonize`).

TEASAR differs from the topological thinner in one important way that these tests
pin down: it always returns an acyclic tree/forest (loops are broken), whereas
`thin` preserves loops. Everything runs on the sparse voxel set - the tests also
guard that memory stays bounded on a large real neuron.
"""

import os

import numpy as np
import pytest

pytest.importorskip("scipy")

import sparsecubes as sc
from sparsecubes import teasar as _teasar
from sparsecubes.skeleton import Skeleton
from sparsecubes.teasar import _neighbors_26, _match_radii

from _shapes import (
    line,
    l_shape,
    y_branch,
    solid_cube,
    slab,
    solid_cylinder,
    annulus,
    betti,
    n_components,
    is_subset,
)


SHAPES = {
    "line": line(10),
    "l_shape": l_shape(),
    "y_branch": y_branch(7),
    "cube3": solid_cube(3),
    "cube5": solid_cube(5),
    "slab": slab(6),
    "cylinder": solid_cylinder(4, 20),
    "annulus": annulus(9, 5),
}


def _cheb(nodes, edges):
    """Chebyshev (max-axis) distance of every edge - 1 means 26-adjacent."""
    if len(edges) == 0:
        return np.zeros(0, dtype=np.int64)
    d = np.abs(nodes[edges[:, 0]] - nodes[edges[:, 1]])
    return d.max(axis=1)


# --- basic contract --------------------------------------------------------

def test_returns_skeleton():
    sk = sc.teasar_skeletonize(line(10))
    assert isinstance(sk, Skeleton)
    assert sk.nodes.ndim == 2 and sk.nodes.shape[1] == 3
    assert sk.edges.ndim == 2 and sk.edges.shape[1] == 2
    assert sk.radii is not None and len(sk.radii) == len(sk.nodes)


def test_preserves_input_dtype():
    v = line(10).astype(np.uint32)
    sk = sc.teasar_skeletonize(v)
    assert sk.nodes.dtype == np.uint32


@pytest.mark.parametrize("name", list(SHAPES))
def test_output_is_subset(name):
    sk = sc.teasar_skeletonize(SHAPES[name])
    assert is_subset(sk.nodes, SHAPES[name])


@pytest.mark.parametrize("name", list(SHAPES))
def test_output_is_acyclic_forest(name):
    # The defining property of TEASAR: no loops. first_betti == 0.
    sk = sc.teasar_skeletonize(SHAPES[name])
    ncomp, first_betti = betti(sk.nodes, sk.edges)
    assert first_betti == 0
    assert len(sk.edges) == len(sk.nodes) - ncomp


@pytest.mark.parametrize("name", list(SHAPES))
def test_preserves_component_count(name):
    sk = sc.teasar_skeletonize(SHAPES[name])
    ncomp, _ = betti(sk.nodes, sk.edges)
    assert ncomp == n_components(SHAPES[name])


@pytest.mark.parametrize("name", list(SHAPES))
def test_edges_are_26_connected(name):
    sk = sc.teasar_skeletonize(SHAPES[name])
    ch = _cheb(sk.nodes, sk.edges)
    assert np.all(ch == 1)


@pytest.mark.parametrize("name", list(SHAPES))
def test_radii_are_positive(name):
    sk = sc.teasar_skeletonize(SHAPES[name])
    assert np.all(sk.radii > 0)


# --- topology specifics ----------------------------------------------------

def test_line_has_two_ends_no_branch():
    sk = sc.teasar_skeletonize(line(12))
    deg = sk.node_degrees()
    assert int((deg == 1).sum()) == 2
    assert int((deg >= 3).sum()) == 0
    # a straight line collapses to itself: one node per voxel
    assert len(sk.nodes) == 12


def test_y_branch_one_junction_three_ends():
    sk = sc.teasar_skeletonize(y_branch(7))
    deg = sk.node_degrees()
    assert int((deg >= 3).sum()) == 1
    assert int((deg == 1).sum()) == 3


def test_annulus_loop_is_broken():
    # Contrast with thinning, which keeps the loop (first_betti >= 1).
    ring = annulus(9, 5)
    sk = sc.teasar_skeletonize(ring)
    ncomp, first_betti = betti(sk.nodes, sk.edges)
    assert ncomp == 1
    assert first_betti == 0  # TEASAR breaks the cycle into an open curve


def test_cylinder_collapses_to_axial_curve():
    r, L = 4, 20
    sk = sc.teasar_skeletonize(solid_cylinder(r, L))
    # a curve, not a blob: node count on the order of the length, not the volume
    assert len(sk.nodes) < 3 * L
    # spans the full length along z
    assert sk.nodes[:, 2].min() <= 1
    assert sk.nodes[:, 2].max() >= L - 2
    # DBF radius in the interior is close to the true radius
    assert abs(np.median(sk.radii) - r) <= 1.5


def test_cube_not_emptied():
    sk = sc.teasar_skeletonize(solid_cube(5))
    assert len(sk.nodes) >= 1
    ncomp, first_betti = betti(sk.nodes, sk.edges)
    assert ncomp == 1 and first_betti == 0
    assert np.all(sk.radii > 0)


def test_disconnected_gives_forest():
    two = np.vstack([line(6), line(6) + np.array([0, 0, 40])]).astype(np.int64)
    sk = sc.teasar_skeletonize(two)
    ncomp, first_betti = betti(sk.nodes, sk.edges)
    assert ncomp == 2
    assert first_betti == 0


# --- degenerate inputs -----------------------------------------------------

def test_single_voxel():
    sk = sc.teasar_skeletonize(np.array([[3, 3, 3]], dtype=np.int64))
    assert len(sk.nodes) == 1
    assert len(sk.edges) == 0
    assert sk.radii[0] > 0


def test_empty():
    sk = sc.teasar_skeletonize(np.empty((0, 3), dtype=np.int64))
    assert len(sk.nodes) == 0
    assert len(sk.edges) == 0


# --- spacing / roots / options --------------------------------------------

def test_unit_spacing_matches_none():
    # spacing=(1,1,1) is a no-op relative to spacing=None (same const default too)
    cyl = solid_cylinder(4, 20)
    sk_none = sc.teasar_skeletonize(cyl)
    sk_unit = sc.teasar_skeletonize(cyl, spacing=(1, 1, 1))
    assert np.array_equal(sk_none.nodes, sk_unit.nodes)
    assert np.array_equal(sk_none.edges, sk_unit.edges)


def test_spacing_scales_vertices():
    # `vertices` is `nodes * spacing`; anisotropy legitimately feeds the algorithm
    # (edge weights + invalidation ball are physical), so topology may differ - but
    # the result is still a single acyclic curve.
    cyl = solid_cylinder(4, 20)
    sk = sc.teasar_skeletonize(cyl, spacing=(1, 1, 5))
    assert np.allclose(sk.vertices, sk.nodes.astype(float) * np.array([1, 1, 5]))
    ncomp, fb = betti(sk.nodes, sk.edges)
    assert ncomp == 1 and fb == 0


@pytest.mark.parametrize("root", ["geodesic", "dbf", (0, 0, 0)])
def test_root_options(root):
    sk = sc.teasar_skeletonize(y_branch(7), root=root)
    ncomp, fb = betti(sk.nodes, sk.edges)
    assert ncomp == 1 and fb == 0
    # still recovers the three tips
    assert int((sk.node_degrees() == 1).sum()) == 3


@pytest.mark.parametrize("branching", ["exact", "tree"])
def test_branching_exact_and_tree(branching):
    sk = sc.teasar_skeletonize(y_branch(7), branching=branching)
    ncomp, fb = betti(sk.nodes, sk.edges)
    assert ncomp == 1 and fb == 0
    assert int((sk.node_degrees() >= 3).sum()) == 1


def test_min_branch_length_prunes_spur():
    # long trunk with a perpendicular side branch that survives invalidation
    trunk = np.array([[i, 0, 0] for i in range(30)])
    spur = np.array([[15, j, 0] for j in range(1, 9)])
    shape = np.unique(np.vstack([trunk, spur]), axis=0).astype(np.int64)

    raw = sc.teasar_skeletonize(shape, min_branch_length=0)
    pruned = sc.teasar_skeletonize(shape, min_branch_length=100)
    # radii stay aligned with nodes through pruning
    assert len(pruned.radii) == len(pruned.nodes)
    # pruning a big threshold removes the side branch -> fewer nodes, no junction
    assert len(pruned.nodes) < len(raw.nodes)
    assert int((pruned.node_degrees() >= 3).sum()) == 0


def test_dbf_term_runs():
    sk = sc.teasar_skeletonize(solid_cylinder(4, 20), dbf_term=True)
    ncomp, fb = betti(sk.nodes, sk.edges)
    assert ncomp == 1 and fb == 0


# --- branching='fast' / batched multi-source mode --------------------------

@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("branching", ["fast", 4])
def test_fast_branching_is_acyclic_subset(name, branching):
    sk = sc.teasar_skeletonize(SHAPES[name], branching=branching)
    ncomp, first_betti = betti(sk.nodes, sk.edges)
    assert first_betti == 0                       # still a tree/forest
    assert ncomp == n_components(SHAPES[name])    # components preserved
    assert is_subset(sk.nodes, SHAPES[name])
    assert np.all(sk.radii > 0)
    if len(sk.edges):
        assert np.all(_cheb(sk.nodes, sk.edges) == 1)


def test_fast_branching_recovers_line_and_branch():
    line_sk = sc.teasar_skeletonize(line(12), branching="fast")
    ld = line_sk.node_degrees()
    assert int((ld == 1).sum()) == 2 and int((ld >= 3).sum()) == 0

    y_sk = sc.teasar_skeletonize(y_branch(7), branching="fast")
    yd = y_sk.node_degrees()
    assert int((yd >= 3).sum()) == 1 and int((yd == 1).sum()) == 3


def test_fast_branching_annulus_still_breaks_loop():
    sk = sc.teasar_skeletonize(annulus(9, 5), branching="fast")
    ncomp, first_betti = betti(sk.nodes, sk.edges)
    assert ncomp == 1 and first_betti == 0


@pytest.mark.parametrize("bad", [0, -1, "nope", True, 1.5])
def test_branching_invalid_raises(bad):
    with pytest.raises(ValueError):
        sc.teasar_skeletonize(line(10), branching=bad)


# --- guards ----------------------------------------------------------------

def test_extent_guard():
    v = np.array([[0, 0, 0], [(1 << 21), 0, 0]], dtype=np.int64)
    with pytest.raises(ValueError):
        sc.teasar_skeletonize(v)


@pytest.mark.parametrize("bad", [
    [[0, 0, 0]],                                   # not an array
    np.zeros((4, 2), dtype=np.int64),              # wrong width
    np.zeros((4, 3), dtype=np.float64),            # non-integer
])
def test_bad_input(bad):
    with pytest.raises(TypeError):
        sc.teasar_skeletonize(bad)


# --- helpers ---------------------------------------------------------------

def test_neighbors_26_symmetric_and_offsets():
    from sparsecubes.thinning import _OFF26

    vox = np.unique(solid_cube(3), axis=0).astype(np.int64)
    nodes, src, dst, off_idx = _neighbors_26(vox)
    # narrow dtypes: int32 node indices, int8 offset index
    assert src.dtype == np.int32 and dst.dtype == np.int32
    assert off_idx.dtype == np.int8
    # the offset index reconstructs the actual coordinate difference
    off = _OFF26[off_idx]
    assert np.array_equal(off, nodes[dst] - nodes[src])
    # every half-edge has a mirror (symmetric adjacency)
    fwd = set(zip(src.tolist(), dst.tolist()))
    rev = set(zip(dst.tolist(), src.tolist()))
    assert fwd == rev
    # offsets are unit steps (Chebyshev 1)
    assert np.abs(off).max() == 1


def test_match_radii_recovers_values():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.int64)
    radii = np.array([1.0, 2.0, 3.0, 4.0])
    keep = nodes[[0, 2, 3]]
    assert np.allclose(_match_radii(nodes, radii, keep), [1.0, 3.0, 4.0])


# --- Dijkstra backend (dijkstra3d_sparse accelerator + scipy fallback) ------
#
# teasar prefers `dijkstra3d_sparse` (Dijkstra straight over the voxel
# coordinates) when installed and falls back to scipy's csgraph.dijkstra
# otherwise. The two must agree on topology; injected graphs always use scipy.

_has_ds = _teasar._dijkstra3d_sparse is not None


def test_scipy_fallback_runs_without_the_library(monkeypatch):
    # Force the accelerator "uninstalled": teasar must still skeletonize via scipy.
    monkeypatch.setattr(_teasar, "_dijkstra3d_sparse", None)
    for name in ("line", "y_branch", "cylinder", "annulus"):
        sk = sc.teasar_skeletonize(SHAPES[name])
        assert betti(sk.nodes, sk.edges)[1] == 0        # still an acyclic forest
        assert is_subset(sk.nodes, SHAPES[name])


@pytest.mark.skipif(not _has_ds, reason="dijkstra3d_sparse not installed")
@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("branching", ["exact", "tree", "fast"])
def test_backends_agree(name, branching, monkeypatch):
    # The accelerator and the scipy fallback must agree on topology: an acyclic
    # forest, the same component count, and a subset of the input. For `tree` and
    # `fast` the two run the same root/multi-source Dijkstra, so node/edge counts
    # match too; `exact` grafts incrementally on the accelerator (via
    # shortest_path_to_set) vs root-sourced on scipy, so only topology need match.
    shp = SHAPES[name]
    sk_ds = sc.teasar_skeletonize(shp, branching=branching)   # accelerator (installed)
    monkeypatch.setattr(_teasar, "_dijkstra3d_sparse", None)
    sk_sp = sc.teasar_skeletonize(shp, branching=branching)   # scipy fallback

    b_ds, b_sp = betti(sk_ds.nodes, sk_ds.edges), betti(sk_sp.nodes, sk_sp.edges)
    assert b_ds[1] == 0 and b_sp[1] == 0            # both acyclic forests
    assert b_ds[0] == b_sp[0]                       # same component count
    assert is_subset(sk_ds.nodes, shp) and is_subset(sk_sp.nodes, shp)
    if branching in ("tree", "fast"):              # identical algorithm -> same size
        assert len(sk_ds.nodes) == len(sk_sp.nodes)
        assert len(sk_ds.edges) == len(sk_sp.edges)


@pytest.mark.skipif(not _has_ds, reason="dijkstra3d_sparse not installed")
@pytest.mark.parametrize("name", list(SHAPES))
def test_incremental_grafting_is_a_valid_forest(name):
    # `branching="exact"` on the accelerator uses the incremental grafting path
    # (shortest_path_to_set). It must still produce an acyclic, 26-connected
    # forest that is a subset of the input with the right component count.
    shp = SHAPES[name]
    sk = sc.teasar_skeletonize(shp, branching="exact")
    ncomp, first_betti = betti(sk.nodes, sk.edges)
    assert first_betti == 0
    assert ncomp == n_components(shp)
    assert is_subset(sk.nodes, shp)
    assert np.all(_cheb(sk.nodes, sk.edges) == 1)   # every edge is 26-adjacent
    assert np.all(sk.radii > 0)


@pytest.mark.skipif(not _has_ds, reason="dijkstra3d_sparse not installed")
@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("branching", ["exact", "tree", "fast"])
def test_reusable_graph_matches_free_functions(name, branching, monkeypatch):
    # The reusable `Graph` handle (index built once) must give exactly the same
    # skeleton as the free-function fallback (index rebuilt per call) - same code
    # runs, only where the index is built moves. Skip if the installed build has
    # no `Graph` (then both paths are already the fallback).
    if not _teasar._HAS_GRAPH:
        pytest.skip("installed dijkstra3d_sparse has no Graph handle")
    shp = SHAPES[name]
    sk_graph = sc.teasar_skeletonize(shp, branching=branching)   # Graph handle
    monkeypatch.setattr(_teasar, "_HAS_GRAPH", False)            # force free-func path
    sk_free = sc.teasar_skeletonize(shp, branching=branching)
    assert np.array_equal(sk_graph.nodes, sk_free.nodes)
    assert np.array_equal(sk_graph.edges, sk_free.edges)
    assert np.array_equal(sk_graph.radii, sk_free.radii)


def test_injected_graph_ignores_accelerator(monkeypatch):
    # An injected/downsampled graph is authoritative and its cells need not be
    # coordinate-26-adjacent, so the coordinate-derived accelerator must never
    # engage: output is identical whether or not the library is importable.
    shp = solid_cylinder(4, 24)
    sk_on = sc.teasar_skeletonize(shp, downsample=2)
    monkeypatch.setattr(_teasar, "_dijkstra3d_sparse", None)
    sk_off = sc.teasar_skeletonize(shp, downsample=2)
    assert np.array_equal(sk_on.nodes, sk_off.nodes)
    assert np.array_equal(sk_on.edges, sk_off.edges)


# --- exporters -------------------------------------------------------------

def test_to_swc_is_valid_forest():
    sk = sc.teasar_skeletonize(y_branch(7))
    swc = sk.to_swc()
    assert swc.shape[1] == 7
    ids = set(swc[:, 0].astype(int))
    roots = 0
    for row in swc:
        parent = int(row[6])
        if parent == -1:
            roots += 1
        else:
            assert parent in ids  # every parent exists
    assert roots == 1  # one connected component -> one root


# --- cross-check against reference kimimaro --------------------------------

def test_matches_kimimaro_on_cylinder():
    kimimaro = pytest.importorskip("kimimaro")

    r, L = 5, 25
    cyl = solid_cylinder(r, L)
    ours = sc.teasar_skeletonize(cyl)

    # densify only for the reference implementation (tiny box, test-only)
    mn = cyl.min(0)
    dims = cyl.max(0) - mn + 1
    labels = np.zeros(tuple(dims), dtype=np.uint32)
    idx = (cyl - mn).T
    labels[idx[0], idx[1], idx[2]] = 1
    skels = kimimaro.skeletonize(
        labels,
        teasar_params={"scale": 1.5, "const": 4, "pdrf_scale": 100000, "pdrf_exponent": 4},
        anisotropy=(1, 1, 1),
        dust_threshold=0,
        fix_branching=True,
        progress=False,
    )
    ref = skels[1]

    # both are single acyclic curves spanning the cylinder length
    assert betti(ours.nodes, ours.edges)[1] == 0
    ours_zspan = ours.nodes[:, 2].max() - ours.nodes[:, 2].min()
    ref_zspan = ref.vertices[:, 2].max() - ref.vertices[:, 2].min()
    assert abs(ours_zspan - ref_zspan) <= 3
    # comparable medial radius
    assert abs(np.median(ours.radii) - np.median(ref.radius)) <= 1.5


# --- large real-neuron regression (bounded memory / time) ------------------

@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "10075_scale3.npy")),
    reason="large fixture not present",
)
def test_large_neuron_stays_bounded():
    import time
    import tracemalloc

    from sparsecubes.core import unique

    path = os.path.join(os.path.dirname(__file__), "10075_scale3.npy")
    v = np.load(path)
    # downsample so the regression runs quickly; still ~36k voxels of real data
    v = unique((v // 8).astype(np.int64), axis=0)

    tracemalloc.start()
    t0 = time.time()
    sk = sc.teasar_skeletonize(v)
    dt = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # completes, stays a forest, and never densifies the (huge) bounding box
    ncomp, first_betti = betti(sk.nodes, sk.edges)
    assert first_betti == 0
    assert len(sk.nodes) > 0
    assert dt < 30
    # dense bounding box here would be billions of voxels; peak must be far less
    assert peak < 1_500_000_000

    # fast branching: a valid tree, and faster than the exact extraction
    t0 = time.time()
    sk_a = sc.teasar_skeletonize(v, branching="fast")
    dt_a = time.time() - t0
    assert betti(sk_a.nodes, sk_a.edges)[1] == 0
    assert len(sk_a.nodes) > 0
    assert dt_a < dt
