import importlib.util

import numpy as np
import pytest

import sparsecubes as sc
from sparsecubes import Skeleton
from _shapes import line, y_branch, annulus, solid_cylinder, betti

HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_NETWORKX = importlib.util.find_spec("networkx") is not None


def test_thin_skeletonize_returns_skeleton():
    skel = sc.thin_skeletonize(line(10))
    assert isinstance(skel, Skeleton)
    assert skel.nodes.shape[1] == 3
    assert skel.edges.shape[1] == 2
    # Edge indices are in range.
    assert skel.edges.min() >= 0
    assert skel.edges.max() < len(skel.nodes)


def test_thin_skeletonize_matches_centerline_of_thin():
    v = solid_cylinder(4, 14)
    a = sc.thin_skeletonize(v)
    b = sc.centerline(sc.thin(v))
    assert np.array_equal(a.nodes, b.nodes)
    assert np.array_equal(a.edges, b.edges)


def test_line_has_two_ends_no_branch():
    skel = sc.thin_skeletonize(line(10))
    deg = skel.node_degrees()
    assert (deg == 1).sum() == 2          # exactly two endpoints
    assert (deg >= 3).sum() == 0          # no branch points
    assert (deg == 2).sum() == len(skel.nodes) - 2


def test_y_branch_has_one_branch_three_ends():
    # centerline on an already-thin Y: exactly three tips and at least one
    # branch node (degree >= 3).
    skel = sc.centerline(y_branch(7))
    deg = skel.node_degrees()
    assert (deg == 1).sum() == 3
    assert (deg >= 3).sum() >= 1


def test_edges_are_26_connected():
    skel = sc.thin_skeletonize(solid_cylinder(4, 14))
    coords = skel.nodes.astype(np.int64)
    for a, b in skel.edges:
        d = np.abs(coords[a] - coords[b])
        assert d.max() == 1                # Chebyshev distance 1 (26-connected)


def test_annulus_skeleton_keeps_loop():
    skel = sc.thin_skeletonize(annulus(9, 5))
    _, b1 = betti(skel.nodes, skel.edges)
    assert b1 >= 1


def test_prune_spurs_removes_short_hair():
    # A long main line with a short perpendicular spur near the middle. The
    # threshold sits between the spur length (~2) and each main arm (~9).
    main = line(20, axis=0)
    spur = np.array([[10, 1, 0], [10, 2, 0], [10, 3, 0]], dtype=np.int64)
    shape = np.unique(np.vstack([main, spur]), axis=0)

    keep_all = sc.centerline(shape, min_branch_length=0)
    pruned = sc.centerline(shape, min_branch_length=5)

    assert len(pruned.nodes) < len(keep_all.nodes)
    surviving = set(map(tuple, pruned.nodes.tolist()))
    # The two ends of the long main line survive; the short spur tip does not.
    assert (0, 0, 0) in surviving
    assert (19, 0, 0) in surviving
    assert (10, 3, 0) not in surviving


def test_to_swc_is_a_valid_forest():
    skel = sc.thin_skeletonize(y_branch(7))
    swc = skel.to_swc()
    assert swc.shape == (len(skel.nodes), 7)

    ids = swc[:, 0].astype(int)
    parents = swc[:, 6].astype(int)
    assert np.array_equal(ids, np.arange(1, len(skel.nodes) + 1))
    # Exactly one root per connected component; no node is its own parent.
    n_roots = int((parents == -1).sum())
    n_components = betti(skel.nodes, skel.edges)[0]
    assert n_roots == n_components
    assert np.all(parents != ids)


def test_to_swc_writes_file(tmp_path):
    skel = sc.thin_skeletonize(line(8))
    out = tmp_path / "skel.swc"
    skel.to_swc(filepath=str(out))
    assert out.exists()
    loaded = np.loadtxt(str(out))
    assert loaded.shape == (len(skel.nodes), 7)


def test_to_path3d():
    trimesh = pytest.importorskip("trimesh")
    skel = sc.thin_skeletonize(line(8))
    path = skel.to_path3d()
    assert path.vertices.shape[0] == len(skel.nodes)
    assert len(path.entities) == len(skel.edges)


def test_spacing_scales_vertices_not_topology():
    spacing = np.array([2.0, 3.0, 4.0])
    plain = sc.thin_skeletonize(solid_cylinder(4, 14))
    scaled = sc.thin_skeletonize(solid_cylinder(4, 14), spacing=spacing)
    # Same graph, vertices scaled.
    assert np.array_equal(plain.nodes, scaled.nodes)
    assert np.array_equal(plain.edges, scaled.edges)
    assert np.allclose(scaled.vertices, plain.nodes.astype(float) * spacing)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_radii_are_positive_and_sized():
    v = solid_cylinder(4, 16)
    skel = sc.thin_skeletonize(v, radii=True)
    assert skel.radii is not None
    assert skel.radii.shape == (len(skel.nodes),)
    # Distance to background is always positive (nodes are inside the object).
    assert np.all(skel.radii > 0)
    # Interior centerline of a radius-4 cylinder sits ~4-5 voxels from empty space.
    assert 3 < np.median(skel.radii) < 6.5


def test_radii_without_object_raises():
    with pytest.raises(ValueError, match="radii"):
        sc.centerline(line(10), radii=True)


@pytest.mark.skipif(not HAS_NETWORKX, reason="networkx not installed")
def test_to_networkx():
    skel = sc.thin_skeletonize(y_branch(7))
    g = skel.to_networkx()
    assert g.number_of_nodes() == len(skel.nodes)
    assert g.number_of_edges() == len(skel.edges)
    assert "pos" in g.nodes[0]
