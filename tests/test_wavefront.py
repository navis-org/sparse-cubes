"""Tests for the wavefront (Reeb-graph) skeletonizer."""

import numpy as np
import pytest

import sparsecubes as sc
from sparsecubes.wavefront import _contract, _longest_step, wavefront_skeletonize

from _shapes import (
    annulus,
    betti,
    line,
    n_components,
    self_touch_hairpin,
    solid_cube,
    solid_cylinder,
    y_branch,
)


SHAPES = {
    "line": line(8),
    "y_branch": y_branch(),
    "cylinder": solid_cylinder(4, 20),
    "cube": solid_cube(10),
    "annulus": annulus(),
    "hairpin": self_touch_hairpin(),
}


@pytest.mark.parametrize("name", list(SHAPES))
def test_component_count_is_preserved(name):
    """The skeleton must have exactly as many pieces as the object."""
    vox = SHAPES[name]
    sk = wavefront_skeletonize(vox)
    assert betti(sk.nodes, sk.edges)[0] == n_components(vox)


@pytest.mark.parametrize("name", list(SHAPES))
def test_tree_is_acyclic_and_nodes_lie_inside(name):
    vox = SHAPES[name]
    sk = wavefront_skeletonize(vox)
    assert betti(sk.nodes, sk.edges)[1] == 0
    assert len(sk.radii) == len(sk.nodes)
    # Centroids of subsets of the object cannot leave its bounding box.
    assert np.all(sk.nodes >= vox.min(axis=0) - 1e-9)
    assert np.all(sk.nodes <= vox.max(axis=0) + 1e-9)


def test_two_components_stay_separate():
    """Each component gets its own seed, so neither is left unreached."""
    vox = np.concatenate([line(8), line(8) + np.array([50, 50, 50])])
    sk = wavefront_skeletonize(vox)
    assert betti(sk.nodes, sk.edges)[0] == 2


def test_disconnected_single_voxels():
    vox = np.array([[0, 0, 0], [10, 10, 10]], dtype=np.int64)
    sk = wavefront_skeletonize(vox)
    assert len(sk.nodes) == 2
    assert len(sk.edges) == 0


def test_empty_input():
    sk = wavefront_skeletonize(np.empty((0, 3), dtype=np.int64))
    assert len(sk.nodes) == 0 and len(sk.edges) == 0


def test_raw_reeb_graph_keeps_genuine_loops_only():
    """`tree=False` must expose the object's own topology, not binning noise.

    A solid cylinder's distance function never splits, so its Reeb graph is a
    path; an annulus splits at the seed and re-merges on the far side, giving
    exactly one cycle. Both come out wrong if `step_size` drops below the
    longest graph step - see `test_step_size_below_the_floor_is_rejected`.
    """
    sk = wavefront_skeletonize(solid_cylinder(8, 40), tree=False)
    assert betti(sk.nodes, sk.edges) == (1, 0)

    sk = wavefront_skeletonize(annulus(), tree=False)
    assert betti(sk.nodes, sk.edges) == (1, 1)


def test_step_size_below_the_floor_is_rejected():
    with pytest.raises(ValueError, match="longest graph step"):
        wavefront_skeletonize(solid_cylinder(4, 20), step_size=1.0)
    # ... and is accepted exactly at the floor.
    wavefront_skeletonize(solid_cylinder(4, 20), step_size=_longest_step(None))


def test_longest_step_is_the_corner_diagonal():
    """The floor is the longest 26-edge, which anisotropy stretches."""
    assert _longest_step(None) == pytest.approx(np.sqrt(3), rel=1e-5)
    assert _longest_step([1.0, 4.0, 2.0]) == pytest.approx(np.sqrt(21), rel=1e-5)


def test_nodes_sit_on_the_axis_of_a_cylinder():
    """Mid-tube rings are true cross-sections, so their centroids are on-axis.

    The window excludes one radius at each end: there the wave is still a
    spherical cap around the seed (or is closing on the far cap) rather than a
    cross-section, so those centroids legitimately bulge off-axis - the one
    accuracy caveat of the method, documented on `wavefront_skeletonize`.
    """
    radius = 8
    vox = solid_cylinder(radius, 60)
    sk = wavefront_skeletonize(vox)
    axis = np.array([float(radius), float(radius)])
    z = sk.nodes[:, 2]
    mid = (z > 2 * radius) & (z < 59 - 2 * radius)
    assert mid.sum() > 5
    assert np.linalg.norm(sk.nodes[mid][:, :2] - axis, axis=1).max() < 0.2


def test_spacing_scales_vertices_not_topology():
    vox = solid_cylinder(4, 20)
    plain = wavefront_skeletonize(vox)
    scaled = wavefront_skeletonize(vox, spacing=[2, 2, 2])
    assert len(scaled.nodes) == len(plain.nodes)
    # `spacing` doubles the level width too, so nodes land at the same places.
    assert np.allclose(scaled.vertices, plain.vertices * 2)


def test_step_size_controls_node_spacing():
    vox = solid_cylinder(4, 40)
    coarse = wavefront_skeletonize(vox, step_size=6.0)
    fine = wavefront_skeletonize(vox)
    assert len(coarse.nodes) < len(fine.nodes)


def test_min_branch_length_prunes_and_keeps_radii_aligned():
    vox = self_touch_hairpin()
    sk = wavefront_skeletonize(vox, min_branch_length=5)
    full = wavefront_skeletonize(vox)
    assert len(sk.nodes) <= len(full.nodes)
    assert len(sk.radii) == len(sk.nodes)


@pytest.mark.parametrize("name", list(SHAPES))
def test_n_voxels_partitions_the_object(name):
    """Every voxel collapses onto exactly one node - so the counts must sum to N.

    This is the property that makes `n_voxels` (and the volume radius) meaningful,
    and the one that silently breaks if a ring is ever dropped or double-counted.
    """
    vox = np.unique(SHAPES[name], axis=0)
    sk = wavefront_skeletonize(vox)
    assert sk.n_voxels is not None
    assert len(sk.n_voxels) == len(sk.nodes)
    assert sk.n_voxels.sum() == len(vox)
    assert (sk.n_voxels > 0).all()  # an empty ring cannot produce a node


def test_volume_property_scales_with_spacing():
    vox = solid_cylinder(4, 20)
    plain = wavefront_skeletonize(vox)
    assert np.array_equal(plain.volume, plain.n_voxels.astype(float))
    assert plain.volume.sum() == len(np.unique(vox, axis=0))

    scaled = wavefront_skeletonize(vox, spacing=[2, 2, 3])
    # Same convention as `measure.volume`: count times the volume of one cell.
    assert np.allclose(scaled.volume, scaled.n_voxels * 12.0)


def test_volume_is_none_for_the_other_backends():
    """Their nodes are picked *from* the voxels, so there is no territory."""
    vox = solid_cylinder(4, 20)
    assert sc.thin_skeletonize(vox).n_voxels is None
    assert sc.thin_skeletonize(vox).volume is None
    assert sc.teasar_skeletonize(vox).n_voxels is None


def test_volume_radius_beats_spread_on_a_known_cylinder():
    """The volume radius is the only one without a ring-shape bias.

    A filled disc has mean spread 2/3 R, so `"mean"` reads well under the true
    radius while `"volume"` solves for it directly. Measured mid-tube, away from
    the end caps where the wave is not yet a cross-section.
    """
    vox = solid_cylinder(8, 60)
    mid = lambda sk: (sk.nodes[:, 2] > 16) & (sk.nodes[:, 2] < 43)

    vol = wavefront_skeletonize(vox, radius_agg="volume")
    assert np.median(vol.radii[mid(vol)]) == pytest.approx(8.0, abs=0.4)

    avg = wavefront_skeletonize(vox, radius_agg="mean")
    assert np.median(avg.radii[mid(avg)]) < 6.5  # the 2/3 R shape bias


def test_volume_radius_sweeps_back_to_the_object_volume():
    """pi r^2 L summed over the skeleton should recover the object's volume."""
    vox = solid_cylinder(8, 60)
    sk = wavefront_skeletonize(vox, radius_agg="volume")
    seg = np.linalg.norm(
        sk.vertices[sk.edges[:, 1]] - sk.vertices[sk.edges[:, 0]], axis=1
    )
    # Each edge is swept with the mean of its endpoints' cross-sections.
    area = np.pi * sk.radii**2
    swept = (0.5 * (area[sk.edges[:, 0]] + area[sk.edges[:, 1]]) * seg).sum()
    assert swept == pytest.approx(len(np.unique(vox, axis=0)), rel=0.1)


def test_isolated_node_falls_back_to_a_sphere():
    """A component with no edges has no tube to sweep, so use the equal sphere."""
    vox = solid_cube(6)
    sk = wavefront_skeletonize(vox, step_size=100.0)  # collapses to a single node
    assert len(sk.nodes) == 1 and len(sk.edges) == 0
    assert sk.radii[0] == pytest.approx(np.cbrt(3 * 216 / (4 * np.pi)), rel=1e-6)


def test_unknown_radius_agg_is_rejected():
    with pytest.raises(ValueError, match="radius_agg must be one of"):
        wavefront_skeletonize(solid_cylinder(4, 20), radius_agg="rms")


def test_radii_track_the_true_radius():
    """A cylinder's reported radius should be within a factor of the real one."""
    sk = wavefront_skeletonize(solid_cylinder(8, 60), radius_agg="max")
    mid = (sk.nodes[:, 2] > 12) & (sk.nodes[:, 2] < 48)
    assert 7.0 < np.median(sk.radii[mid]) < 10.0


def test_contract_matches_a_naive_grouping():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n_labels = int(rng.integers(2, 8))
        labels = rng.integers(0, n_labels, size=12)
        edges = rng.integers(0, 12, size=(int(rng.integers(1, 20)), 2))
        expected = sorted(
            {
                (min(labels[i], labels[j]), max(labels[i], labels[j]))
                for i, j in edges
                if labels[i] != labels[j]
            }
        )
        got = sorted(map(tuple, _contract(labels, edges, n_labels).tolist()))
        assert got == expected


@pytest.mark.parametrize("name", list(SHAPES))
def test_backends_agree(name, monkeypatch):
    """The `dijkstra3d_sparse` and scipy paths must produce identical skeletons.

    They compute the same three things by completely different routes - the
    former over the coordinates via `connected_components(group=)` and
    `label_adjacency`, the latter over an explicit edge list via scipy - so this
    is the check that keeps the fast path honest.
    """
    import sparsecubes.wavefront as W

    if W._dijkstra3d_sparse is None:
        pytest.skip("dijkstra3d_sparse not installed; only one backend available")

    vox = SHAPES[name]
    fast = wavefront_skeletonize(vox)
    monkeypatch.setattr(W, "_dijkstra3d_sparse", None)
    slow = wavefront_skeletonize(vox)

    assert np.allclose(fast.nodes, slow.nodes)
    assert np.array_equal(fast.edges, slow.edges)
    assert np.allclose(fast.radii, slow.radii)
    assert np.array_equal(fast.n_voxels, slow.n_voxels)


def test_matches_top_level_export():
    vox = solid_cylinder(4, 20)
    assert np.array_equal(
        sc.wavefront_skeletonize(vox).nodes, wavefront_skeletonize(vox).nodes
    )
