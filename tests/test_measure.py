"""Tests for `sparsecubes.measure` - labelling and measurements.

Component labelling and the distance transform are cross-checked against dense
`scipy.ndimage` oracles: the sparse result must agree exactly with what you would
get by densifying the bounding box.
"""

import importlib.util

import numpy as np
import pytest

from sparsecubes import measure as sm
from _shapes import solid_cube, hollow_box, l_shape, line, annulus, slab

_HAS_SCIPY = importlib.util.find_spec("scipy") is not None
_HAS_CC = _HAS_SCIPY or importlib.util.find_spec("dijkstra3d_sparse") is not None
needs_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="needs scipy")
needs_cc = pytest.mark.skipif(not _HAS_CC, reason="needs scipy or dijkstra3d_sparse")


def _as_set(a):
    return set(map(tuple, np.asarray(a).tolist()))


def _dense(voxels, pad=1):
    lo = voxels.min(axis=0) - pad
    shape = tuple(voxels.max(axis=0) - lo + pad + 1)
    grid = np.zeros(shape, dtype=bool)
    grid[tuple((voxels - lo).T)] = True
    return grid, lo


def _two_blobs(gap=5):
    """Two solid cubes separated by `gap` empty voxels along x."""
    a = solid_cube(3)
    b = solid_cube(3) + np.array([3 + gap, 0, 0])
    return np.vstack([a, b]).astype(np.int64)


# --- connected components ---------------------------------------------------


@needs_cc
@pytest.mark.parametrize("connectivity", [6, 26])
@pytest.mark.parametrize(
    "shape", [solid_cube(4), hollow_box(), l_shape(), annulus(), _two_blobs()]
)
def test_component_count_matches_scipy(shape, connectivity):
    from scipy import ndimage

    grid, _ = _dense(shape)
    rank = {6: 1, 26: 3}[connectivity]
    _, want = ndimage.label(grid, structure=ndimage.generate_binary_structure(3, rank))
    got, _ = sm.connected_components(shape, connectivity=connectivity)
    assert got == want


@needs_cc
def test_two_blobs_are_two_components():
    n, labels = sm.connected_components(_two_blobs())
    assert n == 2
    assert len(labels) == len(_two_blobs())
    assert set(np.bincount(labels)) == {27}


@needs_cc
def test_labels_align_with_input_rows():
    """Labels must follow the caller's row order, not an internal sorted order."""
    vox = _two_blobs()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(vox))
    shuffled = vox[perm]
    _, labels = sm.connected_components(shuffled)
    # Every voxel of the left cube shares a label, distinct from the right cube's.
    left = shuffled[:, 0] < 4
    assert len(set(labels[left])) == 1
    assert len(set(labels[~left])) == 1
    assert set(labels[left]) != set(labels[~left])


@needs_cc
def test_duplicate_rows_share_a_label():
    vox = np.array([[0, 0, 0], [0, 0, 0], [9, 9, 9]], dtype=np.int64)
    n, labels = sm.connected_components(vox)
    assert n == 2
    assert labels[0] == labels[1] != labels[2]


@needs_cc
def test_connectivity_matters_for_corner_touching():
    """Two corner-touching voxels: one component under 26, two under 6."""
    vox = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    assert sm.connected_components(vox, connectivity=26)[0] == 1
    assert sm.connected_components(vox, connectivity=6)[0] == 2


@needs_cc
def test_connectivity_18_rejected():
    with pytest.raises(ValueError, match="connectivity"):
        sm.connected_components(solid_cube(3), connectivity=18)


@needs_cc
def test_largest_component():
    big = solid_cube(4)
    small = np.array([[100, 100, 100]], dtype=np.int64)
    out = sm.largest_component(np.vstack([big, small]))
    assert _as_set(out) == _as_set(big)


@needs_cc
def test_remove_small_objects():
    big = solid_cube(4)  # 64 voxels
    speck = np.array([[100, 100, 100]], dtype=np.int64)
    vox = np.vstack([big, speck])
    assert _as_set(sm.remove_small_objects(vox, 2)) == _as_set(big)
    assert _as_set(sm.remove_small_objects(vox, 1)) == _as_set(vox)


@needs_cc
def test_remove_small_objects_boundary_is_inclusive():
    """A component of exactly `min_size` is kept."""
    speck = np.array([[0, 0, 0]], dtype=np.int64)
    assert len(sm.remove_small_objects(speck, 1)) == 1
    assert len(sm.remove_small_objects(speck, 2)) == 0


@needs_cc
@pytest.mark.parametrize("fn", [sm.largest_component, sm.connected_components])
def test_components_on_empty(fn):
    out = fn(np.empty((0, 3), dtype=np.int64))
    if isinstance(out, tuple):
        assert out[0] == 0
    else:
        assert out.shape == (0, 3)


# --- volume / surface area --------------------------------------------------


def test_volume_counts_unique_voxels():
    assert sm.volume(solid_cube(4)) == 64.0
    dup = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int64)
    assert sm.volume(dup) == 1.0


def test_volume_with_spacing():
    assert sm.volume(solid_cube(4), spacing=2.0) == 64.0 * 8
    assert sm.volume(solid_cube(4), spacing=(1, 2, 3)) == 64.0 * 6


def test_surface_area_of_a_cube():
    """A k x k x k solid block has 6 * k^2 exposed unit faces."""
    for k in (1, 3, 5):
        assert sm.surface_area(solid_cube(k)) == 6.0 * k * k


def test_surface_area_matches_face_count_oracle():
    vox = l_shape()
    occupied = _as_set(vox)
    want = sum(
        (tuple(np.array(v) + d) not in occupied)
        for v in occupied
        for d in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    )
    assert sm.surface_area(vox) == float(want)


def test_surface_area_with_anisotropic_spacing():
    """A unit voxel with spacing (1,2,3): faces are 2*6 + 2*3 + 2*2 = 22."""
    one = np.zeros((1, 3), dtype=np.int64)
    assert sm.surface_area(one, spacing=(1, 2, 3)) == pytest.approx(22.0)


def test_volume_and_area_on_empty():
    empty = np.empty((0, 3), dtype=np.int64)
    assert sm.volume(empty) == 0.0
    assert sm.surface_area(empty) == 0.0


# --- bounding box / centroid ------------------------------------------------


def test_bounding_box():
    bb = sm.bounding_box(solid_cube(4))
    assert bb.tolist() == [[0, 0, 0], [3, 3, 3]]


def test_bounding_box_with_negatives():
    vox = np.array([[-3, 0, 2], [5, -7, 2]], dtype=np.int64)
    assert sm.bounding_box(vox).tolist() == [[-3, -7, 2], [5, 0, 2]]


def test_centroid():
    assert sm.centroid(solid_cube(4)).tolist() == [1.5, 1.5, 1.5]
    assert sm.centroid(solid_cube(4), spacing=2).tolist() == [3.0, 3.0, 3.0]


def test_centroid_ignores_duplicates():
    vox = np.array([[0, 0, 0], [0, 0, 0], [2, 0, 0]], dtype=np.int64)
    assert sm.centroid(vox)[0] == 1.0


@pytest.mark.parametrize("fn", [sm.bounding_box, sm.centroid])
def test_bbox_centroid_raise_on_empty(fn):
    with pytest.raises(ValueError, match="empty"):
        fn(np.empty((0, 3), dtype=np.int64))


# --- distance transform -----------------------------------------------------


@needs_scipy
@pytest.mark.parametrize(
    "shape", [solid_cube(7), hollow_box(), l_shape(), slab(5), annulus(), line(6)]
)
def test_distance_transform_matches_dense_edt(shape):
    """Must equal a true dense EDT - the sparse shell shortcut is not an approximation."""
    from scipy import ndimage

    grid, lo = _dense(shape, pad=2)
    dense = ndimage.distance_transform_edt(grid)
    want = dense[tuple((shape - lo).T)]
    got = sm.distance_transform(shape)
    assert np.allclose(got, want)


@needs_scipy
def test_distance_transform_anisotropic_matches_dense():
    from scipy import ndimage

    spacing = (1.0, 2.0, 3.0)
    shape = solid_cube(7)
    grid, lo = _dense(shape, pad=3)
    dense = ndimage.distance_transform_edt(grid, sampling=spacing)
    want = dense[tuple((shape - lo).T)]
    got = sm.distance_transform(shape, spacing=spacing)
    assert np.allclose(got, want)


@needs_scipy
def test_distance_transform_is_row_aligned_and_positive():
    vox = solid_cube(5)
    d = sm.distance_transform(vox)
    assert d.shape == (len(vox),)
    assert (d > 0).all()  # even surface voxels sit ~1 from the background


@needs_scipy
def test_distance_transform_on_empty():
    assert len(sm.distance_transform(np.empty((0, 3), dtype=np.int64))) == 0


# --- input validation -------------------------------------------------------


@pytest.mark.parametrize(
    "fn", [sm.volume, sm.surface_area, sm.bounding_box, sm.centroid]
)
def test_rejects_non_integer(fn):
    with pytest.raises(TypeError, match="integer"):
        fn(solid_cube(3).astype(float))


@pytest.mark.parametrize("bad", [0, -1, (1, 2), (1, 2, 3, 4)])
def test_bad_spacing_raises(bad):
    with pytest.raises(ValueError, match="spacing"):
        sm.volume(solid_cube(3), spacing=bad)


@needs_scipy
@pytest.mark.parametrize("workers", [1, 2, -1])
def test_distance_transform_workers_is_speed_only(workers):
    """`workers` threads the KD-tree query; distances must be unchanged."""
    vox = solid_cube(7)
    assert np.allclose(
        sm.distance_transform(vox, workers=workers), sm.distance_transform(vox, workers=1)
    )
