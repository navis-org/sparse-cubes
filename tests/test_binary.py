"""Tests for `sparsecubes.binary` - morphology and set algebra.

The morphology is cross-checked against `scipy.ndimage` on a dense grid: the
sparse implementation must agree voxel-for-voxel with the dense oracle, which is
the whole contract (same answer, without allocating the volume).
"""

import importlib.util

import numpy as np
import pytest

import sparsecubes as sc
from sparsecubes import binary as sb
from _shapes import solid_cube, hollow_box, l_shape, line, annulus

_HAS_SCIPY = importlib.util.find_spec("scipy") is not None
needs_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="needs scipy")

CONNECTIVITIES = [6, 18, 26]


def _as_set(a):
    return set(map(tuple, np.asarray(a).tolist()))


def _structure(connectivity):
    """scipy.ndimage structuring element matching our `connectivity`."""
    from scipy import ndimage

    rank = {6: 1, 18: 2, 26: 3}[connectivity]
    return ndimage.generate_binary_structure(3, rank)


def _dense(voxels, pad):
    """Dense bool grid of `voxels` with `pad` empty cells on every side."""
    lo = voxels.min(axis=0) - pad
    shape = tuple(voxels.max(axis=0) - lo + pad + 1)
    grid = np.zeros(shape, dtype=bool)
    idx = (voxels - lo).T
    grid[tuple(idx)] = True
    return grid, lo


def _from_dense(grid, lo):
    return np.stack(np.nonzero(grid), axis=1) + lo


# --- morphology vs. the dense oracle ---------------------------------------


@needs_scipy
@pytest.mark.parametrize("connectivity", CONNECTIVITIES)
@pytest.mark.parametrize("iterations", [1, 2, 3])
@pytest.mark.parametrize(
    "shape", [solid_cube(4), hollow_box(), l_shape(), line(6), annulus()]
)
def test_dilate_matches_scipy(shape, iterations, connectivity):
    from scipy import ndimage

    pad = iterations + 1
    grid, lo = _dense(shape, pad)
    want = ndimage.binary_dilation(
        grid, structure=_structure(connectivity), iterations=iterations
    )
    got = sb.dilate(shape, iterations=iterations, connectivity=connectivity)
    assert _as_set(got) == _as_set(_from_dense(want, lo))


@needs_scipy
@pytest.mark.parametrize("connectivity", CONNECTIVITIES)
@pytest.mark.parametrize("iterations", [1, 2])
@pytest.mark.parametrize("shape", [solid_cube(6), hollow_box(), l_shape(), annulus()])
def test_erode_matches_scipy(shape, iterations, connectivity):
    from scipy import ndimage

    grid, lo = _dense(shape, iterations + 1)
    want = ndimage.binary_erosion(
        grid, structure=_structure(connectivity), iterations=iterations,
        border_value=0,
    )
    got = sb.erode(shape, iterations=iterations, connectivity=connectivity)
    assert _as_set(got) == _as_set(_from_dense(want, lo))


@needs_scipy
@pytest.mark.parametrize("connectivity", CONNECTIVITIES)
@pytest.mark.parametrize("op,scipy_op", [("opening", "binary_opening"),
                                         ("closing", "binary_closing")])
@pytest.mark.parametrize("shape", [solid_cube(6), hollow_box(), l_shape()])
def test_opening_closing_match_scipy(shape, op, scipy_op, connectivity):
    from scipy import ndimage

    # Pad generously: closing dilates first, and scipy needs room for it too.
    grid, lo = _dense(shape, 4)
    want = getattr(ndimage, scipy_op)(grid, structure=_structure(connectivity))
    got = getattr(sb, op)(shape, connectivity=connectivity)
    assert _as_set(got) == _as_set(_from_dense(want, lo))


# --- morphology: properties -------------------------------------------------


def test_dilate_is_extensive_erode_is_antiextensive():
    cube = solid_cube(5)
    assert _as_set(cube) <= _as_set(sb.dilate(cube))
    assert _as_set(sb.erode(cube)) <= _as_set(cube)


def test_erode_can_empty_a_thin_object():
    assert len(sb.erode(line(8))) == 0


def test_zero_iterations_is_identity():
    cube = solid_cube(4)
    assert _as_set(sb.dilate(cube, iterations=0)) == _as_set(cube)
    assert _as_set(sb.erode(cube, iterations=0)) == _as_set(cube)


def test_dilate_erode_roundtrip_on_solid_cube():
    """Dilating then eroding a solid convex block returns it unchanged."""
    cube = solid_cube(6)
    assert _as_set(sb.erode(sb.dilate(cube))) == _as_set(cube)


def test_closing_fills_a_one_voxel_pit():
    cube = solid_cube(5)
    pit = cube[~np.all(cube == [0, 0, 0], axis=1)]  # knock out a corner voxel
    assert _as_set(sb.closing(pit)) >= _as_set(pit)


@pytest.mark.parametrize("bad", [0, 5, 7, 27, -1])
def test_bad_connectivity_raises(bad):
    with pytest.raises(ValueError, match="connectivity"):
        sb.dilate(solid_cube(3), connectivity=bad)


def test_negative_iterations_raises():
    with pytest.raises(ValueError, match="iterations"):
        sb.erode(solid_cube(3), iterations=-1)


# --- set algebra ------------------------------------------------------------


@pytest.fixture
def ab():
    a = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int64)
    b = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.int64)
    return a, b


def test_union(ab):
    a, b = ab
    assert _as_set(sb.union(a, b)) == _as_set(a) | _as_set(b)


def test_intersection(ab):
    a, b = ab
    assert _as_set(sb.intersection(a, b)) == _as_set(a) & _as_set(b)


def test_difference(ab):
    a, b = ab
    assert _as_set(sb.difference(a, b)) == _as_set(a) - _as_set(b)
    assert _as_set(sb.difference(b, a)) == _as_set(b) - _as_set(a)


def test_symmetric_difference(ab):
    a, b = ab
    assert _as_set(sb.symmetric_difference(a, b)) == _as_set(a) ^ _as_set(b)


def test_isin_preserves_row_order_and_length(ab):
    a, b = ab
    mask = sb.isin(a, b)
    assert mask.shape == (len(a),)
    assert list(mask) == [False, True, True]


def test_isin_handles_duplicate_rows(ab):
    _, b = ab
    dup = np.array([[1, 0, 0], [9, 9, 9], [1, 0, 0]], dtype=np.int64)
    assert list(sb.isin(dup, b)) == [True, False, True]


def test_set_ops_dedup_their_inputs():
    dup = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.int64)
    assert len(sb.union(dup, dup)) == 2
    assert len(sb.intersection(dup, dup)) == 2


def test_variadic_set_ops():
    a = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    b = np.array([[1, 0, 0], [2, 0, 0]], dtype=np.int64)
    c = np.array([[1, 0, 0], [3, 0, 0]], dtype=np.int64)
    assert _as_set(sb.intersection(a, b, c)) == {(1, 0, 0)}
    assert len(sb.union(a, b, c)) == 4


def test_set_ops_on_far_apart_clouds():
    """A shared origin must not mangle sets that live far from each other."""
    a = np.array([[0, 0, 0]], dtype=np.int64)
    b = np.array([[100000, 5, 7]], dtype=np.int64)
    assert len(sb.union(a, b)) == 2
    assert len(sb.intersection(a, b)) == 0
    assert _as_set(sb.difference(a, b)) == _as_set(a)


def test_set_ops_with_negative_coordinates():
    a = np.array([[-5, -5, -5], [0, 0, 0]], dtype=np.int64)
    b = np.array([[-5, -5, -5]], dtype=np.int64)
    assert _as_set(sb.difference(a, b)) == {(0, 0, 0)}
    assert _as_set(sb.intersection(a, b)) == {(-5, -5, -5)}


# --- empty input / dtype ----------------------------------------------------


EMPTY = np.empty((0, 3), dtype=np.int64)


@pytest.mark.parametrize("fn", [sb.dilate, sb.erode, sb.opening, sb.closing])
def test_morphology_on_empty(fn):
    out = fn(EMPTY)
    assert out.shape == (0, 3)


def test_set_ops_on_empty():
    a = np.array([[1, 2, 3]], dtype=np.int64)
    assert _as_set(sb.union(a, EMPTY)) == _as_set(a)
    assert len(sb.intersection(a, EMPTY)) == 0
    assert _as_set(sb.difference(a, EMPTY)) == _as_set(a)
    assert len(sb.difference(EMPTY, a)) == 0
    assert len(sb.isin(EMPTY, a)) == 0


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64, np.uint32])
def test_dtype_preserved(dtype):
    cube = solid_cube(4).astype(dtype)
    assert sb.dilate(cube).dtype == dtype
    assert sb.erode(cube).dtype == dtype
    assert sb.union(cube, cube).dtype == dtype


def test_rejects_non_integer():
    with pytest.raises(TypeError, match="integer"):
        sb.dilate(solid_cube(3).astype(float))


def test_rejects_wrong_shape():
    with pytest.raises(TypeError, match=r"\(N, 3\)"):
        sb.dilate(np.zeros((5, 2), dtype=np.int64))


# --- relocation from the top level ------------------------------------------


def test_thin_and_fill_cavities_are_reachable_here():
    assert sb.thin is not None and sb.fill_cavities is not None


@pytest.mark.parametrize("name", ["thin", "fill_cavities"])
def test_old_toplevel_spelling_explains_the_move(name):
    with pytest.raises(AttributeError, match="moved to"):
        getattr(sc, name)
