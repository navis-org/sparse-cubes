"""Tests for transparent 3-D `scipy.sparse` interop (`sparsecubes._sparse`).

Two contracts are pinned here:

1. Every voxel-taking function accepts a 3-D `coo_array` and returns exactly what
   it would have for the equivalent ``(N, 3)`` index array.
2. `sparsecubes._sparse` never imports scipy. Detection is duck-typing on the
   type's module name, and the module is only ever fetched out of `sys.modules`,
   which a sparse argument proves is already populated.
"""

import subprocess
import sys

import numpy as np
import pytest

import sparsecubes as sc
from sparsecubes import binary as sb, measure as sm
from sparsecubes._sparse import is_sparse, to_voxels, to_sparse_like

pytest.importorskip("scipy")
from scipy import sparse as spx  # noqa: E402

from _shapes import solid_cube, l_shape, y_branch, solid_cylinder  # noqa: E402


def _as_set(a):
    return set(map(tuple, np.asarray(a).tolist()))


def _coo(voxels, shape=None, dtype=bool):
    """Build a 3-D coo_array holding `voxels` (which must be non-negative)."""
    voxels = np.asarray(voxels)
    if shape is None:
        shape = tuple(int(v) + 1 for v in voxels.max(axis=0))
    data = np.ones(len(voxels), dtype=dtype)
    return spx.coo_array((data, tuple(voxels[:, i] for i in range(3))), shape=shape)


SHAPES = {
    "cube": solid_cube(5),
    "l_shape": l_shape(),
    "y_branch": y_branch(),
    "cylinder": solid_cylinder(),
}


# --- detection --------------------------------------------------------------


def test_detects_scipy_sparse():
    assert is_sparse(_coo(solid_cube(3)))


@pytest.mark.parametrize(
    "obj", [np.zeros((4, 3), dtype=np.int64), None, 42, "str", [1, 2, 3], {"a": 1}]
)
def test_does_not_misdetect_ordinary_objects(obj):
    assert is_sparse(obj) is False


def test_sparse_module_never_imports_scipy():
    """Loading `_sparse` and calling `is_sparse` must not pull scipy in.

    Run in a subprocess so the ambient (trimesh-triggered) scipy import in this
    test session cannot mask a regression.
    """
    code = (
        "import sys, importlib.util, numpy as np;"
        "spec = importlib.util.spec_from_file_location('s', 'sparsecubes/_sparse.py');"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "assert not m.is_sparse(np.zeros((2, 3)));"
        "assert not m.is_sparse(None);"
        "print(sum(1 for k in sys.modules if k.startswith('scipy')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "0", f"scipy got imported: {out.stdout}"


# --- round trip -------------------------------------------------------------


@pytest.mark.parametrize("name", list(SHAPES))
def test_to_voxels_round_trip(name):
    vox = SHAPES[name]
    assert _as_set(to_voxels(_coo(vox))) == _as_set(vox)


def test_to_voxels_drops_explicit_zeros():
    """Occupancy follows values, not the storage pattern."""
    coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    coo = spx.coo_array(
        (np.array([1.0, 0.0]), tuple(coords[:, i] for i in range(3))), shape=(2, 2, 2)
    )
    assert _as_set(to_voxels(coo)) == {(0, 0, 0)}


@pytest.mark.parametrize("dtype", [bool, np.int32, np.float64, np.uint8])
def test_to_voxels_dtype_agnostic(dtype):
    vox = solid_cube(3)
    assert _as_set(to_voxels(_coo(vox, dtype=dtype))) == _as_set(vox)


def test_rejects_2d_matrix():
    m = spx.coo_array(np.eye(4))
    with pytest.raises(TypeError, match="3-D sparse array"):
        sb.dilate(m)


# --- results match the ndarray path -----------------------------------------


@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize(
    "op", ["dilate", "erode", "opening", "closing", "thin", "fill_cavities"]
)
def test_binary_ops_match_ndarray_path(name, op):
    # Offset off index 0: a growing op would otherwise need negative indices,
    # which a sparse array cannot represent (see the bounds tests below).
    vox = SHAPES[name] + 2
    coo = _coo(vox, shape=tuple(int(v) + 4 for v in vox.max(axis=0)))
    want = getattr(sb, op)(vox)
    got = getattr(sb, op)(coo)
    assert is_sparse(got), f"{op} should mirror its sparse input"
    assert _as_set(to_voxels(got)) == _as_set(want)


@pytest.mark.parametrize(
    "op", ["union", "intersection", "difference", "symmetric_difference"]
)
def test_set_ops_match_ndarray_path(op):
    a, b = solid_cube(4), solid_cube(4) + np.array([2, 0, 0])
    want = getattr(sb, op)(a, b)
    got = getattr(sb, op)(_coo(a, shape=(8, 8, 8)), _coo(b, shape=(8, 8, 8)))
    assert is_sparse(got)
    assert _as_set(to_voxels(got)) == _as_set(want)


def test_set_ops_accept_mixed_sparse_and_dense():
    """One sparse argument is enough to get a sparse result."""
    a, b = solid_cube(4), solid_cube(4) + np.array([2, 0, 0])
    got = sb.union(_coo(a, shape=(8, 8, 8)), b)
    assert is_sparse(got)
    assert _as_set(to_voxels(got)) == _as_set(sb.union(a, b))


@pytest.mark.parametrize("name", list(SHAPES))
def test_measure_returns_plain_values(name):
    """Measurements have no sparse form - they must come back as plain values."""
    vox = SHAPES[name]
    coo = _coo(vox)
    assert sm.volume(coo) == sm.volume(vox)
    assert sm.surface_area(coo) == sm.surface_area(vox)
    assert np.array_equal(sm.bounding_box(coo), sm.bounding_box(vox))
    assert np.allclose(sm.centroid(coo), sm.centroid(vox))
    n_got, labels = sm.connected_components(coo)
    assert n_got == sm.connected_components(vox)[0]
    assert isinstance(labels, np.ndarray)


@pytest.mark.parametrize("op", ["largest_component", "remove_small_objects"])
def test_measure_voxel_returning_ops_mirror(op):
    big = solid_cube(4)
    speck = np.array([[20, 20, 20]], dtype=np.int64)
    vox = np.vstack([big, speck])
    args = (2,) if op == "remove_small_objects" else ()
    got = getattr(sm, op)(_coo(vox), *args)
    assert is_sparse(got)
    assert _as_set(to_voxels(got)) == _as_set(big)


def test_isin_returns_a_plain_mask():
    a = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    b = np.array([[1, 0, 0]], dtype=np.int64)
    mask = sb.isin(_coo(a, shape=(4, 4, 4)), _coo(b, shape=(4, 4, 4)))
    assert isinstance(mask, np.ndarray) and mask.dtype == bool
    assert list(mask) == [False, True]


def test_mesh_accepts_sparse():
    vox = solid_cube(4)
    m_sparse, m_dense = sc.mesh(_coo(vox)), sc.mesh(vox)
    assert np.allclose(m_sparse.vertices, m_dense.vertices)
    assert np.array_equal(m_sparse.faces, m_dense.faces)


def test_skeletonizers_accept_sparse():
    vox = solid_cylinder()
    coo = _coo(vox)
    for fn in (sc.thin_skeletonize, sc.teasar_skeletonize):
        a, b = fn(coo), fn(vox)
        assert np.array_equal(a.nodes, b.nodes)
        assert np.array_equal(a.edges, b.edges)


def test_downsample_graph_accepts_sparse():
    vox = solid_cube(6)
    c_sparse, e_sparse = sc.downsample_graph(_coo(vox), 2)
    c_dense, e_dense = sc.downsample_graph(vox, 2)
    assert np.array_equal(c_sparse, c_dense)  # coarse frame: stays an ndarray
    assert np.array_equal(e_sparse, e_dense)


# --- output shape semantics -------------------------------------------------


def test_shape_is_preserved_when_the_result_fits():
    vox = solid_cube(3)
    out = sb.erode(_coo(vox, shape=(10, 10, 10)))
    assert out.shape == (10, 10, 10)


def test_shape_grows_when_the_result_outgrows_it():
    """The template shape is a floor, not a clamp - dilation must not truncate."""
    vox = solid_cube(3) + 1  # occupies 1..3, clear of index 0
    out = sb.dilate(_coo(vox, shape=(4, 4, 4)))
    assert out.shape == (5, 5, 5)
    assert _as_set(to_voxels(out)) == _as_set(sb.dilate(vox))


def test_negative_result_coordinates_raise():
    """Growing an object that touches index 0 cannot be represented."""
    at_origin = np.array([[0, 0, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="negative coordinates"):
        sb.dilate(_coo(at_origin, shape=(4, 4, 4)))


def test_output_dtype_follows_the_template():
    for dtype in (bool, np.int32, np.float64):
        out = sb.erode(_coo(solid_cube(4), dtype=dtype))
        assert out.dtype == dtype


def test_empty_result_keeps_the_template_shape():
    out = sb.erode(_coo(np.array([[1, 1, 1]], dtype=np.int64), shape=(6, 6, 6)))
    assert is_sparse(out)
    assert out.shape == (6, 6, 6)
    assert to_voxels(out).shape == (0, 3)


def test_to_sparse_like_is_a_faithful_round_trip():
    vox = solid_cube(4) + 2
    template = _coo(vox)
    assert _as_set(to_voxels(to_sparse_like(vox, template))) == _as_set(vox)


# --- the ndarray path is untouched ------------------------------------------


@pytest.mark.parametrize("name", list(SHAPES))
def test_ndarray_in_ndarray_out(name):
    """Decorating must not change behaviour for ordinary array callers."""
    vox = SHAPES[name]
    for fn in (sb.dilate, sb.erode, sb.thin, sm.largest_component):
        out = fn(vox)
        assert isinstance(out, np.ndarray) and not is_sparse(out)
