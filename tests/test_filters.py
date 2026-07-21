"""Tests for `sparsecubes.filters` - value-domain filtering.

The contract for `smooth` is exactness, not approximation: run on a densified
bounding box it must reproduce `scipy.ndimage.gaussian_filter(..., mode="constant",
cval=0)` to floating-point round-off. That oracle is the bulk of this file, since
it pins the kernel, the truncation rounding and the zero-outside boundary
semantics all at once.
"""

import importlib.util

import numpy as np
import pytest

import sparsecubes as sc
from sparsecubes import filters as sf
from _shapes import solid_cube, hollow_box, l_shape, line, annulus, solid_cylinder

_HAS_SCIPY = importlib.util.find_spec("scipy") is not None
needs_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="needs scipy")

SHAPES = {
    "cube": solid_cube(4),
    "line": line(8),
    "l_shape": l_shape(),
    "annulus": annulus(9, 5),
    "hollow_box": hollow_box(7, 3),
    "cylinder": solid_cylinder(3, 10),
}


def _dense_oracle(voxels, values, sigma, truncate=4.0, radius=None):
    """`scipy.ndimage.gaussian_filter` on a box padded past the kernel reach.

    With enough padding, ``mode="constant", cval=0`` on the padded box *is* the
    unbounded-domain answer, so this is a fair reference and not an
    approximation of one.
    """
    from scipy import ndimage

    r = radius if radius is not None else np.max(truncate * np.atleast_1d(sigma) + 0.5)
    pad = int(np.max(r)) + 2
    lo = voxels.min(axis=0) - pad
    grid = np.zeros(tuple(voxels.max(axis=0) - lo + pad + 1), dtype=float)
    np.add.at(grid, tuple((voxels - lo).T), values)
    kwargs = {} if radius is None else {"radius": radius}
    out = ndimage.gaussian_filter(
        grid, sigma, mode="constant", cval=0.0, truncate=truncate, **kwargs
    )
    return out, lo


def _as_grid(voxels, values, shape, lo):
    """Scatter a sparse result back onto the oracle's grid for comparison."""
    grid = np.zeros(shape, dtype=float)
    idx = voxels - lo
    assert (idx >= 0).all() and (idx < np.array(shape)).all(), "result left the box"
    grid[tuple(idx.T)] = values
    return grid


# --- exactness against the dense oracle ------------------------------------


@needs_scipy
@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_matches_scipy_on_a_binary_mask(name, sigma):
    voxels = SHAPES[name].astype(np.int64)
    expected, lo = _dense_oracle(voxels, np.ones(len(voxels)), sigma)
    out_v, out_val = sc.filters.smooth(voxels, sigma=sigma)
    assert np.allclose(_as_grid(out_v, out_val, expected.shape, lo), expected, atol=1e-14)


@needs_scipy
@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_matches_scipy_with_values(sigma):
    rng = np.random.RandomState(0)
    voxels = np.unique(rng.randint(0, 9, (80, 3)), axis=0).astype(np.int64)
    values = rng.rand(len(voxels))
    expected, lo = _dense_oracle(voxels, values, sigma)
    out_v, out_val = sc.filters.smooth(voxels, values=values, sigma=sigma)
    assert np.allclose(_as_grid(out_v, out_val, expected.shape, lo), expected, atol=1e-14)


@needs_scipy
@pytest.mark.parametrize("sigma", [(0.5, 1.0, 2.0), (2.0, 0.5, 1.0), (1.0, 1.0, 0.0)])
def test_matches_scipy_anisotropic(sigma):
    """Per-axis sigma, including an axis switched off entirely."""
    rng = np.random.RandomState(1)
    voxels = np.unique(rng.randint(0, 7, (60, 3)), axis=0).astype(np.int64)
    values = rng.rand(len(voxels))
    expected, lo = _dense_oracle(voxels, values, sigma)
    out_v, out_val = sc.filters.smooth(voxels, values=values, sigma=sigma)
    assert np.allclose(_as_grid(out_v, out_val, expected.shape, lo), expected, atol=1e-14)


@needs_scipy
@pytest.mark.parametrize("truncate", [1.0, 2.0, 3.0, 4.0])
def test_matches_scipy_across_truncate(truncate):
    """`truncate` sets the radius via scipy's exact rounding - pin that too."""
    voxels = solid_cube(4).astype(np.int64)
    values = np.ones(len(voxels))
    expected, lo = _dense_oracle(voxels, values, 1.0, truncate=truncate)
    out_v, out_val = sc.filters.smooth(voxels, sigma=1.0, truncate=truncate)
    assert np.allclose(_as_grid(out_v, out_val, expected.shape, lo), expected, atol=1e-14)


@needs_scipy
@pytest.mark.parametrize("radius", [0, 1, 3])
def test_matches_scipy_with_explicit_radius(radius):
    voxels = solid_cube(3).astype(np.int64)
    values = np.ones(len(voxels))
    expected, lo = _dense_oracle(voxels, values, 1.0, radius=radius)
    out_v, out_val = sc.filters.smooth(voxels, sigma=1.0, radius=radius)
    assert np.allclose(_as_grid(out_v, out_val, expected.shape, lo), expected, atol=1e-14)


@needs_scipy
def test_matches_scipy_with_negative_coordinates():
    voxels = (solid_cube(3) - 40).astype(np.int64)
    values = np.arange(len(voxels), dtype=float)
    expected, lo = _dense_oracle(voxels, values, 1.0)
    out_v, out_val = sc.filters.smooth(voxels, values=values, sigma=1.0)
    assert np.allclose(_as_grid(out_v, out_val, expected.shape, lo), expected, atol=1e-14)


@needs_scipy
def test_kernel_matches_scipys():
    """Our 1-D taps must be scipy's, or nothing above could hold."""
    from scipy.ndimage._filters import _gaussian_kernel1d

    for sigma in (0.5, 1.0, 2.5):
        for radius in (1, 4, 9):
            assert np.allclose(
                sf.gaussian_kernel1d(sigma, radius),
                _gaussian_kernel1d(sigma, 0, radius),
                atol=0,
                rtol=1e-15,
            )


# --- properties that hold without scipy ------------------------------------


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_mass_is_conserved(name, sigma):
    """A normalised kernel redistributes the total, it does not create or lose it."""
    voxels = SHAPES[name].astype(np.int64)
    values = np.arange(len(voxels), dtype=float) + 1.0
    _, out_val = sc.filters.smooth(voxels, values=values, sigma=sigma)
    assert out_val.sum() == pytest.approx(values.sum(), rel=1e-12)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_support_is_the_input_dilated_by_the_radius(sigma):
    voxels = solid_cube(4).astype(np.int64)
    r = int(4.0 * sigma + 0.5)
    out_v, _ = sc.filters.smooth(voxels, sigma=sigma)
    assert out_v.min(axis=0).tolist() == (voxels.min(axis=0) - r).tolist()
    assert out_v.max(axis=0).tolist() == (voxels.max(axis=0) + r).tolist()


def test_output_is_sorted_and_unique():
    out_v, out_val = sc.filters.smooth(l_shape().astype(np.int64), sigma=1.0)
    assert len(np.unique(out_v, axis=0)) == len(out_v)
    assert np.array_equal(out_v, np.unique(out_v, axis=0))
    assert len(out_val) == len(out_v)


def test_smoothing_is_non_negative_for_non_negative_input():
    out_v, out_val = sc.filters.smooth(l_shape().astype(np.int64), sigma=1.5)
    assert (out_val > 0).all()


def test_separability_equals_sequential_axis_passes():
    """Smoothing all axes at once == smoothing them one at a time."""
    voxels = solid_cube(3).astype(np.int64)
    both = sc.filters.smooth(voxels, sigma=1.0)

    v, val = sc.filters.smooth(voxels, sigma=(1.0, 0.0, 0.0))
    v, val = sc.filters.smooth(v, values=val, sigma=(0.0, 1.0, 0.0))
    v, val = sc.filters.smooth(v, values=val, sigma=(0.0, 0.0, 1.0))

    assert np.array_equal(both[0], v)
    assert np.allclose(both[1], val, atol=1e-14)


def test_sigma_zero_is_the_identity():
    voxels = np.unique(l_shape(), axis=0).astype(np.int64)
    values = np.arange(len(voxels), dtype=float) + 1.0  # no explicit zeros
    out_v, out_val = sc.filters.smooth(voxels, values=values, sigma=0.0)
    assert np.array_equal(out_v, voxels)
    assert np.allclose(out_val, values)


# --- the absent-means-zero convention ---------------------------------------


@pytest.mark.parametrize("sigma", [0.0, 1.0])
def test_zero_valued_input_voxels_are_dropped(sigma):
    """An explicit 0 is indistinguishable from absence, so it is canonicalised away.

    Uniformly, including for the sigma=0 no-op - otherwise the identity filter
    would preserve zeros that every other filter discards.
    """
    voxels = np.array([[0, 0, 0], [5, 5, 5]], dtype=np.int64)
    values = np.array([0.0, 2.0])
    out_v, out_val = sc.filters.smooth(voxels, values=values, sigma=sigma)
    assert not (out_val == 0).any()
    if sigma == 0.0:
        assert np.array_equal(out_v, [[5, 5, 5]])


def test_zero_valued_input_is_equivalent_to_absence():
    """Dropping the zero must change nothing, which is what justifies dropping it."""
    kept = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int64)
    with_zero = sc.filters.smooth(kept, values=np.array([0.0, 1.0, 2.0]), sigma=1.0)
    without = sc.filters.smooth(kept[1:], values=np.array([1.0, 2.0]), sigma=1.0)
    assert np.array_equal(with_zero[0], without[0])
    assert np.allclose(with_zero[1], without[1])


def test_output_never_contains_explicit_zeros():
    rng = np.random.RandomState(7)
    voxels = np.unique(rng.randint(0, 7, (50, 3)), axis=0).astype(np.int64)
    # An antisymmetric kernel produces exact cancellations.
    k = np.array([1.0, 0.0, -1.0])
    _, val = sc.filters.correlate(voxels, [k, k, k], values=rng.rand(len(voxels)))
    assert not (val == 0).any()


def test_duplicate_input_rows_are_summed():
    """Reading the input as a sparse array means duplicates add, not overwrite."""
    voxels = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int64)
    _, out_val = sc.filters.smooth(voxels, values=np.array([1.0, 2.0]), sigma=1.0)
    assert out_val.sum() == pytest.approx(3.0)


def test_multi_column_values():
    rng = np.random.RandomState(2)
    voxels = np.unique(rng.randint(0, 6, (40, 3)), axis=0).astype(np.int64)
    values = rng.rand(len(voxels), 3)
    out_v, out_val = sc.filters.smooth(voxels, values=values, sigma=1.0)
    assert out_val.shape == (len(out_v), 3)
    # Each column must equal the same filter run on that column alone.
    for c in range(3):
        v_c, val_c = sc.filters.smooth(voxels, values=values[:, c], sigma=1.0)
        assert np.array_equal(v_c, out_v)
        assert np.allclose(val_c, out_val[:, c])


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64, np.uint16])
def test_coordinate_dtype_is_preserved(dtype):
    # uint16 would underflow on the outward growth, so keep the shape away from 0.
    voxels = (solid_cube(3) + 20).astype(dtype)
    out_v, out_val = sc.filters.smooth(voxels, sigma=1.0)
    assert out_v.dtype == dtype
    assert out_val.dtype == np.float64


def test_integer_values_are_promoted_to_float():
    voxels = solid_cube(3).astype(np.int64)
    _, out_val = sc.filters.smooth(voxels, values=np.arange(27, dtype=np.int32), sigma=1.0)
    assert np.issubdtype(out_val.dtype, np.floating)


# --- epsilon ----------------------------------------------------------------


def test_epsilon_shrinks_the_support():
    voxels = solid_cube(4).astype(np.int64)
    exact_v, exact_val = sc.filters.smooth(voxels, sigma=2.0)
    pruned_v, pruned_val = sc.filters.smooth(
        voxels, sigma=2.0, epsilon=float(exact_val.max()) * 1e-3
    )
    assert len(pruned_v) < len(exact_v)
    # What survives must be a subset, and close to the exact values.
    assert sc.binary.isin(pruned_v, exact_v).all()
    src = sc.binary.index_of(pruned_v, exact_v)
    assert np.allclose(pruned_val, exact_val[src], rtol=0.05)


def test_epsilon_zero_is_exact():
    voxels = l_shape().astype(np.int64)
    a = sc.filters.smooth(voxels, sigma=1.0)
    b = sc.filters.smooth(voxels, sigma=1.0, epsilon=0.0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_epsilon_loses_mass_monotonically():
    """Pruning only ever removes; the total can fall but never rise."""
    voxels = solid_cube(4).astype(np.int64)
    _, exact = sc.filters.smooth(voxels, sigma=2.0)
    peak = float(exact.max())
    prev = exact.sum()
    for rel in (1e-6, 1e-3, 1e-1):
        _, val = sc.filters.smooth(voxels, sigma=2.0, epsilon=peak * rel)
        assert val.sum() <= prev + 1e-12
        prev = val.sum()


# --- degenerate input / validation -----------------------------------------


def test_empty():
    empty = np.empty((0, 3), dtype=np.int64)
    out_v, out_val = sc.filters.smooth(empty, sigma=1.0)
    assert out_v.shape == (0, 3) and len(out_val) == 0


def test_single_voxel_is_the_kernel_itself():
    out_v, out_val = sc.filters.smooth(np.array([[0, 0, 0]]), sigma=1.0, radius=1)
    assert len(out_v) == 27  # a 3x3x3 box
    assert out_val.sum() == pytest.approx(1.0)
    # The centre must be the heaviest cell.
    centre = np.flatnonzero((out_v == 0).all(axis=1))[0]
    assert out_val[centre] == out_val.max()


@pytest.mark.parametrize("bad", [-1.0, (1.0, 2.0), (1, 2, 3, 4), np.inf, np.nan])
def test_rejects_bad_sigma(bad):
    with pytest.raises(ValueError, match="sigma"):
        sc.filters.smooth(solid_cube(3).astype(np.int64), sigma=bad)


@pytest.mark.parametrize("bad", [-1, 1.5, (1, 2), (1, 2, 3, 4)])
def test_rejects_bad_radius(bad):
    with pytest.raises(ValueError, match="radius"):
        sc.filters.smooth(solid_cube(3).astype(np.int64), sigma=1.0, radius=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_rejects_bad_truncate(bad):
    with pytest.raises(ValueError, match="truncate"):
        sc.filters.smooth(solid_cube(3).astype(np.int64), sigma=1.0, truncate=bad)


def test_rejects_negative_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        sc.filters.smooth(solid_cube(3).astype(np.int64), sigma=1.0, epsilon=-1.0)


def test_rejects_misaligned_values():
    with pytest.raises(ValueError, match="aligned"):
        sc.filters.smooth(solid_cube(3).astype(np.int64), values=np.zeros(5), sigma=1.0)


def test_rejects_3d_values():
    voxels = solid_cube(3).astype(np.int64)
    with pytest.raises(ValueError, match=r"\(N,\) or \(N, C\)"):
        sc.filters.smooth(voxels, values=np.zeros((len(voxels), 2, 2)), sigma=1.0)


def test_rejects_non_integer_voxels():
    with pytest.raises(TypeError, match="integer"):
        sc.filters.smooth(solid_cube(3).astype(float), sigma=1.0)


def test_rejects_wrong_shape():
    with pytest.raises(TypeError, match=r"\(N, 3\)"):
        sc.filters.smooth(np.zeros((5, 2), dtype=np.int64), sigma=1.0)


# --- interop ----------------------------------------------------------------


def test_accepts_sparse_input():
    spx = pytest.importorskip("scipy.sparse")
    voxels = solid_cube(4).astype(np.int64) + 5
    try:
        coo = spx.coo_array(
            (np.ones(len(voxels), bool), tuple(voxels.T)), shape=(20, 20, 20)
        )
    except Exception:
        pytest.skip("scipy is too old for 3-D sparse arrays (needs >= 1.15)")
    a_v, a_val = sc.filters.smooth(coo, sigma=1.0)
    b_v, b_val = sc.filters.smooth(voxels, sigma=1.0)
    assert np.array_equal(a_v, b_v) and np.allclose(a_val, b_val)


def test_result_feeds_back_into_the_binary_primitives():
    """The documented workflow: smooth, threshold, carry on as a voxel set."""
    voxels = solid_cube(4).astype(np.int64)
    out_v, out_val = sc.filters.smooth(voxels, sigma=1.0)
    blurred = out_v[out_val > 0.5]
    assert len(blurred) and sc.binary.isin(blurred, out_v).all()
    assert sc.measure.volume(blurred) > 0


# --- correlate --------------------------------------------------------------


def _dense_correlate(voxels, values, weights, axis=None):
    """`scipy.ndimage.correlate`(1d) on a box padded past the kernel reach."""
    from scipy import ndimage

    reach = max(np.asarray(weights).shape) if axis is None else len(weights)
    pad = int(reach) + 2
    lo = voxels.min(axis=0) - pad
    grid = np.zeros(tuple(voxels.max(axis=0) - lo + pad + 1), dtype=float)
    np.add.at(grid, tuple((voxels - lo).T), values)
    if axis is None:
        return ndimage.correlate(grid, weights, mode="constant", cval=0.0), lo
    return ndimage.correlate1d(grid, weights, axis=axis, mode="constant", cval=0.0), lo


ASYMMETRIC = [
    np.array([1.0, -3.0, 2.0]),
    np.array([2.0, 1.0, -1.0, 0.5, 4.0]),
    np.array([0.0, 5.0, 1.0]),
]


@needs_scipy
def test_correlate_separable_matches_scipy():
    """An asymmetric kernel pins the correlation *direction*, which a symmetric
    one (every Gaussian, every box) cannot."""
    rng = np.random.RandomState(0)
    voxels = np.unique(rng.randint(0, 10, (150, 3)), axis=0).astype(np.int64)
    values = rng.rand(len(voxels)) + 0.5

    expected, lo = _dense_correlate(voxels, values, ASYMMETRIC[0], axis=0)
    from scipy import ndimage

    expected = ndimage.correlate1d(expected, ASYMMETRIC[1], axis=1, mode="constant")
    expected = ndimage.correlate1d(expected, ASYMMETRIC[2], axis=2, mode="constant")

    out_v, out_val = sc.filters.correlate(voxels, ASYMMETRIC, values=values)
    assert np.allclose(_as_grid(out_v, out_val, expected.shape, lo), expected, atol=1e-12)


@needs_scipy
@pytest.mark.parametrize("shape", [(3, 3, 3), (1, 3, 5), (5, 5, 1)])
def test_correlate_dense_matches_scipy(shape):
    rng = np.random.RandomState(1)
    voxels = np.unique(rng.randint(0, 9, (120, 3)), axis=0).astype(np.int64)
    values = rng.rand(len(voxels)) + 0.5
    kernel = rng.rand(*shape) - 0.5

    expected, lo = _dense_correlate(voxels, values, kernel)
    out_v, out_val = sc.filters.correlate(voxels, kernel, values=values)
    assert np.allclose(_as_grid(out_v, out_val, expected.shape, lo), expected, atol=1e-12)


def test_correlate_separable_equals_its_outer_product():
    """The two code paths must agree where the kernel genuinely separates."""
    rng = np.random.RandomState(2)
    voxels = np.unique(rng.randint(0, 8, (80, 3)), axis=0).astype(np.int64)
    values = rng.rand(len(voxels)) + 0.5
    a, b, c = np.array([1.0, 2.0, 1.0]), np.array([1.0, -1.0, 3.0]), np.array([2.0, 1.0, 1.0])
    outer = a[:, None, None] * b[None, :, None] * c[None, None, :]

    sep_v, sep_val = sc.filters.correlate(voxels, [a, b, c], values=values)
    dense_v, dense_val = sc.filters.correlate(voxels, outer, values=values)
    assert np.array_equal(sep_v, dense_v)
    assert np.allclose(sep_val, dense_val)


@needs_scipy
def test_smooth_is_correlate_with_a_gaussian_kernel():
    """`smooth` is documented as a wrapper - hold it to that."""
    voxels = solid_cube(4).astype(np.int64)
    k = sc.filters.gaussian_kernel1d(1.0, int(4.0 * 1.0 + 0.5))
    a = sc.filters.smooth(voxels, sigma=1.0)
    b = sc.filters.correlate(voxels, [k, k, k])
    assert np.array_equal(a[0], b[0])
    assert np.allclose(a[1], b[1])


def test_correlate_dense_ignores_zero_taps():
    """Zero taps cost nothing and contribute nothing, so they must not widen the
    support either."""
    voxels = np.array([[0, 0, 0]], dtype=np.int64)
    kernel = np.zeros((5, 5, 5))
    kernel[2, 2, 2] = 1.0
    kernel[1, 2, 2] = 2.0
    out_v, out_val = sc.filters.correlate(voxels, kernel)
    assert len(out_v) == 2  # only the two non-zero taps land
    assert sorted(out_val.tolist()) == [1.0, 2.0]


@pytest.mark.parametrize(
    "bad", [np.ones(4), [np.ones(3), np.ones(3)], np.ones((2, 3, 3)), np.ones((3, 3))]
)
def test_correlate_rejects_bad_weights(bad):
    with pytest.raises(ValueError):
        sc.filters.correlate(solid_cube(3).astype(np.int64), bad)


def test_correlate_on_empty():
    empty = np.empty((0, 3), dtype=np.int64)
    k = np.ones(3) / 3
    out_v, out_val = sc.filters.correlate(empty, [k, k, k])
    assert out_v.shape == (0, 3) and len(out_val) == 0


# --- maximum / minimum ------------------------------------------------------


def _dense_rank(voxels, values, size, which):
    from scipy import ndimage

    pad = int(np.max(size)) + 2
    lo = voxels.min(axis=0) - pad
    grid = np.zeros(tuple(voxels.max(axis=0) - lo + pad + 1), dtype=float)
    np.add.at(grid, tuple((voxels - lo).T), values)
    fn = ndimage.maximum_filter if which == "max" else ndimage.minimum_filter
    return fn(grid, size=size, mode="constant", cval=0.0), lo


@needs_scipy
@pytest.mark.parametrize("size", [3, 5, (3, 5, 3), (1, 3, 1)])
@pytest.mark.parametrize("which", ["max", "min"])
@pytest.mark.parametrize("signed", [False, True])
def test_rank_filters_match_scipy(which, size, signed):
    """Signed values are the interesting case: a window reaching outside the set
    picks up the implicit background zero, which sparse must reproduce."""
    rng = np.random.RandomState(3)
    voxels = np.unique(rng.randint(0, 9, (140, 3)), axis=0).astype(np.int64)
    values = rng.rand(len(voxels)) * 4 - 2 if signed else rng.rand(len(voxels)) + 0.5
    values = values[values != 0]
    voxels = voxels[: len(values)]

    expected, lo = _dense_rank(voxels, values, size, which)
    fn = sc.filters.maximum if which == "max" else sc.filters.minimum
    out_v, out_val = fn(voxels, values=values, size=size)
    assert np.allclose(_as_grid(out_v, out_val, expected.shape, lo), expected, atol=1e-14)


@needs_scipy
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_rank_filters_on_shapes(name):
    voxels = SHAPES[name].astype(np.int64)
    values = np.arange(len(voxels), dtype=float) + 1.0
    for which, fn in (("max", sc.filters.maximum), ("min", sc.filters.minimum)):
        expected, lo = _dense_rank(voxels, values, 3, which)
        out_v, out_val = fn(voxels, values=values, size=3)
        assert np.allclose(
            _as_grid(out_v, out_val, expected.shape, lo), expected, atol=1e-14
        )


def test_maximum_on_a_mask_is_the_box_dilation():
    """On an indicator field, grayscale and binary dilation coincide."""
    voxels = l_shape().astype(np.int64)
    out_v, out_val = sc.filters.maximum(voxels, size=3)
    assert (out_val == 1.0).all()
    assert set(map(tuple, out_v.tolist())) == set(
        map(tuple, sc.binary.dilate(voxels, connectivity=26).tolist())
    )


def test_minimum_on_a_mask_is_the_box_erosion():
    voxels = solid_cube(6).astype(np.int64)
    out_v, out_val = sc.filters.minimum(voxels, size=3)
    assert (out_val == 1.0).all()
    assert set(map(tuple, out_v.tolist())) == set(
        map(tuple, sc.binary.erode(voxels, connectivity=26).tolist())
    )


def test_minimum_shrinks_and_maximum_grows():
    """The property that makes `minimum` the one filter that stays sparse."""
    voxels = solid_cylinder(4, 12).astype(np.int64)
    assert len(sc.filters.minimum(voxels, size=3)[0]) < len(voxels)
    assert len(sc.filters.maximum(voxels, size=3)[0]) > len(voxels)


def test_minimum_can_empty_a_thin_object():
    out_v, out_val = sc.filters.minimum(line(8).astype(np.int64), size=3)
    assert out_v.shape == (0, 3) and len(out_val) == 0


def test_rank_filters_size_one_is_the_identity():
    voxels = np.unique(l_shape(), axis=0).astype(np.int64)
    values = np.arange(len(voxels), dtype=float) + 1.0
    for fn in (sc.filters.maximum, sc.filters.minimum):
        out_v, out_val = fn(voxels, values=values, size=1)
        assert np.array_equal(out_v, voxels)
        assert np.allclose(out_val, values)


def test_rank_filters_multi_column_values():
    rng = np.random.RandomState(4)
    voxels = np.unique(rng.randint(0, 7, (60, 3)), axis=0).astype(np.int64)
    values = rng.rand(len(voxels), 3) + 0.5
    for fn in (sc.filters.maximum, sc.filters.minimum):
        out_v, out_val = fn(voxels, values=values, size=3)
        assert out_val.shape == (len(out_v), 3)
        for c in range(3):
            v_c, val_c = fn(voxels, values=values[:, c], size=3)
            assert np.array_equal(v_c, out_v)
            assert np.allclose(val_c, out_val[:, c])


@pytest.mark.parametrize("bad", [2, 4, 0, -1, 1.5, (1, 2), (3, 3, 3, 3)])
def test_rank_filters_reject_bad_size(bad):
    with pytest.raises(ValueError, match="size"):
        sc.filters.maximum(solid_cube(3).astype(np.int64), size=bad)


def test_rank_filters_on_empty():
    empty = np.empty((0, 3), dtype=np.int64)
    for fn in (sc.filters.maximum, sc.filters.minimum):
        out_v, out_val = fn(empty, size=3)
        assert out_v.shape == (0, 3) and len(out_val) == 0
