"""Value-domain filtering on sparse ``(N, 3)`` voxel clouds.

`binary` maps voxel sets to voxel sets and `measure` reduces them to numbers;
this module is the third kind - it carries a *value per voxel* and redistributes
it. Filtering is a weighted (or rank) combination of a neighbourhood, not a set
operation, so the value and the coordinate move together and both come back.

Everything here is one mechanism: shift the packed keys by a constant, combine
the collisions. `_pass_1d` does that along a single axis, which is what makes the
**separable** filters cheap - a radius-``r`` kernel costs ``3 * (2r + 1)`` taps
instead of ``(2r + 1) ** 3``. `correlate` also accepts a full 3-D kernel for the
non-separable case, at the cost you would expect.

Two conventions run through the module:

* **Absent means zero.** A coordinate not in the set has value 0, exactly as
  `scipy.ndimage`'s ``mode="constant", cval=0`` treats everything beyond its
  array. That is what makes the sparse and dense answers identical rather than
  merely similar, and it is why the rank filters below have to reason about
  whether a window was fully covered.
* **`correlate`, not convolve.** ``out[m] = sum_j w[j] * in[m + j]``, matching
  `scipy.ndimage.correlate`. For a symmetric kernel (any Gaussian, any box) the
  distinction is invisible; for an asymmetric one - a derivative - it is a sign
  flip, so it is spelled out at the one place it is implemented.

Two honest caveats, both inherent rather than implementation artefacts:

* **Most of these grow the support.** The output covers the input dilated by the
  kernel radius, so a cloud at occupancy ``p`` produces roughly
  ``p * (2r + 1) ** 3`` worth of output cells. Past a certain radius the dense
  grid genuinely wins - see ``benchmarks/bench_smooth.py``, which measures the
  crossover rather than guessing at it. `minimum` is the exception: it can only
  shrink the support, so it stays sparse at any radius.
* **Values are required to mean something.** With ``values=None`` the set is
  treated as an indicator function (1.0 inside, 0.0 outside), which is what
  filtering a binary mask means; anything else has to be supplied explicitly,
  because a voxel set alone does not say what is being combined.
"""

import numpy as np

from ._keys import from_keys, to_keys, validate
from ._sparse import sparse_aware
from .core import pack

__all__ = ["smooth", "correlate", "maximum", "minimum", "gaussian_kernel1d"]

# Bit offsets of pack()'s default (21, 21, 21) layout, as packed-key deltas for a
# one-voxel step along x, y and z. Adding these is exact only because `to_keys`
# shifts the cloud away from the field edges - see `_keys`.
_AXIS_KEY_DELTA = (1 << 42, 1 << 21, 1)


# --- kernels ----------------------------------------------------------------


def gaussian_kernel1d(sigma, radius):
    """The normalised 1-D Gaussian taps for ``x in [-radius, radius]``.

    Deliberately byte-identical to `scipy.ndimage`'s internal
    ``_gaussian_kernel1d(sigma, 0, radius)``: sampled at the integer offsets and
    normalised *over the truncated window*, not over the analytic Gaussian. That
    renormalisation is what makes the filter preserve total mass despite the
    truncation, and matching it is what lets `smooth` agree with the dense
    implementation exactly rather than approximately.
    """
    if sigma <= 0:
        return np.ones(1, dtype=float)
    x = np.arange(-radius, radius + 1, dtype=float)
    phi = np.exp(-0.5 / (sigma * sigma) * x * x)
    return phi / phi.sum()


def _as_sigma(sigma):
    """Normalise `sigma` to a length-3 float array of non-negative values."""
    s = np.asarray(sigma, dtype=float)
    if s.ndim == 0:
        s = np.repeat(s, 3)
    if s.shape != (3,):
        raise ValueError(f"sigma must be a scalar or length-3, got shape {s.shape}")
    if np.any(s < 0) or not np.all(np.isfinite(s)):
        raise ValueError(f"sigma must be finite and >= 0, got {sigma!r}")
    return s


def _radii(sigma, truncate, radius):
    """Per-axis kernel radii, matching `scipy.ndimage.gaussian_filter1d`.

    scipy uses ``int(truncate * sigma + 0.5)`` unless an explicit `radius`
    overrides it; reproducing that rounding exactly is a precondition for the
    two implementations agreeing tap for tap.
    """
    if radius is not None:
        return _as_radius(radius)
    if truncate <= 0:
        raise ValueError(f"truncate must be > 0, got {truncate!r}")
    return np.array([int(truncate * s + 0.5) for s in sigma], dtype=np.int64)


def _as_radius(radius, name="radius"):
    """Normalise an int-or-length-3 radius to a non-negative int64 triple."""
    r = np.asarray(radius)
    if not np.issubdtype(r.dtype, np.integer):
        raise ValueError(f"{name} must be an integer, got {radius!r}")
    if r.ndim == 0:
        r = np.repeat(r, 3)
    if r.shape != (3,):
        raise ValueError(f"{name} must be an int or length-3, got shape {r.shape}")
    if np.any(r < 0):
        raise ValueError(f"{name} must be >= 0, got {radius!r}")
    return r.astype(np.int64)


def _as_size(size):
    """Normalise a rank filter's box `size` to per-axis radii. Sides must be odd."""
    s = np.asarray(size)
    if not np.issubdtype(s.dtype, np.integer):
        raise ValueError(f"size must be an integer, got {size!r}")
    if s.ndim == 0:
        s = np.repeat(s, 3)
    if s.shape != (3,):
        raise ValueError(f"size must be an int or length-3, got shape {s.shape}")
    if np.any(s < 1) or np.any(s % 2 == 0):
        raise ValueError(
            f"size must be odd and >= 1 so the window is centred, got {size!r}"
        )
    return (s.astype(np.int64) - 1) // 2


def _as_weights(weights):
    """Classify `weights` as ``("separable", [k0, k1, k2])`` or ``("dense", k)``.

    A list of three 1-D arrays means "apply these along x, y and z in turn"; a
    single 3-D array is correlated directly. Kernels must have odd extent so the
    window is centred - `scipy.ndimage`'s `origin` shift has no analogue here and
    an off-centre kernel is far more often a mistake than an intention.
    """
    if (
        isinstance(weights, (list, tuple))
        and len(weights) == 3
        and all(np.ndim(w) == 1 for w in weights)
    ):
        kernels = [np.asarray(w, dtype=float) for w in weights]
        for k in kernels:
            if len(k) == 0 or len(k) % 2 == 0:
                raise ValueError(
                    f"separable kernels must have odd, non-zero length so they are "
                    f"centred; got lengths {[len(k) for k in kernels]}"
                )
        return "separable", kernels

    arr = np.asarray(weights, dtype=float)
    if arr.ndim == 3:
        if any(s == 0 or s % 2 == 0 for s in arr.shape):
            raise ValueError(
                f"a 3-D kernel must have odd, non-zero extent on every axis so it "
                f"is centred; got shape {arr.shape}"
            )
        return "dense", arr

    raise ValueError(
        "weights must be either three 1-D kernels [kx, ky, kz] (separable) or a "
        f"single 3-D array (direct); got {type(weights).__name__} with shape "
        f"{getattr(arr, 'shape', None)}"
    )


# --- the shared machinery ---------------------------------------------------


def _sum_duplicates(keys, values):
    """Collapse repeated keys, summing their values. Returns sorted keys + sums.

    Only used for the cheap initial dedup of the caller's own rows; the hot path
    goes through `_accumulate`, which is careful about references in a way a
    plain two-argument helper cannot be.
    """
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    starts = np.flatnonzero(np.concatenate(([True], ks[1:] != ks[:-1])))
    return ks[starts], np.add.reduceat(values[order], starts, axis=0)


def _accumulate(box, ufunc, n_taps, clamp, epsilon):
    """Group the contributions in `box` by key and reduce each group with `ufunc`.

    This is the peak-memory step of every filter - `box` holds one contribution
    per voxel per tap, i.e. ``n_taps`` times the cloud - so each intermediate is
    released the moment it dies.

    `box` is a **list that this function empties**. That is not decoration: were
    the two arrays passed as ordinary arguments, the caller's frame would keep a
    second reference to each and the ``del``\\ s below would free nothing. Handing
    them over inside a container the callee can clear is what actually lets the
    big arrays die early, and it measured ~17% off the peak on million-voxel
    clouds.

    `clamp`, when given, is `np.maximum` or `np.minimum`: a rank filter over a
    window that was *not* fully covered has to fold in the implicit background
    zero, and `n_taps` is what makes "fully covered" checkable. Linear filters
    pass None - a missing contribution is a zero term, which adds nothing.
    """
    cat_keys, cat_values = box
    box.clear()  # drop the container's references; ours are now the only ones

    order = np.argsort(cat_keys, kind="stable")
    ks = cat_keys[order]
    del cat_keys
    starts = np.flatnonzero(np.concatenate(([True], ks[1:] != ks[:-1])))
    out_keys = ks[starts]
    n_contrib = len(ks)
    del ks
    sorted_values = cat_values[order]
    del cat_values, order
    out_values = ufunc.reduceat(sorted_values, starts, axis=0)
    del sorted_values

    if clamp is not None:
        # A group smaller than n_taps means some window cell was absent, i.e. 0.
        counts = np.diff(np.append(starts, n_contrib))
        partial = counts < n_taps
        if partial.any():
            folded = clamp(out_values, 0.0)
            out_values = np.where(
                partial.reshape((-1,) + (1,) * (out_values.ndim - 1)),
                folded,
                out_values,
            )

    # An exact zero is indistinguishable from absence in this representation (the
    # same rule `_sparse.to_voxels` applies to a sparse array's stored zeros), so
    # it leaves the set rather than becoming an explicit zero. That keeps the
    # output canonical, keeps `correlate`'s separable and direct paths agreeing
    # when a kernel has zero taps, and is exact - a dropped cell contributes
    # nothing to any later pass. `epsilon` is the same cut at a higher threshold.
    keep = np.abs(out_values).reshape(len(out_keys), -1).max(axis=1) > epsilon
    if not keep.all():
        out_keys, out_values = out_keys[keep], out_values[keep]
    return out_keys, out_values


def _pass_1d(keys, values, delta, offsets, weights, ufunc, clamp, epsilon):
    """One separable pass along the axis that `delta` steps through.

    `offsets` are the kernel's integer positions ``j``; the contribution from a
    source cell is pushed by ``-j`` so that ``out[m] = sum_j w[j] * in[m + j]``,
    i.e. a **correlation**, matching `scipy.ndimage`. Getting that sign right is
    invisible for symmetric kernels and a silent negation for antisymmetric ones.
    """
    if len(offsets) == 1 and offsets[0] == 0:
        return keys, values if weights is None else values * weights[0]

    shift = -np.asarray(offsets, dtype=np.int64) * delta
    if weights is None:  # rank filter: the value travels unscaled
        contributions = np.tile(values, (len(offsets),) + (1,) * (values.ndim - 1))
    else:
        scale = np.asarray(weights).reshape((-1,) + (1,) * values.ndim)
        contributions = (values[None, ...] * scale).reshape((-1,) + values.shape[1:])

    # Built inline so `_accumulate` receives the only references - see its docstring.
    return _accumulate(
        [(keys[None, :] + shift[:, None]).ravel(), contributions],
        ufunc,
        len(offsets),
        clamp,
        epsilon,
    )


def _prepare(voxels, values, margin):
    """Validate, promote and pack. Returns ``(keys, values, shift)`` or None if empty.

    The margin has to cover the widest kernel, or a shifted key could carry out
    of its packed field and alias onto an unrelated voxel.
    """
    validate(voxels)
    if values is None:
        values = np.ones(len(voxels), dtype=float)
    else:
        values = np.asarray(values)
        if len(values) != len(voxels):
            raise ValueError(
                f"values must be aligned with voxels: got {len(values)} values "
                f"for {len(voxels)} voxels."
            )
        if values.ndim > 2:
            raise ValueError(f"values must be (N,) or (N, C), got shape {values.shape}")
        # Filtering is a weighted average - an integer field does not survive it.
        values = values.astype(np.result_type(values.dtype, np.float64), copy=False)

    if len(voxels) == 0:
        return None, values[:0].copy()

    _, shift = to_keys(voxels, margin=max(1, int(margin)))
    keys = pack(voxels.astype(np.int64, copy=False) - shift)
    keys, values = _sum_duplicates(keys, values)

    # Canonicalise: a voxel whose value is exactly 0 is indistinguishable from an
    # absent one - it contributes nothing to any window and still receives every
    # contribution an absent cell would - so dropping it here changes no result
    # and makes the convention uniform across input, output and intermediates.
    # Without this a no-op filter (sigma=0) would preserve explicit zeros that
    # any other filter silently discards.
    nz = np.abs(values).reshape(len(keys), -1).max(axis=1) > 0
    if not nz.all():
        keys, values = keys[nz], values[nz]
    return (keys, values, shift), None


def _separable(keys, values, kernels, ufunc, clamp, epsilon):
    """Run the three 1-D passes in order."""
    for axis, kernel in enumerate(kernels):
        r = len(kernel) // 2
        offsets = np.arange(-r, r + 1, dtype=np.int64)
        keys, values = _pass_1d(
            keys, values, _AXIS_KEY_DELTA[axis], offsets, kernel, ufunc, clamp, epsilon
        )
        if len(keys) == 0:
            break
    return keys, values


# --- public filters ---------------------------------------------------------


@sparse_aware
def correlate(voxels, weights, values=None, epsilon=0.0):
    """Correlate a sparse voxel cloud with an arbitrary kernel.

    The general linear filter, and the primitive the others are built from:
    ``out[m] = sum_j weights[j] * in[m + j]``, with everything outside the voxel
    set reading as zero. That is exactly `scipy.ndimage.correlate` with
    ``mode="constant", cval=0`` - pinned by the test suite - just without
    allocating the volume. For `scipy.ndimage.convolve` semantics, reverse the
    kernel along every axis first (``w[::-1]`` per 1-D kernel, ``w[::-1, ::-1,
    ::-1]`` for a 3-D one).

    Parameters
    ----------
    voxels :    (N, 3) integer array of XYZ voxel coordinates. Duplicate rows are
                summed, consistent with reading the input as a sparse array.
    weights :   three 1-D kernels, or one 3-D array
                ``[kx, ky, kz]`` applies the kernels along x, y and z in turn -
                use this whenever the filter separates, since it costs
                ``sum(len(k))`` taps instead of their product. A single 3-D array
                is correlated directly, at ``count_nonzero(weights)`` taps. Every
                kernel must have odd extent so it is centred.
    values :    (N,) or (N, C) array, optional
                The field being filtered, aligned with `voxels` row for row.
                Defaults to an indicator function - 1.0 on every voxel.
    epsilon :   float, optional
                Drop output voxels whose magnitude falls to `epsilon` or below,
                after each pass. See `smooth`. Default ``0.0`` - exact.

    Returns
    -------
    voxels : (M, 3) array of the input dtype, sorted - the input dilated by the
             kernel's reach (minus anything `epsilon` pruned).
    values : (M,) or (M, C) float array, aligned with the returned voxels.

    Examples
    --------
    A box blur, separably:

    >>> import numpy as np, sparsecubes as sc
    >>> k = np.full(3, 1 / 3)
    >>> vox, val = sc.filters.correlate(np.array([[0, 0, 0]]), [k, k, k])
    >>> len(vox), round(float(val.sum()), 12)
    (27, 1.0)

    See Also
    --------
    smooth : the Gaussian case, with the kernel built for you.
    maximum, minimum : the rank counterparts, which are not linear filters.
    """
    kind, kernels = _as_weights(weights)
    epsilon = _check_epsilon(epsilon)

    if kind == "separable":
        margin = max(len(k) // 2 for k in kernels)
    else:
        margin = max(s // 2 for s in kernels.shape)

    state, empty_values = _prepare(voxels, values, margin)
    if state is None:
        return voxels[:0].copy(), empty_values
    keys, vals, shift = state

    if kind == "separable":
        keys, vals = _separable(keys, vals, kernels, np.add, None, epsilon)
    else:
        keys, vals = _correlate_dense(keys, vals, kernels, epsilon)
    return from_keys(keys, shift, voxels.dtype), vals


def _correlate_dense(keys, values, kernel, epsilon):
    """Direct (non-separable) correlation with a 3-D kernel, in one accumulate."""
    centre = np.array(kernel.shape, dtype=np.int64) // 2
    idx = np.argwhere(kernel != 0)
    if len(idx) == 0:
        return keys[:0], values[:0]

    taps = kernel[tuple(idx.T)]
    offsets = idx - centre  # kernel positions j
    # Correlation pushes each contribution by -j, as in `_pass_1d`.
    deltas = -(
        offsets[:, 0] * _AXIS_KEY_DELTA[0]
        + offsets[:, 1] * _AXIS_KEY_DELTA[1]
        + offsets[:, 2] * _AXIS_KEY_DELTA[2]
    )
    scale = taps.reshape((-1,) + (1,) * values.ndim)
    return _accumulate(
        [
            (keys[None, :] + deltas[:, None]).ravel(),
            (values[None, ...] * scale).reshape((-1,) + values.shape[1:]),
        ],
        np.add,
        len(taps),
        None,
        epsilon,
    )


@sparse_aware
def smooth(voxels, values=None, sigma=1.0, truncate=4.0, radius=None, epsilon=0.0):
    """Gaussian-smooth a sparse voxel cloud without densifying it.

    The sparse counterpart of `scipy.ndimage.gaussian_filter` with
    ``mode="constant", cval=0`` - and *exactly* that, not an approximation:
    outside the voxel set the field is zero, which is precisely what the dense
    version assumes beyond its array bounds. Run on a densified bounding box the
    two agree to floating-point round-off (pinned by the test suite).

    Because a Gaussian has infinite support it is truncated at `truncate` standard
    deviations, again matching scipy's rounding. The output therefore covers the
    input dilated by that radius along each axis: **smoothing grows the voxel
    set**, and past a certain sigma or occupancy the dense grid is genuinely the
    better tool. ``benchmarks/bench_smooth.py`` measures where that crossover
    falls; `epsilon` moves it.

    Parameters
    ----------
    voxels :    (N, 3) integer array of XYZ voxel coordinates. Duplicate rows are
                summed, consistent with reading the input as a sparse array.
    values :    (N,) or (N, C) array, optional
                The field being smoothed, aligned with `voxels` row for row.
                Defaults to an indicator function - 1.0 on every voxel - which is
                what smoothing a binary mask means. Integer values are promoted
                to float; smoothing is a weighted average and does not stay
                integral.
    sigma :     float or length-3, optional
                Standard deviation **in voxels**, per axis. For a physical sigma
                divide by the voxel spacing yourself (``sigma / spacing``), which
                keeps this unambiguous when the grid is anisotropic. ``0`` along
                an axis disables smoothing on it.
    truncate :  float, optional
                Truncate the kernel at this many standard deviations. Default 4.0,
                as scipy. Lowering it is the cheapest way to bound the cost: the
                support grows as ``(2 * truncate * sigma + 1) ** 3``.
    radius :    int or length-3, optional
                Explicit kernel radius, overriding `truncate` (scipy's `radius`).
    epsilon :   float, optional
                Drop output voxels whose magnitude falls to `epsilon` or below,
                after *each* pass. The Gaussian's tail contributes vanishingly
                little but occupies the bulk of the grown support, so a small
                epsilon (say ``1e-6`` of the peak) keeps the cloud sparse at a
                bounded cost in exactness. Default ``0.0`` - exact, nothing
                dropped. Note the values are no longer guaranteed to sum to the
                input's total once this is non-zero.

    Returns
    -------
    voxels : (M, 3) array of the input dtype, sorted, ``M >= N`` - the input
             dilated by the kernel radius (minus anything `epsilon` pruned).
    values : (M,) or (M, C) float array, aligned with the returned voxels.

    Examples
    --------
    >>> import numpy as np, sparsecubes as sc
    >>> vox = np.array([[0, 0, 0]])
    >>> out, val = sc.filters.smooth(vox, sigma=1.0, radius=1)
    >>> len(out), round(float(val.sum()), 6)      # mass is preserved
    (27, 1.0)

    Threshold the result to get a voxel set back:

    >>> out, val = sc.filters.smooth(vox, sigma=1.0)
    >>> blurred = out[val > 0.01]

    See Also
    --------
    correlate : the same machinery with a kernel of your own.
    sparsecubes.binary.opening : removes specks by morphology rather than blur.
    sparsecubes.downsample : pools onto a coarser lattice, which also smooths.
    """
    sigma = _as_sigma(sigma)
    radii = _radii(sigma, truncate, radius)
    epsilon = _check_epsilon(epsilon)

    state, empty_values = _prepare(voxels, values, int(radii.max()))
    if state is None:
        return voxels[:0].copy(), empty_values
    keys, vals, shift = state

    kernels = [gaussian_kernel1d(sigma[a], int(radii[a])) for a in range(3)]
    keys, vals = _separable(keys, vals, kernels, np.add, None, epsilon)
    return from_keys(keys, shift, voxels.dtype), vals


def _rank_filter(voxels, values, size, ufunc, clamp):
    """Shared body of `maximum` and `minimum` - a separable box rank filter."""
    radii = _as_size(size)
    state, empty_values = _prepare(voxels, values, int(radii.max()))
    if state is None:
        return voxels[:0].copy(), empty_values
    keys, vals, shift = state

    kernels = [np.empty(2 * int(r) + 1) for r in radii]  # length only; no weights
    for axis, kernel in enumerate(kernels):
        r = len(kernel) // 2
        offsets = np.arange(-r, r + 1, dtype=np.int64)
        keys, vals = _pass_1d(
            keys, vals, _AXIS_KEY_DELTA[axis], offsets, None, ufunc, clamp, 0.0
        )
        if len(keys) == 0:
            break
    return from_keys(keys, shift, voxels.dtype), vals


@sparse_aware
def maximum(voxels, values=None, size=3):
    """Grayscale dilation: the maximum over a centred box window.

    `sparsecubes.binary.dilate` generalised from sets to values, and exactly
    `scipy.ndimage.maximum_filter` with ``mode="constant", cval=0``. On an
    indicator field the two coincide - the maximum of a 0/1 window is 1 wherever
    the box dilation reaches - so this is the version to use when the voxels
    carry intensities rather than mere membership.

    Like `smooth` this **grows** the support, to the input dilated by
    ``(size - 1) // 2``. Note the implicit background: a window that pokes
    outside the voxel set includes a zero, so the result is never negative there
    even if the surrounding values are.

    Parameters
    ----------
    voxels :    (N, 3) integer array of XYZ voxel coordinates.
    values :    (N,) or (N, C) array, optional. Defaults to an indicator field.
    size :      int or length-3, optional
                Side of the box window; must be odd so it is centred. Default 3,
                i.e. the 26-neighbourhood plus the voxel itself.

    Returns
    -------
    voxels : (M, 3) array of the input dtype, sorted.
    values : (M,) or (M, C) float array, aligned with the returned voxels.

    See Also
    --------
    minimum : the dual, which shrinks the support instead.
    sparsecubes.binary.dilate : the set-valued version, with 6/18/26 connectivity.
    """
    return _rank_filter(voxels, values, size, np.maximum, np.maximum)


@sparse_aware
def minimum(voxels, values=None, size=3):
    """Grayscale erosion: the minimum over a centred box window.

    `sparsecubes.binary.erode` generalised from sets to values, and exactly
    `scipy.ndimage.minimum_filter` with ``mode="constant", cval=0``.

    Alone among the filters here this **shrinks** the support, so it stays cheap
    at any radius - the sparsity advantage never erodes. That follows from the
    background convention: a window reaching outside the set contains a zero, so
    the minimum there is at most zero, and an exact zero is indistinguishable
    from absence and leaves the set. For a non-negative field the surviving
    support is therefore precisely the box erosion of the input.

    Parameters
    ----------
    voxels :    (N, 3) integer array of XYZ voxel coordinates.
    values :    (N,) or (N, C) array, optional. Defaults to an indicator field.
    size :      int or length-3, optional
                Side of the box window; must be odd so it is centred. Default 3.

    Returns
    -------
    voxels : (M, 3) array of the input dtype, sorted, ``M <= N``. May be empty -
             erosion can consume a thin object entirely.
    values : (M,) or (M, C) float array, aligned with the returned voxels.

    See Also
    --------
    maximum : the dual, which grows the support.
    sparsecubes.binary.erode : the set-valued version, with 6/18/26 connectivity.
    """
    return _rank_filter(voxels, values, size, np.minimum, np.minimum)


def _check_epsilon(epsilon):
    epsilon = float(epsilon)
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon!r}")
    return epsilon
