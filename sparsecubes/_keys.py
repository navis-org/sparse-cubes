"""Shared packed-key machinery for the sparse set-algebra and measurement primitives.

Everything here works on the int64 keys `core.pack` produces, so membership,
dedup and neighbour steps become 1-D sorted-array operations instead of row-wise
comparisons on ``(N, 3)`` coordinates. No dense grid is ever allocated: cost
scales with the voxel count, never the bounding-box volume.

The one invariant worth stating explicitly: stepping one voxel along an axis is
a *constant addition* on the key only while no field carries or borrows into its
neighbour (see `core.pack`'s ``(21, 21, 21)`` layout). `to_keys` therefore shifts
the cloud so the minimum coordinate sits at `margin` and refuses inputs whose
extent plus that slack would leave the 21-bit range - without it, a ``-1`` step
at ``y == 0`` would silently borrow and alias onto a genuine voxel.
"""

import numpy as np

from .core import INT_DTYPES, _PACK_FIELD, pack, unpack, unique

# Bit offsets of pack()'s default (21, 21, 21) layout.
_X_SHIFT, _Y_SHIFT = 42, 21


def validate(voxels, name="voxels"):
    """Same input contract as `core.mesh`: an ``(N, 3)`` integer array."""
    if not isinstance(voxels, np.ndarray):
        raise TypeError(f'{name}: expected numpy array, got "{type(voxels)}"')
    if voxels.ndim != 2:
        raise TypeError(f'{name}: expected 2d numpy array, got "{voxels.ndim}"')
    if voxels.shape[1] != 3:
        raise TypeError(f'{name}: expected shape (N, 3), got "{voxels.shape}"')
    if voxels.dtype not in INT_DTYPES:
        raise TypeError(f"{name}: expected integer dtype, got {voxels.dtype}")


def as_spacing(spacing):
    """Normalise `spacing` to a length-3 float array, or None."""
    if spacing is None:
        return None
    s = np.asarray(spacing, dtype=float)
    if s.ndim == 0:
        s = np.repeat(s, 3)
    if s.shape != (3,):
        raise ValueError(f"spacing must be a scalar or length-3, got shape {s.shape}")
    if np.any(s <= 0):
        raise ValueError(f"spacing must be positive, got {spacing!r}")
    return s


def neighbor_offsets(connectivity):
    """``(K, 3)`` int64 neighbour offsets for 6-, 18- or 26-connectivity."""
    if connectivity not in (6, 18, 26):
        raise ValueError(f"connectivity must be 6, 18 or 26, got {connectivity!r}")
    off = np.array(
        [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if dx or dy or dz
        ],
        dtype=np.int64,
    )
    l1 = np.abs(off).sum(axis=1)  # 1 = face, 2 = edge, 3 = corner
    if connectivity == 6:
        return off[l1 == 1]
    if connectivity == 18:
        return off[l1 <= 2]
    return off


def key_deltas(connectivity):
    """Packed-key deltas for a connectivity's neighbour offsets.

    Addition (not bitwise or) is deliberate - the offsets are signed, and the
    no-carry guarantee `to_keys` establishes is what makes ``key + delta`` equal
    the key of the neighbouring coordinate.
    """
    off = neighbor_offsets(connectivity)
    return (off[:, 0] << _X_SHIFT) + (off[:, 1] << _Y_SHIFT) + off[:, 2]


def to_keys(voxels, margin=1, name="voxels"):
    """Validate `voxels`; return its sorted unique packed keys and the shift used.

    Coordinates are shifted so the minimum sits at `margin`, leaving room for
    `margin` steps in either direction without a field carry corrupting a key.
    Raises if the extent plus that slack would exceed pack()'s per-axis range.
    """
    validate(voxels, name)
    if len(voxels) == 0:
        return np.empty(0, dtype=np.int64), np.zeros(3, dtype=np.int64)
    vox = voxels.astype(np.int64, copy=False)
    shift = vox.min(axis=0) - margin
    v = vox - shift
    _check_range(int(v.max()), margin)
    return unique(pack(v)), shift


def _common_origin(arrays, margin, names):
    """Validate `arrays` and find the shared origin their keys are measured from.

    Returns ``(shift, dtype)``, or ``(None, dtype)`` when every input is empty -
    empty sets do not constrain the origin and have no keys to place on it.
    """
    names = names or [f"voxels[{i}]" for i in range(len(arrays))]
    for arr, name in zip(arrays, names):
        validate(arr, name)
    dtype = np.result_type(*[a.dtype for a in arrays]) if arrays else np.int64

    filled = [a for a in arrays if len(a)]
    if not filled:
        return None, dtype

    lo = np.min([a.min(axis=0) for a in filled], axis=0).astype(np.int64)
    hi = np.max([a.max(axis=0) for a in filled], axis=0).astype(np.int64)
    shift = lo - margin
    _check_range(int((hi - shift).max()), margin)
    return shift, dtype


def to_common_keys(arrays, margin=0, names=None):
    """Pack several voxel sets onto one shared origin.

    Returns ``(list_of_key_arrays, shift, dtype)``, each key array sorted and
    deduplicated. A shared shift is what makes the keys comparable across sets,
    so set algebra reduces to sorted-array membership. Empty inputs pack to empty
    key arrays and do not constrain the origin. Raises if the *combined* extent
    leaves pack()'s range.

    Use `to_row_keys` instead when the result has to stay aligned with the
    caller's rows.
    """
    shift, dtype = _common_origin(arrays, margin, names)
    if shift is None:
        return [np.empty(0, dtype=np.int64) for _ in arrays], np.zeros(3, np.int64), dtype

    keys = [
        unique(pack(a.astype(np.int64, copy=False) - shift))
        if len(a)
        else np.empty(0, dtype=np.int64)
        for a in arrays
    ]
    return keys, shift, dtype


def to_row_keys(arrays, margin=0, names=None):
    """Pack several voxel sets onto one shared origin, *preserving row order*.

    Same contract as `to_common_keys` minus the sort and dedup, so key ``i`` is
    row ``i`` of the corresponding input. That alignment is the whole point for
    the per-row operations (`binary.isin`, `binary.index_of`), which have to hand
    back an answer the caller can lay against their own array.
    """
    shift, dtype = _common_origin(arrays, margin, names)
    if shift is None:
        return [np.empty(0, dtype=np.int64) for _ in arrays], np.zeros(3, np.int64), dtype

    keys = [
        pack(a.astype(np.int64, copy=False) - shift)
        if len(a)
        else np.empty(0, dtype=np.int64)
        for a in arrays
    ]
    return keys, shift, dtype


def _check_range(max_coord, margin):
    """Guard pack()'s 21-bit-per-axis range, accounting for the outward slack."""
    if max_coord + margin >= _PACK_FIELD:
        raise ValueError(
            f"Voxel extent (+ {margin} voxel margin) reaches {max_coord + margin}, "
            f"exceeding pack()'s {_PACK_FIELD}-per-axis range. Downsample "
            f"(e.g. `voxels // k`) or split the volume first."
        )


def from_keys(keys, shift, dtype=np.int64):
    """Inverse of `to_keys`: unpack and undo the shift, back to `dtype`."""
    if len(keys) == 0:
        return np.empty((0, 3), dtype=dtype)
    return (unpack(keys) + shift).astype(dtype, copy=False)


def sorted_hit(keys, ref_sorted):
    """Boolean mask: which `keys` are present in the sorted array `ref_sorted`."""
    if len(ref_sorted) == 0 or len(keys) == 0:
        return np.zeros(len(keys), dtype=bool)
    pos = np.clip(np.searchsorted(ref_sorted, keys), 0, len(ref_sorted) - 1)
    return ref_sorted[pos] == keys
