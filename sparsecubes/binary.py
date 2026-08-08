"""Set algebra and morphology on sparse ``(N, 3)`` voxel clouds.

Everything in this module maps voxel set(s) to a voxel set. The companion
`sparsecubes.measure` holds the operations that reduce a voxel set to numbers or
labels.

All of it runs on packed int64 keys (see `sparsecubes._keys`), so membership and
neighbour steps are sorted-array lookups rather than dense-grid indexing. Nothing
here allocates the bounding box; peak memory tracks the voxel count, which is the
whole point of the library - `scipy.ndimage.binary_dilation` and friends would
need the volume densified first.

Results are always deduplicated and returned in sorted (x, y, z) order, with the
input's integer dtype preserved.
"""

import itertools
import warnings

import numpy as np

from ._keys import (
    AXIS_STEPS,
    from_keys,
    key_deltas,
    sorted_hit,
    to_common_keys,
    to_keys,
    to_row_keys,
    unique,
    unique_sorted,
)
from ._sparse import sparse_aware
from .core import fill_cavities
from .thinning import thin

__all__ = [
    "dilate",
    "erode",
    "opening",
    "closing",
    "union",
    "intersection",
    "difference",
    "symmetric_difference",
    "isin",
    "index_of",
    "thin",
    "fill_cavities",
    "make_manifold",
]

# Voxel pairs that touch along an edge or at a corner but share no face. The 12
# edge and 8 corner directions come in opposite pairs, and a pair describes the
# same two voxels from either end, so fixing the first axis' step to +1 visits
# every unordered pair exactly once: 6 edges, 4 corners.
#
# Each entry is ``(delta, first_step, between)`` - the direction to the far
# voxel, the step to fill if the pair needs separating, and every cell strictly
# between the two, which is every proper non-empty subset sum of the axis steps.
def _contact_table():
    axis = [s * AXIS_STEPS[a] for a in range(3) for s in (1, -1)]
    combos = [(axis[0], axis[2 + s]) for s in (0, 1)]  # +x with +-y
    combos += [(axis[0], axis[4 + s]) for s in (0, 1)]  # +x with +-z
    combos += [(axis[2], axis[4 + s]) for s in (0, 1)]  # +y with +-z
    combos += [(axis[0], axis[2 + sy], axis[4 + sz]) for sy in (0, 1) for sz in (0, 1)]
    table = []
    for steps in combos:
        between = [
            sum(c)
            for r in range(1, len(steps))
            for c in itertools.combinations(steps, r)
        ]
        table.append((sum(steps), steps[0], between))
    return table


_CONTACTS = _contact_table()


def _dilate_keys(keys, deltas):
    """One dilation step: the set plus every neighbour step, deduplicated."""
    return unique(np.concatenate([keys] + [keys + d for d in deltas]))


def _erode_keys(keys, deltas):
    """One erosion step: keep only voxels whose every neighbour is also set."""
    keep = np.ones(len(keys), dtype=bool)
    for d in deltas:
        keep &= sorted_hit(keys + d, keys)
    return keys[keep]


@sparse_aware(mirror=True)
def dilate(voxels, iterations=1, connectivity=6):
    """Grow a voxel set by its neighbourhood.

    Adds every background voxel adjacent to the object, `iterations` times over.
    Note that iterating a 6-connected step is *not* the same as one step with a
    radius-``iterations`` ball (it grows a diamond) - this matches the
    ``iterations`` semantics of `scipy.ndimage.binary_dilation`.

    Parameters
    ----------
    voxels :        (N, 3) integer array of XYZ voxel coordinates.
    iterations :    int, optional. Number of dilation steps. Default 1.
    connectivity :  {6, 18, 26}, optional
                    Which neighbours count as adjacent: 6 = faces only
                    (default), 18 = faces + edges, 26 = faces + edges + corners.

    Returns
    -------
    (M, 3) array of the input dtype, ``M >= N``, deduplicated and sorted.

    Notes
    -----
    Peak memory is ``(connectivity + 1) * N`` int64 keys for one step, so a
    26-connected dilation of a very large cloud is the expensive case; prefer
    the default 6-connectivity where it suffices.
    """
    iterations = int(iterations)
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    keys, shift = to_keys(voxels, margin=max(1, iterations))
    if iterations == 0 or len(keys) == 0:
        return from_keys(keys, shift, voxels.dtype)
    deltas = key_deltas(connectivity)
    for _ in range(iterations):
        keys = _dilate_keys(keys, deltas)
    return from_keys(keys, shift, voxels.dtype)


@sparse_aware(mirror=True)
def erode(voxels, iterations=1, connectivity=6):
    """Shrink a voxel set by peeling voxels that touch the background.

    A voxel survives only if *all* of its `connectivity` neighbours are also in
    the set. Everything outside the input counts as background (equivalent to
    `scipy.ndimage.binary_erosion` with ``border_value=0``), so the outer shell
    is always removed.

    Parameters
    ----------
    voxels :        (N, 3) integer array of XYZ voxel coordinates.
    iterations :    int, optional. Number of erosion steps. Default 1.
    connectivity :  {6, 18, 26}, optional. See `dilate`.

    Returns
    -------
    (M, 3) array of the input dtype, ``M <= N``. May be empty - erosion can
    consume a thin object entirely. Use `sparsecubes.binary.thin` instead when
    you want to keep topology and endpoints.
    """
    iterations = int(iterations)
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    keys, shift = to_keys(voxels, margin=1)
    deltas = key_deltas(connectivity)
    for _ in range(iterations):
        if len(keys) == 0:
            break
        keys = _erode_keys(keys, deltas)
    return from_keys(keys, shift, voxels.dtype)


@sparse_aware(mirror=True)
def opening(voxels, iterations=1, connectivity=6):
    """Erode then dilate: removes specks and thin spurs, keeps bulk shape.

    The inverse pairing of `closing`. Small structures that erosion destroys do
    not come back, so this is the standard way to strip surface noise before
    meshing or skeletonizing. See `dilate` for the parameters.
    """
    return dilate(
        erode(voxels, iterations=iterations, connectivity=connectivity),
        iterations=iterations,
        connectivity=connectivity,
    )


@sparse_aware(mirror=True)
def closing(voxels, iterations=1, connectivity=6):
    """Dilate then erode: bridges narrow gaps and fills small pits.

    **Can fuse distinct structures** that pass within ``2 * iterations`` voxels
    of each other - the dilation merges them and the erosion cannot separate
    them again. When the goal is specifically to fill *enclosed* voids without
    that risk, use `fill_cavities` (``mode="exact"``), which is topology-safe.

    See `dilate` for the parameters.
    """
    return erode(
        dilate(voxels, iterations=iterations, connectivity=connectivity),
        iterations=iterations,
        connectivity=connectivity,
    )


@sparse_aware(mirror=True)
def union(*voxel_sets):
    """Voxels present in *any* of the given sets (deduplicated).

    Parameters
    ----------
    *voxel_sets :   one or more (N, 3) integer arrays.

    Returns
    -------
    (M, 3) array in the promoted dtype of the inputs, sorted.
    """
    if not voxel_sets:
        raise TypeError("union() needs at least one voxel set")
    keys, shift, dtype = to_common_keys(voxel_sets)
    return from_keys(unique(np.concatenate(keys)), shift, dtype)


@sparse_aware(mirror=True)
def intersection(*voxel_sets):
    """Voxels present in *every* one of the given sets.

    Parameters
    ----------
    *voxel_sets :   one or more (N, 3) integer arrays.

    Returns
    -------
    (M, 3) array in the promoted dtype of the inputs, sorted.
    """
    if not voxel_sets:
        raise TypeError("intersection() needs at least one voxel set")
    keys, shift, dtype = to_common_keys(voxel_sets)
    out = keys[0]
    for other in keys[1:]:
        if len(out) == 0:
            break
        out = out[sorted_hit(out, other)]
    return from_keys(out, shift, dtype)


@sparse_aware(mirror=True)
def difference(voxels, other):
    """Voxels in `voxels` but not in `other` (set subtraction).

    Returns
    -------
    (M, 3) array in the promoted dtype of the inputs, sorted.
    """
    keys, shift, dtype = to_common_keys([voxels, other], names=["voxels", "other"])
    a, b = keys
    return from_keys(a[~sorted_hit(a, b)], shift, dtype)


@sparse_aware(mirror=True)
def symmetric_difference(voxels, other):
    """Voxels in exactly one of the two sets (set XOR).

    Returns
    -------
    (M, 3) array in the promoted dtype of the inputs, sorted.
    """
    keys, shift, dtype = to_common_keys([voxels, other], names=["voxels", "other"])
    a, b = keys
    out = np.concatenate([a[~sorted_hit(a, b)], b[~sorted_hit(b, a)]])
    out.sort()
    return from_keys(out, shift, dtype)


def _contact_fixes(keys):
    """Keys to add to break every edge-only and corner-only contact in `keys`.

    Two voxels touching along an edge, with both of the cells that would share a
    face with each of them empty, meet the surface at a single *edge* - four
    quads hinge on it, so the mesh is non-manifold there and `is_watertight` is
    False. Two voxels touching at a corner, with all six cells between them
    empty, pinch the surface at a single vertex. Filling one of the in-between
    cells removes the pinch; which one is arbitrary, so take the first axis'.

    A corner contact is resolved to an edge contact rather than all the way, so
    the caller has to iterate - see `make_manifold`.

    The ten directions between them ask about the same neighbour offsets over and
    over (all six faces, and eight of the edges, recur), so the membership tests
    are memoized: every distinct offset costs exactly one pass, and the rest is
    boolean algebra on the masks. That turns ~46 sorted lookups into 17.
    """
    probe, add = {}, []

    def present(delta):
        """Mask over `keys`: is the voxel `delta` away also in the set?"""
        hit = probe.get(delta)
        if hit is None:
            hit = probe[delta] = sorted_hit(keys + delta, keys)
        return hit

    for delta, first, between in _CONTACTS:
        sel = present(delta)
        for gap in between:
            if not sel.any():
                break
            sel = sel & ~present(gap)
        if sel.any():
            add.append(keys[sel] + first)
    if not add:
        return np.empty(0, dtype=np.int64)
    return unique(np.concatenate(add))


@sparse_aware(mirror=True)
def make_manifold(voxels, max_iter=8):
    """Add voxels until no two touch along an edge or at a corner alone.

    Such a contact (see `_contact_fixes`) makes `sparsecubes.mesh`'s output
    non-manifold - trimesh reports ``is_watertight == False`` for an edge contact
    and a corrupted Euler number for a corner one - and downstream tools that
    assume a manifold surface (boolean ops, smoothing, slicers) misbehave on
    either. A clean solid contains none, so this is a no-op on one. They show up
    when a voxel set is built from something *incomplete*, notably
    `sparsecubes.voxelize` on a mesh with holes, where the columns it could not
    close leave thin shell fragments that graze each other.

    Filling a cell between the two offending voxels is the only repair
    available - nothing can be *removed* to separate them without eating into
    the object - so the set only grows, and never outside its bounding box.

    Parameters
    ----------
    voxels :    (N, 3) integer array of XYZ voxel coordinates.
    max_iter :  int, optional
                Safety cap on the repair rounds. Sealing one contact can create
                another (a corner contact resolves to an edge contact first), so
                this iterates to a fixed point; 8 rounds is far more than the two
                or three real inputs need. A warning is emitted if the cap is hit
                with contacts still outstanding.

    Returns
    -------
    (M, 3) array of the input dtype, ``M >= N``, deduplicated and sorted.

    Notes
    -----
    This changes topology on purpose: welding a pinch point can close a tunnel
    or merge two components that previously met at a single edge. When that
    matters, compare `sparsecubes.measure` counts before and after.

    See Also
    --------
    sparsecubes.binary.fill_cavities : Fill enclosed voids; never adds a voxel
                                that touches the exterior, so it cannot fuse.
    """
    max_iter = int(max_iter)
    if max_iter < 0:
        raise ValueError(f"max_iter must be >= 0, got {max_iter}")
    # Added cells stay strictly inside the input's bounding box, so a 1-voxel
    # margin is all the neighbour probing needs.
    keys, shift = to_keys(voxels, margin=1)
    if len(keys) == 0:
        return from_keys(keys, shift, voxels.dtype)

    rounds = 0
    add = _contact_fixes(keys) if max_iter else np.empty(0, dtype=np.int64)
    while len(add) and rounds < max_iter:
        keys = unique_sorted(keys, add)  # both sides are sorted already
        add = _contact_fixes(keys)
        rounds += 1
    if len(add):
        warnings.warn(
            f"make_manifold: still found edge/corner-only contacts after "
            f"{max_iter} rounds; the result may not be manifold. Raise `max_iter`.",
            stacklevel=3,
        )
    return from_keys(keys, shift, voxels.dtype)


@sparse_aware
def isin(voxels, other):
    """Per-voxel membership test: is each row of `voxels` also in `other`?

    Unlike the set operations above this preserves the input's row order and
    length, so the mask lines up with `voxels` (and with any per-voxel array you
    are carrying alongside it).

    Returns
    -------
    (N,) boolean array, aligned with `voxels` row for row.

    See Also
    --------
    index_of : *where* in `other` each voxel is, not just whether.
    """
    # Row keys for `voxels` (alignment), sorted-unique for `other` (lookup).
    keys, _, _ = to_row_keys([voxels, other], names=["voxels", "other"])
    own, b = keys
    if len(own) == 0:
        return np.zeros(0, dtype=bool)
    return sorted_hit(own, unique(b))


@sparse_aware
def index_of(voxels, other):
    """Locate each row of `voxels` in `other`: its row index there, or ``-1``.

    The lookup that makes it possible to carry per-voxel data (intensity,
    confidence, labels) through the *growing* operations. `isin` covers the
    shrinking ones - `erode`, `thin`, `difference` - where every output row came
    from the input. After a `dilate`, `union` or `fill_cavities` some output rows
    are new, and this is what tells you which::

        out = binary.dilate(voxels)
        src = binary.index_of(out, voxels)
        out_values = np.where(src >= 0, values[src], fill)

    Parameters
    ----------
    voxels :    (N, 3) integer array of XYZ voxel coordinates. Row order is
                preserved; duplicate rows each get the same answer.
    other :     (M, 3) integer array to look the voxels up in. Need not be
                sorted or deduplicated; where it holds a coordinate more than
                once the *lowest* matching row index is returned.

    Returns
    -------
    (N,) int64 array, aligned with `voxels` row for row. Entry ``i`` is the row
    index in `other` of ``voxels[i]``, or ``-1`` if that voxel is absent.
    Equivalent to ``isin(voxels, other)`` under ``>= 0``, so use `isin` when the
    mask is all you need.
    """
    keys, _, _ = to_row_keys([voxels, other], names=["voxels", "other"])
    q, src = keys
    if len(q) == 0:
        return np.empty(0, dtype=np.int64)
    if len(src) == 0:
        return np.full(len(q), -1, dtype=np.int64)
    # Stable argsort, so ties within an equal-key run stay in row order and
    # `searchsorted`'s left-hand landing point is the lowest matching row.
    order = np.argsort(src, kind="stable")
    ref = src[order]
    pos = np.clip(np.searchsorted(ref, q), 0, len(ref) - 1)
    return np.where(ref[pos] == q, order[pos], -1).astype(np.int64, copy=False)
