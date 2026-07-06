"""Sparse, vectorised 3D topological thinning.

Peels a sparse ``(N, 3)`` voxel set down to a one-voxel-wide medial *curve*
(a centerline), reusing the same sparse machinery as the meshing code: exact
integer coordinate hashing (`sparsecubes.core.pack`) for 26-neighbour lookups,
so memory scales with the current number of voxels rather than the dense
bounding-box volume.

The algorithm is a topological thinning in the spirit of Lee et al. (1994) /
Palagyi-Kuba: each pass runs six directional sub-iterations (the six face
directions); within a sub-iteration a voxel is removed if it is a *border*
point (its face-neighbour in that direction is empty), *not* an endpoint (so
curve tips are preserved) and a *simple* point (its removal preserves topology).
Simultaneous deletion is made topology-safe by splitting each sub-iteration into
eight ``(x & 1, y & 1, z & 1)`` sub-fields: two voxels sharing a parity class
differ by an even amount on every axis, so they are never 26-adjacent and can be
removed together without interacting. Simplicity is re-evaluated between
sub-fields because deleting one sub-field changes its neighbours' neighbourhoods.

Simplicity uses the Malandain-Bertrand characterisation of a 3D simple point for
the (26, 6) adjacency pair: ``p`` is simple if the foreground 26-neighbours form
a single 26-connected component (``T26 == 1``) and the background forms a single
6-connected component touching ``p`` (``T6 == 1``). Both conditions are evaluated
as bit-parallel flood fills on the 26-bit neighbourhood masks (one uint32 per
voxel, the whole batch at once), so the per-voxel work stays out of Python.
"""

import numpy as np

from .core import pack, unpack, log, INT_DTYPES, fill_cavities as _fill_cavities

__all__ = ["thin"]


# The 26 neighbour offsets in a fixed canonical (raster) order. Bit ``k`` of a
# neighbourhood mask corresponds to ``_OFF26[k]``; this ordering is load-bearing
# and pinned by a unit test.
_OFF26 = np.array(
    [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ],
    dtype=np.int64,
)

# The six face directions (Lee's border sub-iteration directions).
_OFF6 = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int64,
)

# Which of the 26 bits each face direction corresponds to.
_BIT_OF_OFF6 = np.array(
    [int(np.flatnonzero((_OFF26 == o).all(axis=1))[0]) for o in _OFF6]
)

# For each of the 26 offsets, the bit index of the *opposite* offset. If voxel v
# is deleted, its neighbour ``u = v + _OFF26[k]`` loses the bit ``_NEG_BIT[k]``
# (the direction from u back to v). Used to update masks incrementally.
_NEG_BIT = np.array(
    [int(np.flatnonzero((_OFF26 == -_OFF26[k]).all(axis=1))[0]) for k in range(26)],
    dtype=np.uint32,
)

# Byte pop-count table for a vectorised 26-bit degree count.
_BYTE_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)

# Per-axis extent must leave room for the min-1 shift and the +-1 neighbour
# stencil within pack()'s 21 bits per axis.
_EXTENT_LIMIT = (1 << 21) - 2

# Every packed coordinate stays within its 21-bit field (>= 1 after the min-1
# shift, extent-checked), so stepping to a 26-neighbour never carries between
# fields and is plain int64 arithmetic on the packed key itself.
_KEY_DELTA = (
    (_OFF26[:, 0] << 42) + (_OFF26[:, 1] << 21) + _OFF26[:, 2]
)  # (26,) int64

# The 13 offsets with a positive key delta (they sort after the voxel itself).
# Their mirror images are the other 13; `_neighbor_mask_full` sets both bits per
# matched pair, so it only ever walks these.
_POS13 = np.flatnonzero(_KEY_DELTA > 0)


def _validate(voxels):
    """Same input contract as `sparsecubes.core.mesh`."""
    if not isinstance(voxels, np.ndarray):
        raise TypeError(f'Expected numpy array, got "{type(voxels)}"')
    elif voxels.ndim != 2:
        raise TypeError(f'Expected 2d numpy array, got "{voxels.ndim}"')
    elif voxels.shape[1] != 3:
        raise TypeError(f'Expected numpy array of shape (N, 3), got "{voxels.shape}"')
    elif voxels.dtype not in INT_DTYPES:
        raise TypeError(f"Expected integer dtype, got {voxels.dtype}")


def _check_extent(vox):
    """Guard the pack() 21-bit-per-axis range (binding constraint is extent)."""
    ext = vox.max(axis=0) - vox.min(axis=0)
    if ext.max() >= _EXTENT_LIMIT:
        ax = int(np.argmax(ext))
        raise ValueError(
            f"Voxel extent along axis {ax} is {int(ext[ax])}, which exceeds the "
            f"packing limit ({_EXTENT_LIMIT}). Downsample (e.g. `voxels // k`) or "
            f"split the volume before thinning."
        )


def _build_key_set(vox):
    """Sorted packed keys of the (already unique) voxel set, plus the shift.

    The ``min - 1`` shift keeps every coordinate (and its -1 neighbours)
    non-negative, exactly as `make_surface_nets` does before packing.
    """
    shift = vox.min(axis=0) - 1
    keys = np.sort(pack(vox - shift))
    return keys, shift


def _neighbor_mask(vox, keys_sorted, shift):
    """For each voxel, a uint32 with bit ``k`` set if ``vox + _OFF26[k]`` occupied."""
    base = vox - shift  # >= 1 on every axis
    nb = base[:, None, :] + _OFF26[None, :, :]  # (N, 26, 3), >= 0
    n = len(vox)
    nb_keys = pack(nb.reshape(-1, 3)).reshape(n, 26)
    pos = np.clip(np.searchsorted(keys_sorted, nb_keys), 0, len(keys_sorted) - 1)
    present = keys_sorted[pos] == nb_keys  # (N, 26) bool
    bits = present.astype(np.uint32) << np.arange(26, dtype=np.uint32)[None, :]
    return np.bitwise_or.reduce(bits, axis=1)  # (N,) uint32


def _neighbor_mask_full(keys_sorted):
    """`_neighbor_mask` for the whole key set at once (rows align with keys).

    ``keys_sorted + _KEY_DELTA[k]`` is still sorted, so membership against
    ``keys_sorted`` is a merge of two sorted runs (numpy's stable argsort
    detects the runs) instead of 26 * N binary searches. Each match pins down
    *both* voxels of a neighbour pair, so only the 13 positive offsets are
    walked and each sets the opposite bit on the partner row.
    """
    n = len(keys_sorted)
    out = np.zeros(n, dtype=np.uint32)
    for k in _POS13:
        q = keys_sorted + _KEY_DELTA[k]
        c = np.concatenate([keys_sorted, q])
        order = np.argsort(c, kind="stable")
        cs = c[order]
        eq = cs[1:] == cs[:-1]  # both runs are unique -> matches are pairs
        # Stable sort puts the keys-element (original index < n) first.
        j = order[1:][eq] - n  # q-row: voxel with a neighbour in direction k
        i = order[:-1][eq]  # keys-row: that neighbour itself
        out[j] |= np.uint32(1) << np.uint32(int(k))
        out[i] |= np.uint32(1) << np.uint32(int(_NEG_BIT[k]))
    return out


def _apply_deletion(keys_sorted, mask, alive, sel):
    """Mark ``sel`` deleted and update the maintained mask incrementally.

    Each deleted voxel ``v`` is removed from the neighbourhood of its occupied
    neighbours ``u = v + off``; every such ``u`` loses the bit for the opposite
    offset. Only the neighbours ``v`` actually has (read straight off ``v``'s own
    maintained mask, which already reflects deletions) are looked up, so the work
    is O(|sel| * degree) and independent of the total voxel count. ``mask`` and
    ``alive`` are edited in place.
    """
    # Which neighbours each deleted voxel has (its mask already excludes absent
    # and previously-deleted ones), so we only search those.
    present = ((mask[sel][:, None] >> np.arange(26, dtype=np.uint32)) & 1).astype(bool)
    alive[sel] = False
    inc = present.ravel()
    if not inc.any():
        return

    nb_keys = (keys_sorted[sel][:, None] + _KEY_DELTA[None, :]).ravel()[inc]
    pos = np.clip(np.searchsorted(keys_sorted, nb_keys), 0, len(keys_sorted) - 1)
    ok = keys_sorted[pos] == nb_keys
    u_pos = pos[ok]
    clear_bit = np.tile(_NEG_BIT, len(sel))[inc][ok]
    # bitwise_and.at applies sequentially, so a neighbour losing several deleted
    # voxels at once has all the corresponding bits cleared.
    np.bitwise_and.at(mask, u_pos, ~(np.uint32(1) << clear_bit))


def _popcount(mask):
    """Number of set bits (foreground 26-neighbours) per uint32 mask."""
    m = mask.astype(np.uint32)
    return (
        _BYTE_POP[m & 0xFF]
        + _BYTE_POP[(m >> 8) & 0xFF]
        + _BYTE_POP[(m >> 16) & 0xFF]
        + _BYTE_POP[(m >> 24) & 0xFF]
    )


def _occ_from_mask(mask):
    """Reconstruct the 3x3x3 boolean occupancy (center excluded) from a mask."""
    occ = np.zeros((3, 3, 3), dtype=bool)
    for k in range(26):
        if (mask >> k) & 1:
            dx, dy, dz = _OFF26[k]
            occ[dx + 1, dy + 1, dz + 1] = True
    return occ


def _components_26(cells):
    """Number of 26-connected components among a set of 3x3x3 grid cells."""
    cells = set(cells)
    seen = set()
    n = 0
    for start in cells:
        if start in seen:
            continue
        n += 1
        stack = [start]
        seen.add(start)
        while stack:
            x, y, z = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        nb = (x + dx, y + dy, z + dz)
                        if nb in cells and nb not in seen:
                            seen.add(nb)
                            stack.append(nb)
    return n


_SIX = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
_FACE_CELLS = [(dx + 1, dy + 1, dz + 1) for dx, dy, dz in _SIX]


def _bg_components_6(occ):
    """Number of 6-connected background components in N18 that touch a face of p."""
    # Background cells within the 18-neighbourhood (faces + edges, no corners).
    bg = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nz = (dx != 0) + (dy != 0) + (dz != 0)
                if nz in (1, 2) and not occ[dx + 1, dy + 1, dz + 1]:
                    bg.add((dx + 1, dy + 1, dz + 1))
    seeds = {c for c in _FACE_CELLS if c in bg}
    if not seeds:
        return 0
    seen = set()
    n = 0
    for start in bg:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        has_seed = False
        while stack:
            c = stack.pop()
            if c in seeds:
                has_seed = True
            for dx, dy, dz in _SIX:
                nb = (c[0] + dx, c[1] + dy, c[2] + dz)
                if nb in bg and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if has_seed:
            n += 1
    return n


def _is_simple_config(mask):
    """Malandain-Bertrand (26, 6) simple-point test for one neighbourhood mask."""
    occ = _occ_from_mask(mask)
    fg = [tuple(c) for c in np.argwhere(occ)]
    if _components_26(fg) != 1:  # T26 must be exactly one foreground component
        return False
    return _bg_components_6(occ) == 1  # T6 must be exactly one background component


# --- vectorised simple-point test -----------------------------------------
# Fixed adjacency among the 26 neighbour positions and their N18 / face flags,
# used to evaluate the Malandain-Bertrand test for a whole batch of masks with
# bit-parallel flood fills instead of a per-mask Python BFS.
_DIFF = _OFF26[:, None, :] - _OFF26[None, :, :]  # (26, 26, 3)
_ADJ26 = np.abs(_DIFF).max(axis=2) == 1  # 26-adjacency (Chebyshev distance 1)
_ADJ6 = np.abs(_DIFF).sum(axis=2) == 1  # 6-adjacency (Manhattan distance 1)
_NNZ = (_OFF26 != 0).sum(axis=1)
_IS_N18 = _NNZ <= 2  # faces + edges (corners excluded)
_IS_FACE = _NNZ == 1  # the six face neighbours

# The same adjacency/position sets as 26-bit masks: bit j of _NB26[k] says
# positions k and j are 26-adjacent (ditto _NB6 for 6-adjacency).
_NB26 = np.array(
    [sum(1 << j for j in range(26) if _ADJ26[k, j]) for k in range(26)],
    dtype=np.uint32,
)
_NB6 = np.array(
    [sum(1 << j for j in range(26) if _ADJ6[k, j]) for k in range(26)],
    dtype=np.uint32,
)
_M18 = np.uint32(sum(1 << j for j in range(26) if _IS_N18[j]))
_MFACE = np.uint32(sum(1 << j for j in range(26) if _IS_FACE[j]))


def _flood(start, domain, nb):
    """Bit-parallel flood fill of ``start`` within ``domain`` (both (K,) uint32).

    Each element is one 3x3x3 neighbourhood; its 26 cells are bits. One
    iteration ORs every active cell's adjacency mask (``nb``) into the
    component, clipped to ``domain``, across the whole batch at once.
    """
    one = np.uint32(1)
    comp = start.copy()
    while True:
        grow = comp.copy()
        for k in range(26):
            grow |= ((comp >> np.uint32(k)) & one) * nb[k]
        grow &= domain
        if np.array_equal(grow, comp):
            return comp
        comp = grow


def _simple_batch(masks):
    """Vectorised Malandain-Bertrand (26, 6) simple-point test for many masks."""
    occ = masks.astype(np.uint32)
    one = np.uint32(1)

    # T26 == 1: the foreground is non-empty and a flood from its lowest set bit
    # reaches every set bit (i.e. it is a single 26-connected component).
    low = occ & (~occ + one)
    result = (occ != 0) & (_flood(low, occ, _NB26) == occ)
    if not result.any():
        return result

    # T6 == 1: the background (in N18) has a face cell, and one 6-flood from the
    # lowest face cell covers all face cells (a single seeded component). Only
    # masks that already passed T26 can be simple, so restrict to those rows.
    occ_c = occ[result]
    bg = ~occ_c & _M18
    seeds = bg & _MFACE
    comp = _flood(seeds & (~seeds + one), bg, _NB6)
    result[result] = (seeds != 0) & ((seeds & ~comp) == 0)
    return result


def _is_simple(masks):
    """Simple-point test for an array of 26-bit neighbourhood masks."""
    masks = np.asarray(masks, dtype=np.uint32)
    if masks.size == 0:
        return np.zeros(0, dtype=bool)
    return _simple_batch(masks)


def thin(
    voxels,
    *,
    preserve_endpoints=True,
    max_iterations=None,
    unique_check=True,
    fill_cavities=False,
    verbose=False,
):
    """Thin sparse voxels to a one-voxel-wide medial curve (centerline).

    Sparse, vectorised 3D topological thinning. Operates directly on the
    ``(N, 3)`` coordinate set - no dense grid is ever allocated, so memory
    scales with the current voxel count rather than the bounding-box volume.

    The result is a strict subset of the input coordinates (thinning only ever
    deletes voxels), preserves the object's topology (connected components and
    tunnels/loops), and preserves curve endpoints so branches are not eroded
    away. It is *not* a smoothed/centered TEASAR skeleton - see the module and
    README notes for when to prefer densify+scikit-image or kimimaro instead.

    Parameters
    ----------
    voxels :            (N, 3) array of integers
                        XYZ voxel coordinates. The result keeps this dtype.
    preserve_endpoints : bool, optional
                        If True (default), never delete endpoints (voxels with a
                        single foreground 26-neighbour). This yields a medial
                        *curve*. If False, endpoints are also erodible.
    max_iterations :    int, optional
                        Cap on the number of full passes. ``None`` (default)
                        runs to convergence (a stable, fully-thinned set).
    unique_check :      bool, optional
                        If True (default), remove duplicates from the input
                        before thinning. If False, the input is assumed unique.
    fill_cavities :     bool, optional
                        If True, fill enclosed background voids before thinning
                        (`sparsecubes.fill_cavities`, exact mode). Thinning
                        preserves such voids, so they otherwise leave thick,
                        un-thinnable "blobs" on the centerline. Recommended for
                        real segmentation data; adds roughly one thinning pass of
                        work and needs scipy (or `dijkstra3d_sparse`). Default
                        False. Call `fill_cavities` directly to tune its depth.
    verbose :           bool
                        If True, log per-pass progress.

    Returns
    -------
    (M, 3) array of the same integer dtype as `voxels`.
    """
    _validate(voxels)
    out_dtype = voxels.dtype

    if fill_cavities:
        voxels = _fill_cavities(voxels, verbose=verbose)

    vox = voxels.astype(np.int64)
    if len(vox) == 0:
        return vox.astype(out_dtype)
    _check_extent(vox)

    # Sort by packed key with a fixed shift so that deleting voxels (a boolean
    # mask) keeps `keys` sorted without ever re-sorting. Dedup and sort are one
    # pass over the packed keys (much cheaper than a row-wise unique). The
    # 26-neighbour mask is then built once and maintained incrementally on
    # deletion, so per-pass work scales with the (shrinking) surface rather
    # than the whole - mostly interior - voxel set.
    shift = vox.min(axis=0) - 1
    keys = pack(vox - shift)
    if unique_check:
        keys = np.unique(keys)
    else:
        keys = np.sort(keys)
    vox = unpack(keys) + shift
    mask = _neighbor_mask_full(keys)
    parity = (((vox[:, 0] & 1) << 2) | ((vox[:, 1] & 1) << 1) | (vox[:, 2] & 1)).astype(
        np.uint8
    )

    it = 0
    changed = True
    while changed:
        if max_iterations is not None and it >= max_iterations:
            break
        changed = False
        alive = np.ones(len(vox), dtype=bool)
        for d in range(6):
            bit = np.uint32(int(_BIT_OF_OFF6[d]))
            # Border in direction d = the maintained mask has that face bit clear.
            # Interior voxels (all 26 bits set) are never borders, so they cost
            # nothing here. Computed once per direction (border is monotonic under
            # deletion); each sub-field's candidates are a distinct parity class,
            # untouched by the other sub-fields' deletions.
            border_idx = np.flatnonzero((((mask >> bit) & np.uint32(1)) == 0) & alive)
            if len(border_idx) == 0:
                continue
            bpar = parity[border_idx]
            for sf in range(8):
                cand = border_idx[bpar == sf]
                if len(cand) == 0:
                    continue
                m = mask[cand]  # up-to-date: prior sub-fields patched it in place
                if preserve_endpoints:
                    # Cheap degree filter first, so the simple-point flood only
                    # runs on non-endpoints.
                    keep = _popcount(m) != 1
                    cand, m = cand[keep], m[keep]
                sel = cand[_is_simple(m)]
                if len(sel):
                    _apply_deletion(keys, mask, alive, sel)
                    changed = True
        if changed:
            vox, keys = vox[alive], keys[alive]
            mask, parity = mask[alive], parity[alive]
        it += 1
        log(f"Thinning pass {it}: {len(vox)} voxels remain.", verbose=verbose)

    return vox.astype(out_dtype)
