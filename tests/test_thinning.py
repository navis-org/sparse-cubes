import os
import time
import importlib.util

import numpy as np
import pytest

import sparsecubes as sc
from sparsecubes.thinning import (
    _OFF26,
    _BIT_OF_OFF6,
    _OFF6,
    _is_simple_config,
    _neighbor_mask,
    _build_key_set,
)
from _shapes import (
    line,
    l_shape,
    solid_cube,
    slab,
    solid_cylinder,
    annulus,
    betti,
    n_components,
    has_2x2x2_block,
    is_subset,
)

DATA = os.path.join(os.path.dirname(__file__), "10075_coarse.npy")
HAS_SKIMAGE = importlib.util.find_spec("skimage") is not None


# --- structural / primitive tests -----------------------------------------

def test_off26_layout():
    # 26 unique offsets, none is the center, each component in {-1, 0, 1}.
    assert _OFF26.shape == (26, 3)
    assert len(set(map(tuple, _OFF26.tolist()))) == 26
    assert not any((r == 0).all() for r in _OFF26)
    assert set(np.unique(_OFF26).tolist()) == {-1, 0, 1}


def test_bit_of_off6_matches_neighbor_mask():
    # For a single voxel with exactly one face neighbour occupied, the mask must
    # have precisely the bit that _BIT_OF_OFF6 assigns to that direction.
    for d, off in enumerate(_OFF6):
        vox = np.array([[5, 5, 5], [5, 5, 5] + off], dtype=np.int64)
        keys, shift = _build_key_set(vox)
        mask = _neighbor_mask(vox, keys, shift)
        # Row 0 is the center voxel; its only neighbour is in direction d.
        bit = int(_BIT_OF_OFF6[d])
        assert (mask[0] >> bit) & 1 == 1
        assert mask[0] == np.uint32(1) << np.uint32(bit)


def test_simple_point_examples():
    def mask_of(offsets):
        m = 0
        for off in offsets:
            k = int(np.flatnonzero((_OFF26 == off).all(axis=1))[0])
            m |= 1 << k
        return m

    # Isolated voxel (no neighbours): deleting removes a component -> not simple.
    assert not _is_simple_config(mask_of([]))
    # Straight line through p (two opposite neighbours, not 26-adjacent): the
    # two neighbours are separate components -> not simple (interior preserved).
    assert not _is_simple_config(mask_of([(1, 0, 0), (-1, 0, 0)]))
    # A flat face: p on the surface of a slab (one empty face, 25 neighbours) is
    # a classic simple point.
    all_but_plusx = [tuple(o) for o in _OFF26 if o[0] != 1]
    assert _is_simple_config(mask_of(all_but_plusx))


def test_simple_batch_matches_scalar_reference():
    # The vectorised simple-point test must agree with the scalar Malandain
    # reference on a large random sample of 26-bit neighbourhood configs.
    from sparsecubes.thinning import _simple_batch

    rng = np.random.default_rng(0)
    masks = rng.integers(0, 1 << 26, size=4000, dtype=np.uint64).astype(np.uint32)
    vec = _simple_batch(masks)
    ref = np.array([_is_simple_config(int(m)) for m in masks])
    assert np.array_equal(vec, ref)


# --- thinning invariants ---------------------------------------------------

SHAPES = {
    "line": line(10),
    "l_shape": l_shape(),
    "cube3": solid_cube(3),
    "cube4": solid_cube(4),
    "slab": slab(7),
    "cylinder": solid_cylinder(4, 16),
    "annulus": annulus(9, 5),
}


@pytest.mark.parametrize("name", list(SHAPES))
def test_thin_is_subset(name):
    v = SHAPES[name]
    t = sc.thin(v)
    assert is_subset(t, v)
    assert t.dtype == v.dtype


@pytest.mark.parametrize("name", list(SHAPES))
def test_thin_one_voxel_wide(name):
    # No fully occupied 2x2x2 block may survive a medial-curve thinning.
    t = sc.thin(SHAPES[name])
    assert not has_2x2x2_block(t)


@pytest.mark.parametrize("name", list(SHAPES))
def test_thin_idempotent(name):
    t = sc.thin(SHAPES[name])
    tt = sc.thin(t)
    assert np.array_equal(np.unique(t, axis=0), np.unique(tt, axis=0))


@pytest.mark.parametrize("name", list(SHAPES))
def test_thin_preserves_components(name):
    # The fundamental topology guarantee of simple-point thinning: the number of
    # connected components is invariant, and a non-empty connected object never
    # thins away to nothing.
    v = SHAPES[name]
    t = sc.thin(v)
    assert len(t) > 0
    assert n_components(t) == n_components(v)


def test_thin_line_is_fixed_point():
    v = line(10)
    assert np.array_equal(np.unique(sc.thin(v), axis=0), np.unique(v, axis=0))


def test_thin_cube_not_emptied():
    # Symmetric solid cubes must not vanish (they stay one connected component).
    for k in (3, 4, 5):
        t = sc.thin(solid_cube(k))
        assert len(t) > 0
        assert n_components(t) == 1


def test_thin_annulus_keeps_loop():
    # A ring must retain its hole: the thinned graph has first Betti number >= 1.
    t = sc.thin(annulus(9, 5))
    from sparsecubes.skeleton import _edges_26

    nodes, edges = _edges_26(t.astype(np.int64))
    _, b1 = betti(nodes, edges)
    assert b1 >= 1


def test_thin_cylinder_is_a_curve():
    # A solid cylinder collapses to a single 1-wide centerline: one component and
    # far fewer voxels than the solid.
    v = solid_cylinder(4, 16)
    t = sc.thin(v)
    assert n_components(t) == 1
    assert len(t) < len(v) / 5


def test_thin_empty_and_single():
    empty = np.empty((0, 3), dtype=np.uint32)
    assert sc.thin(empty).shape == (0, 3)
    single = np.array([[3, 3, 3]], dtype=np.uint32)
    assert np.array_equal(sc.thin(single), single)


def test_thin_max_iterations():
    # Capping iterations stops early: the result is a (non-strict) superset of the
    # fully converged skeleton but still a subset of the input.
    v = solid_cylinder(4, 16)
    partial = sc.thin(v, max_iterations=1)
    full = sc.thin(v)
    assert is_subset(full, partial) or len(partial) >= len(full)
    assert is_subset(partial, v)


def test_thin_extent_guard():
    # A coordinate range beyond pack()'s per-axis budget raises a clear error.
    v = np.array([[0, 0, 0], [2**21, 0, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="extent"):
        sc.thin(v)


def test_thin_bad_input():
    with pytest.raises(TypeError):
        sc.thin([[1, 2, 3]])
    with pytest.raises(TypeError):
        sc.thin(np.zeros((4, 2), dtype=np.int64))
    with pytest.raises(TypeError):
        sc.thin(np.zeros((4, 3), dtype=np.float64))


# --- cross-check against scikit-image (topology only) ----------------------

@pytest.mark.skipif(not HAS_SKIMAGE, reason="scikit-image not installed")
@pytest.mark.parametrize("name", ["line", "l_shape", "cylinder", "annulus"])
def test_thin_matches_skimage_topology(name):
    from skimage.morphology import skeletonize

    v = SHAPES[name]
    # Rasterise into a padded dense grid (skimage needs a background border).
    shift = v.min(axis=0) - 1
    vs = v - shift
    dims = vs.max(axis=0) + 2
    grid = np.zeros(tuple(dims), dtype=bool)
    grid[vs[:, 0], vs[:, 1], vs[:, 2]] = True

    ski = skeletonize(grid, method="lee").astype(bool)
    ski_vox = np.argwhere(ski)

    ours = sc.thin(v).astype(np.int64) - shift

    # Both preserve the object's component count; compare that invariant rather
    # than exact voxels (sub-field order differs from skimage's raster order).
    assert n_components(ours) == n_components(ski_vox) == n_components(vs)


# --- performance / memory regression on the real cloud ---------------------

@pytest.mark.skipif(not os.path.exists(DATA), reason="test voxel cloud not available")
def test_thin_no_perf_regression():
    voxels = np.load(DATA)

    t0 = time.perf_counter()
    thinned = sc.thin(voxels)
    elapsed = time.perf_counter() - t0

    assert len(thinned) > 0
    assert is_subset(thinned, voxels)
    assert not has_2x2x2_block(thinned)
    assert n_components(thinned) == n_components(voxels)
    # Absolute ceiling - the vectorised thinner does this ~40k-voxel cloud in
    # well under a second locally; this guards against a return to the old
    # per-voxel / per-subfield full-neighbourhood cost without CI flakiness.
    assert elapsed < 30
