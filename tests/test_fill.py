"""Tests for `fill_cavities` and the `thin(..., fill_cavities=True)` path."""

import importlib.util
import os

import numpy as np
import pytest

import sparsecubes as sc
from sparsecubes.core import fill_cavities
from _shapes import (
    solid_cube,
    hollow_box,
    box_with_tunnel,
    n_components,
    has_2x2x2_block,
    is_subset,
)

# Exact mode needs a connected-components backend (scipy or dijkstra3d_sparse).
_HAS_CC = (
    importlib.util.find_spec("scipy") is not None
    or importlib.util.find_spec("dijkstra3d_sparse") is not None
)
exact_only = pytest.mark.skipif(not _HAS_CC, reason="needs scipy or dijkstra3d_sparse")


def _as_set(a):
    return set(map(tuple, np.asarray(a).tolist()))


# --- exact mode: correctness ----------------------------------------------

@exact_only
def test_solid_cube_is_noop():
    cube = solid_cube(5)
    out = fill_cavities(cube)
    assert _as_set(out) == _as_set(cube)


@exact_only
def test_single_void_restores_solid():
    # hollow_box(5, 1) is a solid cube minus its single center voxel.
    shell = hollow_box(outer=5, void=1)
    out = fill_cavities(shell)
    assert _as_set(out) == _as_set(solid_cube(5))
    assert is_subset(shell, out)  # only ever adds cells


@exact_only
def test_multivoxel_void_fully_filled():
    shell = hollow_box(outer=7, void=3)  # 27 enclosed cells
    out = fill_cavities(shell)
    assert _as_set(out) == _as_set(solid_cube(7))


@exact_only
def test_through_tunnel_preserved():
    # A straight tunnel opens to the exterior at both ends -> not enclosed.
    tunnel = box_with_tunnel(5)
    out = fill_cavities(tunnel)
    assert _as_set(out) == _as_set(tunnel)


@exact_only
def test_dtype_and_superset_preserved():
    shell = hollow_box(7, 3).astype(np.uint32)
    out = fill_cavities(shell)
    assert out.dtype == np.uint32
    assert is_subset(shell, out)
    assert len(out) > len(shell)


@exact_only
def test_empty_input():
    empty = np.empty((0, 3), dtype=np.int32)
    out = fill_cavities(empty)
    assert out.shape == (0, 3) and out.dtype == np.int32


# --- exact mode: topology preservation ------------------------------------

@exact_only
def test_preserves_components_and_fills_void():
    # A hollow box (its void must fill) plus a far-away separate cube: the object
    # must stay two components and the void must be filled.
    two = np.vstack([hollow_box(7, 3), solid_cube(3) + np.array([20, 0, 0])]).astype(
        np.int64
    )
    out = fill_cavities(two)
    assert n_components(out) == n_components(two) == 2
    assert len(out) == len(two) + 27  # the 3x3x3 void filled


# --- exact mode: the max_depth knob ---------------------------------------

@exact_only
def test_max_depth_gates_thick_voids():
    # hollow_box(11, 7): a 7^3 void with inradius 3, so it fills iff max_depth > 3.
    # Filling is all-or-nothing per void (a void touching the cut-off frontier is
    # treated as exterior), so a too-shallow depth adds nothing rather than a
    # partial fill.
    shell = hollow_box(outer=11, void=7)
    assert len(fill_cavities(shell, max_depth=3)) == len(shell)
    assert _as_set(fill_cavities(shell, max_depth=4)) == _as_set(solid_cube(11))


# --- integration with thin() / skeletonize() ------------------------------

@exact_only
def test_thin_flag_matches_manual_fill():
    shell = hollow_box(7, 3)
    a = sc.thin(shell, fill_cavities=True)
    b = sc.thin(fill_cavities(shell))
    assert np.array_equal(np.unique(a, axis=0), np.unique(b, axis=0))


@exact_only
def test_thin_fill_removes_blob():
    shell = hollow_box(7, 3)
    plain = sc.thin(shell)
    filled = sc.thin(shell, fill_cavities=True)
    # The enclosed void (b2) can't collapse: thinning the shell leaves a whole
    # residual surface. Filling first lets it thin to a compact medial curve -
    # far fewer voxels - while staying a clean one-voxel-wide subset of the solid.
    assert len(filled) < len(plain)
    assert not has_2x2x2_block(filled)
    assert is_subset(filled, solid_cube(7))


@exact_only
def test_thin_skeletonize_fill_flag_runs():
    sk = sc.thin_skeletonize(hollow_box(7, 3), fill_cavities=True)
    assert len(sk.nodes) > 0
    assert n_components(sc.thin(hollow_box(7, 3), fill_cavities=True)) == 1


# --- closing mode ---------------------------------------------------------

def test_closing_fills_single_void():
    shell = hollow_box(5, 1)
    out = fill_cavities(shell, mode="closing")
    assert _as_set(solid_cube(5)) <= _as_set(out)


@exact_only
def test_closing_can_fuse_where_exact_does_not():
    # Two 3x3-thick bars one empty plane apart. Exact never fuses; radius-1
    # closing bridges the gap and welds them into one component.
    def bar(x0):
        g = np.indices((3, 3, 10)).reshape(3, -1).T
        g[:, 0] += x0
        return g

    bars = np.vstack([bar(0), bar(4)]).astype(np.int64)  # gap plane at x == 3
    assert n_components(bars) == 2
    assert n_components(fill_cavities(bars, mode="exact")) == 2
    assert n_components(fill_cavities(bars, mode="closing")) == 1


# --- integration on the real voxel cloud ----------------------------------

DATA = os.path.join(os.path.dirname(__file__), "10075_coarse.npy")


@exact_only
@pytest.mark.skipif(not os.path.exists(DATA), reason="test voxel cloud not available")
def test_real_cloud_fill_is_topology_safe():
    voxels = np.load(DATA)
    filled = fill_cavities(voxels)
    # Fills real enclosed voids, only ever adds cells, and preserves components.
    assert len(filled) > len(voxels)
    assert is_subset(voxels, filled)
    assert n_components(filled) == n_components(voxels)
    # Thinning the filled cloud stays a valid one-voxel-wide subset.
    thinned = sc.thin(voxels, fill_cavities=True)
    assert not has_2x2x2_block(thinned)
    assert is_subset(thinned, filled)


# --- input validation -----------------------------------------------------

def test_bad_mode_raises():
    with pytest.raises(ValueError):
        fill_cavities(solid_cube(3), mode="nope")


def test_bad_shape_raises():
    with pytest.raises(TypeError):
        fill_cavities(np.zeros((4, 2), dtype=np.int64))
