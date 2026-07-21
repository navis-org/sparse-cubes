"""Tests for `sparsecubes.edges` - voxel adjacency as an explicit edge list.

The contract is checked against an O(N^2) coordinate-distance oracle: for every
pair of voxels, are they within the Chebyshev/L1 bounds that define 6-, 18- and
26-connectivity? The packed-key implementation has to agree exactly, including
which rows the returned indices refer to.
"""

import numpy as np
import pytest

import sparsecubes as sc
from sparsecubes.graph import _edge_pairs
from sparsecubes._keys import to_keys

from _shapes import (
    line,
    l_shape,
    y_branch,
    solid_cube,
    slab,
    solid_cylinder,
    annulus,
    hollow_box,
    n_components,
    _union_find,
)

CONNECTIVITIES = [6, 18, 26]

SHAPES = {
    "line": line(10),
    "l_shape": l_shape(),
    "y_branch": y_branch(7),
    "cube3": solid_cube(3),
    "cube5": solid_cube(5),
    "slab": slab(6),
    "cylinder": solid_cylinder(4, 20),
    "annulus": annulus(9, 5),
    "hollow_box": hollow_box(7, 3),
}


def _oracle(voxels, connectivity):
    """Brute-force adjacency of the deduplicated voxels: set of (lo, hi) pairs."""
    u = np.unique(np.asarray(voxels, dtype=np.int64), axis=0)
    d = np.abs(u[:, None, :] - u[None, :, :])
    touching = d.max(axis=2) == 1
    l1 = d.sum(axis=2)
    limit = {6: 1, 18: 2, 26: 3}[connectivity]
    adj = touching & (l1 <= limit)
    return u, set(map(tuple, np.stack(np.nonzero(np.triu(adj, 1)), axis=1)))


def _as_pairs(e):
    return set(map(tuple, np.asarray(e).tolist()))


# --- adjacency vs. the brute-force oracle ----------------------------------


@pytest.mark.parametrize("connectivity", CONNECTIVITIES)
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_edges_match_oracle(name, connectivity):
    voxels = SHAPES[name]
    nodes, e = sc.edges(voxels, connectivity=connectivity)
    exp_nodes, exp_edges = _oracle(voxels, connectivity)

    assert np.array_equal(nodes, exp_nodes)
    assert _as_pairs(e) == exp_edges


@pytest.mark.parametrize("connectivity", CONNECTIVITIES)
def test_edges_match_oracle_random(connectivity):
    """Random clouds cover the sparse, non-convex cases the shapes do not."""
    rng = np.random.RandomState(0)
    voxels = rng.randint(0, 14, (600, 3)).astype(np.int32)
    nodes, e = sc.edges(voxels, connectivity=connectivity)
    exp_nodes, exp_edges = _oracle(voxels, connectivity)

    assert np.array_equal(nodes, exp_nodes)
    assert _as_pairs(e) == exp_edges


def test_connectivity_is_nested():
    """6-adjacency is a subset of 18, which is a subset of 26."""
    voxels = solid_cylinder(4, 12)
    e6, e18, e26 = (_as_pairs(sc.edges(voxels, connectivity=c)[1]) for c in CONNECTIVITIES)
    assert e6 < e18 < e26


# --- output contract -------------------------------------------------------


@pytest.mark.parametrize("connectivity", CONNECTIVITIES)
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_edges_are_canonical_unique_and_sorted(name, connectivity):
    _, e = sc.edges(SHAPES[name], connectivity=connectivity)
    assert e.dtype == np.int64
    assert e.shape[1] == 2
    assert (e[:, 0] < e[:, 1]).all(), "edges must be canonical (lo < hi)"
    assert len(np.unique(e, axis=0)) == len(e), "edges must be deduplicated"
    # Lexicographically sorted by (src, dst) - a deterministic order.
    assert np.array_equal(e, e[np.lexsort((e[:, 1], e[:, 0]))])


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64, np.uint16])
def test_nodes_keep_input_dtype_and_are_deduplicated(dtype):
    voxels = np.concatenate([solid_cube(3), solid_cube(3)]).astype(dtype)
    nodes, e = sc.edges(voxels)
    assert nodes.dtype == dtype
    assert len(nodes) == len(np.unique(voxels, axis=0)) == 27
    assert len(e) and e.max() < len(nodes)


def test_edge_indices_reference_returned_nodes():
    """The indices index `nodes`, and every pair really is 26-adjacent there."""
    voxels = y_branch(6)
    nodes, e = sc.edges(voxels)
    d = np.abs(nodes[e[:, 0]].astype(np.int64) - nodes[e[:, 1]].astype(np.int64))
    assert (d.max(axis=1) == 1).all()


def test_negative_coordinates():
    """Coordinates are shifted internally; negatives must round-trip unchanged."""
    voxels = solid_cube(3) - 50
    nodes, e = sc.edges(voxels)
    exp_nodes, exp_edges = _oracle(voxels, 26)
    assert np.array_equal(nodes, exp_nodes)
    assert _as_pairs(e) == exp_edges


# --- degenerate input ------------------------------------------------------


def test_empty():
    nodes, e = sc.edges(np.empty((0, 3), dtype=np.int64))
    assert nodes.shape == (0, 3)
    assert e.shape == (0, 2) and e.dtype == np.int64


def test_single_voxel():
    nodes, e = sc.edges(np.array([[3, 4, 5]]))
    assert np.array_equal(nodes, [[3, 4, 5]])
    assert e.shape == (0, 2)


def test_isolated_voxels_have_no_edges():
    voxels = np.array([[0, 0, 0], [10, 10, 10], [20, 20, 20]])
    nodes, e = sc.edges(voxels)
    assert len(nodes) == 3
    assert e.shape == (0, 2)


def test_rejects_bad_connectivity():
    with pytest.raises(ValueError, match="connectivity"):
        sc.edges(solid_cube(2), connectivity=8)


def test_rejects_non_integer():
    with pytest.raises(TypeError):
        sc.edges(solid_cube(2).astype(float))


# --- consistency with the rest of the library ------------------------------


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_agrees_with_downsample_graph_at_factor_one(name):
    """`downsample_graph(v, 1)` is this primitive - navis relies on that today."""
    voxels = SHAPES[name]
    nodes, e = sc.edges(voxels, connectivity=26)
    coarse, coarse_edges = sc.downsample_graph(voxels, 1)
    assert np.array_equal(nodes, coarse)
    assert _as_pairs(e) == _as_pairs(coarse_edges)


@pytest.mark.parametrize("connectivity", [6, 26])
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_components_match_measure(name, connectivity):
    """Component count over the edge list must match `measure` on the same set."""
    pytest.importorskip("scipy")
    voxels = SHAPES[name]
    nodes, e = sc.edges(voxels, connectivity=connectivity)
    n_comp, _ = sc.measure.connected_components(nodes, connectivity=connectivity)
    assert _union_find(len(nodes), e) == n_comp


def test_disconnected_shapes_stay_disconnected():
    """Two clouds a two-voxel gap apart share no edge, whatever the connectivity."""
    voxels = np.concatenate([solid_cube(3), solid_cube(3) + [5, 0, 0]])
    for connectivity in CONNECTIVITIES:
        nodes, e = sc.edges(voxels, connectivity=connectivity)
        assert _union_find(len(nodes), e) == 2
        assert n_components(voxels) == 2


def test_edge_pairs_is_symmetric_under_reflection():
    """Adjacency is a geometric property: mirroring the cloud preserves it."""
    voxels = l_shape()
    nodes, e = sc.edges(voxels)
    m_nodes, m_e = sc.edges(-voxels)
    # Reflection reverses the sort order, so index i maps to len-1-i.
    flipped = np.sort(len(nodes) - 1 - e, axis=1)
    assert _as_pairs(flipped) == _as_pairs(m_e)


def test_edge_pairs_helper_needs_sorted_keys():
    """The private helper's precondition, pinned so callers keep honouring it."""
    keys, _ = to_keys(solid_cube(3), margin=1)
    assert np.array_equal(keys, np.sort(keys))
    e = _edge_pairs(keys, 26)
    assert (e[:, 0] < e[:, 1]).all()
    # `sort=False` returns the same set, just grouped by neighbour offset.
    assert _as_pairs(e) == _as_pairs(_edge_pairs(keys, 26, sort=False))
