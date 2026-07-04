"""Tests for connectivity-safe downsampling (`downsample_graph`) and its
integration with the skeletonizers.

The whole point of `downsample_graph` is that it downsamples *without* connecting
voxels that were not previously adjacent - unlike the naive ``unique(voxels //
f)`` pool, which fuses structures a sub-``2*f`` gap apart. These tests pin that
guarantee: the coarse edge list defines connectivity (a strict subset of the
coarse coordinates' geometric adjacency), and the connected-component partition
is preserved exactly, end to end through TEASAR.
"""

import numpy as np
import pytest

import sparsecubes as sc
from sparsecubes.downsample import downsample_graph
from sparsecubes.skeleton import _edges_26

from _shapes import (
    line,
    l_shape,
    y_branch,
    solid_cube,
    slab,
    solid_cylinder,
    annulus,
    betti,
    n_components,
)


SHAPES = {
    "line": line(10),
    "l_shape": l_shape(),
    "y_branch": y_branch(7),
    "cube3": solid_cube(3),
    "cube5": solid_cube(5),
    "slab": slab(6),
    "cylinder": solid_cylinder(4, 20),
    "annulus": annulus(9, 5),
}


def two_lines(gap=3, n=8):
    """Two parallel x-lines a `gap`-voxel gap apart in y (two components).

    With `gap` >= 2 they are never 26-adjacent, but ``// 2`` maps y=0 -> 0 and
    y=gap -> gap//2, which for gap in {2, 3} lands them on adjacent coarse rows -
    exactly the spurious fusion `downsample_graph` must avoid.
    """
    a = line(n, axis=0)
    b = line(n, axis=0) + np.array([0, gap, 0])
    return np.vstack([a, b]).astype(np.int64)


def two_cubes(offset=4, k=3):
    """Two solid k-cubes a gap apart in x (two components, fused by ``// 2``)."""
    a = solid_cube(k)
    b = solid_cube(k) + np.array([offset, 0, 0])
    return np.vstack([a, b]).astype(np.int64)


def _edge_coords(coords, edges):
    """Undirected edges as a set of sorted coordinate-tuple pairs (order-free)."""
    out = set()
    for a, b in np.asarray(edges):
        pa = tuple(int(x) for x in coords[a])
        pb = tuple(int(x) for x in coords[b])
        out.add((pa, pb) if pa <= pb else (pb, pa))
    return out


# --- basic contract --------------------------------------------------------

def test_returns_coords_and_edges():
    coords, edges = downsample_graph(line(10), 2)
    assert coords.ndim == 2 and coords.shape[1] == 3
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert coords.dtype == line(10).dtype
    assert edges.dtype == np.int64


# --- the headline: no spurious fusion --------------------------------------

@pytest.mark.parametrize("gap", [2, 3])
def test_parallel_lines_not_fused(gap):
    vox = two_lines(gap=gap)
    assert n_components(vox) == 2  # sanity: two real components

    # Naive block-pooling fuses them into one.
    naive = np.unique(vox // 2, axis=0)
    assert n_components(naive) == 1

    # downsample_graph keeps them apart.
    coords, edges = downsample_graph(vox, 2)
    assert betti(coords, edges)[0] == 2


def test_two_cubes_not_fused():
    vox = two_cubes(offset=4)
    assert n_components(vox) == 2
    assert n_components(np.unique(vox // 2, axis=0)) == 1  # naive fuses
    coords, edges = downsample_graph(vox, 2)
    assert betti(coords, edges)[0] == 2


# --- component preservation across shapes ----------------------------------

@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("factor", [2, 3])
def test_components_preserved(name, factor):
    vox = SHAPES[name]
    coords, edges = downsample_graph(vox, factor)
    assert betti(coords, edges)[0] == n_components(vox)
    # coarse cloud is strictly smaller (these shapes span more than `factor`).
    assert len(coords) <= len(np.unique(vox, axis=0))


def test_composite_components_preserved():
    vox = two_cubes(offset=4)
    for factor in (2, 3):
        coords, edges = downsample_graph(vox, factor)
        assert betti(coords, edges)[0] == 2


# --- Property P: edges define adjacency, subset of geometry -----------------

@pytest.mark.parametrize("name", list(SHAPES))
def test_edges_are_chebyshev_1_no_self_loops(name):
    coords, edges = downsample_graph(SHAPES[name], 2)
    if len(edges) == 0:
        return
    assert (edges[:, 0] != edges[:, 1]).all()  # no self-edges
    cheb = np.abs(coords[edges[:, 0]] - coords[edges[:, 1]]).max(axis=1)
    assert (cheb == 1).all()


@pytest.mark.parametrize("name", list(SHAPES))
def test_edges_subset_of_geometric(name):
    coords, edges = downsample_graph(SHAPES[name], 2)
    geo_nodes, geo_edges = _edges_26(coords)
    lifted = _edge_coords(coords, edges)
    geometric = _edge_coords(geo_nodes, geo_edges)
    assert lifted <= geometric


def test_edges_strict_subset_on_fusion_fixture():
    # The coarse geometry has the spurious cross-line edge; the lifted graph
    # does not -> lifted is a *proper* subset.
    vox = two_lines(gap=3)
    coords, edges = downsample_graph(vox, 2)
    geo_nodes, geo_edges = _edges_26(coords)
    lifted = _edge_coords(coords, edges)
    geometric = _edge_coords(geo_nodes, geo_edges)
    assert lifted < geometric


# --- factor == 1 is an identity --------------------------------------------

@pytest.mark.parametrize("name", list(SHAPES))
def test_factor_one_identity(name):
    vox = SHAPES[name]
    coords, edges = downsample_graph(vox, 1)
    assert set(map(tuple, coords.tolist())) == set(map(tuple, np.unique(vox, axis=0).tolist()))
    fine_nodes, fine_edges = _edges_26(vox.astype(np.int64))
    assert _edge_coords(coords, edges) == _edge_coords(fine_nodes, fine_edges)


# --- edge cases ------------------------------------------------------------

def test_empty():
    coords, edges = downsample_graph(np.empty((0, 3), dtype=np.int64), 2)
    assert coords.shape == (0, 3)
    assert edges.shape == (0, 2)


def test_dtype_preserved():
    vox = line(10).astype(np.int32)
    coords, _ = downsample_graph(vox, 2)
    assert coords.dtype == np.int32


def test_negative_coordinates():
    vox = two_lines(gap=3) - np.array([50, 50, 50])
    coords, edges = downsample_graph(vox, 2)
    assert betti(coords, edges)[0] == 2


def test_anisotropic_factor():
    # Downsample only along x; y-separated lines must stay separate regardless.
    vox = two_lines(gap=3)
    coords, edges = downsample_graph(vox, (2, 1, 1))
    assert betti(coords, edges)[0] == 2


def test_invalid_factor():
    with pytest.raises(ValueError):
        downsample_graph(line(10), 0)
    with pytest.raises(ValueError):
        downsample_graph(line(10), (2, 2))  # wrong length


# --- integration with teasar_skeletonize -----------------------------------

def test_teasar_edges_injection_keeps_components():
    vox = two_cubes(offset=4)
    coords, edges = downsample_graph(vox, 2)
    sk = sc.teasar_skeletonize(coords, edges=edges)
    assert betti(sk.nodes, sk.edges)[0] == 2


def test_teasar_downsample_convenience_keeps_components():
    vox = two_cubes(offset=4)
    sk = sc.teasar_skeletonize(vox, downsample=2)
    assert betti(sk.nodes, sk.edges)[0] == 2


def test_teasar_naive_prepool_fuses_documenting_the_bug():
    # Contrast: pre-pooling the plain cloud fuses the two cubes into one tree.
    vox = two_cubes(offset=4)
    naive = np.unique(vox // 2, axis=0).astype(vox.dtype)
    sk = sc.teasar_skeletonize(naive)
    assert betti(sk.nodes, sk.edges)[0] == 1


def test_teasar_downsample_and_edges_mutually_exclusive():
    coords, edges = downsample_graph(two_cubes(), 2)
    with pytest.raises(ValueError):
        sc.teasar_skeletonize(coords, downsample=2, edges=edges)


def test_teasar_edges_matches_default_on_single_component():
    # On a single line, feeding the lifted graph gives the same component count
    # as the default coordinate-driven path.
    vox = line(12)
    coords, edges = downsample_graph(vox, 2)
    sk_inj = sc.teasar_skeletonize(coords, edges=edges)
    sk_def = sc.teasar_skeletonize(coords)
    assert betti(sk_inj.nodes, sk_inj.edges)[0] == 1
    assert betti(sk_def.nodes, sk_def.edges)[0] == 1


# --- integration with centerline -------------------------------------------

def test_centerline_edges_injection_keeps_components():
    vox = two_lines(gap=3)
    coords, edges = downsample_graph(vox, 2)
    sk = sc.centerline(coords, edges=edges)
    assert betti(sk.nodes, sk.edges)[0] == 2
