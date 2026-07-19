"""Tests for `sparsecubes.voxelize` (mesh -> sparse voxels)."""

import os
import warnings

import numpy as np
import pytest
import trimesh as tm

import sparsecubes as sc
from sparsecubes.core import INT_DTYPES
from _shapes import solid_cube, n_components
from _voxel_oracle import reference_bounds

DATA = os.path.join(os.path.dirname(__file__), "10075_coarse.npy")


def as_set(a):
    return set(map(tuple, np.asarray(a).tolist()))


def primitives():
    """Small meshes covering flat faces, curvature, tunnels and axis vertices."""
    rot = tm.transformations.rotation_matrix
    return {
        "icosphere": tm.creation.icosphere(subdivisions=2, radius=4.0),
        "aligned_box": tm.creation.box(extents=[5, 5, 5]),
        "rotated_box": tm.creation.box(extents=[7, 5, 3]).apply_transform(
            rot(0.6, [1, 2, 3])
        ),
        "cone": tm.creation.cone(radius=3, height=7, sections=16),
        "cylinder": tm.creation.cylinder(radius=2, height=5, sections=12),
        "capsule": tm.creation.capsule(height=6, radius=2.5, count=[12, 12]),
        "torus": tm.creation.torus(major_radius=5, minor_radius=1.6),
        "offset_sphere": tm.creation.icosphere(subdivisions=2, radius=3.0)
        .apply_translation([13.3, -7.7, 4.1]),
    }


PRIMITIVES = primitives()


# --- exact ground truth ----------------------------------------------------


def test_axis_aligned_box_is_exact():
    # Extents 4 placed on [0, 4]: with voxel i centred at i, the faces land at
    # +-0.5 of a cell, so there is no boundary ambiguity and the answer is
    # exactly the 5x5x5 block of voxels 0..4.
    m = tm.creation.box(extents=[4, 4, 4]).apply_translation([2, 2, 2])
    out = sc.voxelize(m, 1.0)
    expected = np.stack(
        np.meshgrid(*[np.arange(5)] * 3, indexing="ij"), -1
    ).reshape(-1, 3)
    assert as_set(out) == as_set(expected)


def test_axis_aligned_box_shell():
    m = tm.creation.box(extents=[4, 4, 4]).apply_translation([2, 2, 2])
    shell = sc.voxelize(m, 1.0, solid=False)
    # The 5x5x5 block minus its 3x3x3 interior.
    assert len(shell) == 125 - 27
    assert as_set(shell) <= as_set(sc.voxelize(m, 1.0))


@pytest.mark.parametrize("name", sorted(PRIMITIVES))
@pytest.mark.parametrize("spacing", [1.0, 0.7, 1.3])
def test_matches_brute_force_oracle(name, spacing):
    """Result must sit inside an exact bracket computed by independent algorithms.

    The two bounds differ only for cells the surface merely grazes; anything
    outside the bracket is a real error in either direction.
    """
    m = PRIMITIVES[name]
    got = as_set(sc.voxelize(m, spacing))
    lower, upper = reference_bounds(m, spacing)
    assert got <= upper, f"{len(got - upper)} voxel(s) the mesh does not reach"
    assert lower <= got, f"{len(lower - got)} voxel(s) missing"


def test_anisotropic_spacing_matches_oracle():
    m = PRIMITIVES["icosphere"]
    spacing = (1.4, 0.8, 1.1)
    got = as_set(sc.voxelize(m, spacing))
    lower, upper = reference_bounds(m, spacing)
    assert lower <= got <= upper


# --- the degenerate cases that make this hard ------------------------------


def test_vertex_on_column_axis_does_not_lose_the_column():
    """A vertex exactly on a column centre must yield one crossing, not two.

    An icosphere has vertices at (0, 0, +-r) and a cone has its apex there. If the
    triangle fan around such a vertex claims the column twice, the column's
    crossings pair up wrongly and the whole interior column drops out.
    """
    for m in (
        tm.creation.icosphere(subdivisions=2, radius=4.0),
        tm.creation.cone(radius=3, height=7, sections=16),
    ):
        out = sc.voxelize(m, 0.7)
        axis = out[(out[:, 0] == 0) & (out[:, 1] == 0)]
        z = np.sort(axis[:, 2])
        assert len(z) > 0, "the mesh's own axis column came out empty"
        # Must be one contiguous run - a gap means a spurious crossing pair.
        assert np.array_equal(z, np.arange(z[0], z[0] + len(z)))


def test_face_diagonal_through_column_centres():
    """Column centres landing on a face's split diagonal must not punch holes.

    A box face is two triangles meeting along a diagonal that passes exactly
    through column centres. If neither triangle claims those columns, the parity
    flips and the diagonal is missing from the solid.
    """
    m = tm.creation.box(extents=[5, 5, 5])
    out = sc.voxelize(m, 1.3)
    expected = np.stack(
        np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij"), -1
    ).reshape(-1, 3)
    assert as_set(out) == as_set(expected)


def test_no_spurious_watertight_warning():
    """Watertight meshes must pair up every column without complaint."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name, m in PRIMITIVES.items():
            assert m.is_watertight, name
            sc.voxelize(m, 0.9)


def test_enclosed_cavity_stays_empty():
    """Even-odd parity must leave a fully enclosed void unfilled."""
    outer = tm.creation.icosphere(subdivisions=4, radius=12.0)
    inner = tm.creation.icosphere(subdivisions=4, radius=6.0)
    inner.invert()
    shell = tm.util.concatenate([outer, inner])
    out = sc.voxelize(shell, 1.0)
    r = np.linalg.norm(out.astype(float), axis=1)
    assert (r < 5).sum() == 0, "the enclosed cavity was filled in"
    assert (r > 13).sum() == 0
    assert ((r > 7) & (r < 11)).sum() > 100, "the shell itself is missing"


@pytest.mark.parametrize("name", ["icosphere", "aligned_box", "cone"])
def test_unmerged_triangle_soup(name):
    """Duplicated vertices must give the same answer as a shared-vertex mesh.

    STL and many exported formats carry no vertex sharing at all, so the
    exact-tie handling has to key off coordinate *values*, not vertex indices.
    """
    m = PRIMITIVES[name]
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
    soup_v = V[F].reshape(-1, 3)
    soup_f = np.arange(len(soup_v)).reshape(-1, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert as_set(sc.voxelize((soup_v, soup_f), 0.9)) == as_set(sc.voxelize(m, 0.9))


def test_degenerate_faces_are_ignored():
    """Zero-area triangles have no plane and must simply drop out."""
    m = PRIMITIVES["icosphere"]
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
    padded = np.vstack([F, [[0, 0, 0], [1, 1, 2], [3, 4, 4]]])
    assert as_set(sc.voxelize((V, padded), 0.9)) == as_set(sc.voxelize(m, 0.9))


def test_winding_is_irrelevant():
    """Parity fill must not care about face orientation."""
    m = tm.creation.icosphere(subdivisions=2, radius=4.0)
    flipped = m.copy()
    flipped.faces = np.asarray(flipped.faces)[:, ::-1]  # reverse every winding
    assert as_set(sc.voxelize(m, 0.8)) == as_set(sc.voxelize(flipped, 0.8))


# --- conventions and plumbing ----------------------------------------------


def test_spacing_scales_the_result():
    """Halving the spacing must not move the object, only resolve it finer."""
    m = tm.creation.icosphere(subdivisions=2, radius=4.0)
    coarse = sc.voxelize(m, 1.0).astype(float)
    fine = sc.voxelize(m, 0.5).astype(float) * 0.5
    assert np.allclose(coarse.min(0), fine.min(0), atol=1.0)
    assert np.allclose(coarse.max(0), fine.max(0), atol=1.0)


def test_translation_is_equivariant():
    """Translating by a whole number of voxels must shift the indices, nothing else."""
    m = tm.creation.icosphere(subdivisions=2, radius=4.0)
    base = sc.voxelize(m, 0.5)
    shifted = sc.voxelize(m.copy().apply_translation([3.0, -2.0, 5.0]), 0.5)
    assert as_set(shifted) == as_set(base + np.array([6, -4, 10]))


def test_negative_coordinates_round_trip():
    m = tm.creation.icosphere(subdivisions=2, radius=4.0)
    out = sc.voxelize(m, 1.0)
    assert (out < 0).any(), "expected negative indices for an origin-centred mesh"
    assert out.dtype in INT_DTYPES


def test_output_is_sorted_and_unique():
    out = sc.voxelize(PRIMITIVES["torus"], 0.9)
    keys = (out[:, 0].astype(np.int64) << 42) + (out[:, 1] << 21) + out[:, 2]
    assert np.all(np.diff(keys) > 0)


def test_accepts_vertices_faces_tuple():
    m = PRIMITIVES["icosphere"]
    a = sc.voxelize(m, 1.0)
    b = sc.voxelize((np.asarray(m.vertices), np.asarray(m.faces)), 1.0)
    assert as_set(a) == as_set(b)


def test_empty_mesh():
    out = sc.voxelize((np.zeros((0, 3)), np.zeros((0, 3), dtype=int)), 1.0)
    assert out.shape == (0, 3)
    assert out.dtype in INT_DTYPES


@pytest.mark.parametrize("spacing", [0, -1.0, np.inf, (1.0, 2.0)])
def test_bad_spacing_raises(spacing):
    with pytest.raises(ValueError):
        sc.voxelize(PRIMITIVES["icosphere"], spacing)


def test_bad_mesh_raises():
    with pytest.raises(TypeError):
        sc.voxelize("not a mesh", 1.0)


def test_extent_guard():
    """Too fine a spacing must raise rather than silently overflow pack()."""
    with pytest.raises(ValueError, match="exceeding pack"):
        sc.voxelize(PRIMITIVES["icosphere"], 1e-6)


def test_non_watertight_warns():
    m = tm.creation.icosphere(subdivisions=3, radius=6.0)
    m.update_faces(m.face_normals[:, 2] < 0.85)  # cut the top cap off
    assert not m.is_watertight
    with pytest.warns(UserWarning, match="not watertight"):
        sc.voxelize(m, 1.0)
    with warnings.catch_warnings():  # surface-only never needs pairing
        warnings.simplefilter("error")
        sc.voxelize(m, 1.0, solid=False)


# --- topology and interop --------------------------------------------------


def test_topology_of_primitives():
    ball = sc.voxelize(tm.creation.icosphere(subdivisions=3, radius=8.0), 1.0)
    assert n_components(ball) == 1
    shell = sc.voxelize(tm.creation.icosphere(subdivisions=3, radius=8.0), 1.0, solid=False)
    assert n_components(shell) == 1
    assert len(shell) < len(ball)


def test_round_trip_through_mesh():
    """voxelize(mesh(voxels)) must recover the original block."""
    cube = solid_cube(6)
    m = sc.mesh(cube, smooth=False)
    back = sc.voxelize(m, 1.0)
    assert as_set(cube) <= as_set(back)


def test_feeds_downstream_functions():
    """Output must be directly usable by the rest of the library."""
    m = tm.creation.capsule(height=20, radius=3, count=[16, 16])
    vox = sc.voxelize(m, 1.0)
    assert vox.dtype in INT_DTYPES
    thinned = sc.thin(vox)
    assert len(thinned) < len(vox)
    assert as_set(thinned) <= as_set(vox)


# --- scale -----------------------------------------------------------------


@pytest.mark.skipif(not os.path.exists(DATA), reason="test data not available")
def test_no_densification_on_a_real_mesh():
    """Peak memory must track the output, not the bounding-box volume.

    Run at a spacing fine enough that a dense occupancy grid would dwarf the
    sparse result, so an accidental densification cannot hide inside the noise.
    """
    import tracemalloc

    vox = np.load(DATA)
    m = sc.mesh(vox, smooth=False)
    spacing = 0.4
    dense_bytes = np.prod((vox.max(0) - vox.min(0) + 1).astype(float)) / spacing**3

    tracemalloc.start()
    out = sc.voxelize(m, spacing)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert len(out) > len(vox)
    assert peak < dense_bytes / 2, (
        f"peak {peak / 1e6:.0f} MB vs {dense_bytes / 1e6:.0f} MB for a dense "
        "bool grid - looks like the bounding box got materialized"
    )
