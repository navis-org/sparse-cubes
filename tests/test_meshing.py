import os
import time

import numpy as np
import pytest

import sparsecubes as sc

DATA = os.path.join(os.path.dirname(__file__), "10075_coarse.npy")

# A few representative shapes reused across the smooth-path tests.
SINGLE = np.array([[1, 1, 1]], dtype=np.uint32)
LSHAPE = np.array([[1, 1, 1], [1, 2, 1], [2, 1, 1]], dtype=np.uint32)
CUBE = np.array(
    [
        [1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1],
        [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2],
    ],
    dtype=np.uint32,
)
# Three voxels stepping along a diagonal in the z=0 plane.
DIAG_STAIR = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]], dtype=np.uint32)


def all_axis_aligned(normals, tol=1e-6):
    """True if every face normal points along a single coordinate axis."""
    near_zero = np.abs(np.asarray(normals)) < tol
    return bool(np.all(near_zero.sum(axis=1) == 2))


def test_meshing():
    # A single voxel
    voxels = np.array([[1,1,1]])

    m = sc.mesh(voxels)

    assert m.vertices.shape[0] == 8
    assert m.faces.shape[0] == 6 * 2

    # 3 L-shaped voxels
    voxels = np.array([[1,1,1],
                       [1,2,1],
                       [2,1,1]])

    m = sc.mesh(voxels)

    assert m.vertices.shape[0] == 16
    assert m.faces.shape[0] == 14 * 2

    # A simple 2 x 2 x 2 cube
    voxels = np.array([[1,1,1],
                       [2,1,1],
                       [1,2,1],
                       [2,2,1],
                       [1,1,2],
                       [2,1,2],
                       [1,2,2],
                       [2,2,2],
                       ])

    m = sc.mesh(voxels)

    assert m.vertices.shape[0] == 26
    assert m.faces.shape[0] == 24 * 2


def test_default_is_smooth():
    # The default path is the smooth (SurfaceNets) one; surface_nets() is the
    # explicit spelling of the same thing.
    for voxels in (SINGLE, LSHAPE, CUBE):
        default = sc.mesh(voxels)
        smooth = sc.mesh(voxels, smooth=True)
        explicit = sc.surface_nets(voxels)

        assert np.array_equal(np.asarray(default.vertices), np.asarray(smooth.vertices))
        assert np.array_equal(default.faces, smooth.faces)
        assert np.array_equal(np.asarray(default.vertices), np.asarray(explicit.vertices))
        assert np.array_equal(default.faces, explicit.faces)


def test_blocky_path_is_unchanged():
    # smooth=False must stay the historical culled-cube-faces output: integer
    # vertices on cube corners and only axis-aligned face normals.
    for voxels in (SINGLE, LSHAPE, CUBE):
        blocky = sc.mesh(voxels, smooth=False)

        verts = np.asarray(blocky.vertices)
        assert verts.dtype == voxels.dtype            # integer-dtype optimisation kept
        assert np.array_equal(verts, verts.astype(np.int64))  # all on integer corners
        assert all_axis_aligned(blocky.face_normals)

    # culled_faces() is the explicit spelling of mesh(..., smooth=False).
    explicit = sc.culled_faces(CUBE)
    direct = sc.mesh(CUBE, smooth=False)
    assert np.array_equal(np.asarray(explicit.vertices), np.asarray(direct.vertices))
    assert np.array_equal(explicit.faces, direct.faces)


def test_deprecated_aliases():
    # dual_contour / marching_cubes still work but warn and forward to mesh,
    # with the old `interpolate` argument mapping onto `smooth`.
    with pytest.warns(DeprecationWarning):
        dc = sc.dual_contour(CUBE, interpolate=False)
    with pytest.warns(DeprecationWarning):
        mc = sc.marching_cubes(CUBE, interpolate=False)

    blocky = sc.mesh(CUBE, smooth=False)
    for old in (dc, mc):
        assert np.array_equal(np.asarray(old.vertices), np.asarray(blocky.vertices))
        assert np.array_equal(old.faces, blocky.faces)

    # Default of the deprecated alias maps to the smooth default.
    with pytest.warns(DeprecationWarning):
        default = sc.dual_contour(CUBE)
    assert np.array_equal(np.asarray(default.vertices), np.asarray(sc.mesh(CUBE).vertices))


def test_smooth_single_voxel():
    # An isolated voxel yields one vertex per surrounding dual cell (8) and one
    # quad per exposed face (6 * 2 tris). It should be a closed, consistent blob.
    m = sc.mesh(SINGLE, smooth=True)

    assert m.vertices.shape[0] == 8
    assert m.faces.shape[0] == 12
    assert np.asarray(m.vertices).dtype.kind == "f"  # verts are floats here
    assert m.is_winding_consistent
    assert m.is_watertight


def test_smooth_winding_consistent():
    for voxels in (SINGLE, LSHAPE, CUBE, DIAG_STAIR):
        m = sc.mesh(voxels, smooth=True)
        assert m.is_winding_consistent
        # No densification: at most one vertex per exposed face corner.
        blocky = sc.mesh(voxels, smooth=False)
        assert m.vertices.shape[0] <= blocky.vertices.shape[0] * 4


def test_smooth_diagonal_is_chamfered():
    # On a diagonal staircase the blocky path only ever emits axis-aligned
    # quads, while the smooth path tilts faces to chamfer the steps.
    blocky = sc.mesh(DIAG_STAIR, smooth=False)
    smooth = sc.mesh(DIAG_STAIR, smooth=True)

    assert all_axis_aligned(blocky.face_normals)
    assert not all_axis_aligned(smooth.face_normals)


def test_spacing_applies_on_smooth_path():
    spacing = np.array([2.0, 3.0, 4.0])
    plain = sc.mesh(CUBE, smooth=True)
    scaled = sc.mesh(CUBE, smooth=True, spacing=spacing)

    assert np.allclose(np.asarray(scaled.vertices), np.asarray(plain.vertices) * spacing)


@pytest.mark.skipif(not os.path.exists(DATA), reason="test voxel cloud not available")
def test_smooth_no_perf_regression():
    # Guard against an accidental densification / per-voxel Python loop: the
    # smooth path should stay within a small constant factor of the blocky one
    # on a real (~40k voxel) cloud. Generous bound to avoid CI flakiness.
    voxels = np.load(DATA)

    def best(smooth):
        sc.mesh(voxels, smooth=smooth)  # warmup
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            sc.mesh(voxels, smooth=smooth)
            times.append(time.perf_counter() - t0)
        return min(times)

    blocky = best(False)
    smooth = best(True)

    m = sc.mesh(voxels, smooth=True)
    assert m.is_winding_consistent
    assert smooth < blocky * 5
