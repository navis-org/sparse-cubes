"""Tests for `sparsecubes.tube` - parametric tube coefficient extraction.

The oracles here are deliberately of three different kinds, because the pipeline
has three independent places to be wrong:

* **analytic** - `lobed_cylinder` has radius ``R + A cos(k theta)`` by construction,
  so ``a0`` and ``m_k`` are known in closed form. This pins the Fourier stage.
* **brute force** - `_tube_oracle.brute_force_radii` walks each ray in fine fixed
  steps against a Python ``set``, sharing no arithmetic with the DDA. This pins
  the raycaster.
* **invariants** - magnitudes must not move when the object is rotated about its
  own axis, and the two membership backends must agree bit for bit. These pin the
  claims the representation actually rests on.
"""

import os
import tracemalloc

import numpy as np
import pytest

import sparsecubes as sc
import sparsecubes.tube as T
from sparsecubes.skeleton import Skeleton
from sparsecubes.teasar import teasar_skeletonize

from _shapes import (
    PAD,
    annulus,
    axial_skeleton,
    elliptic_cylinder,
    line,
    disc_cylinder,
    lobed_cylinder,
    self_touch_hairpin,
    solid_cylinder,
    y_branch,
)
from _tube_oracle import brute_force_radii, ring_directions

# The continuum Fourier expansion of the polar-form ellipse r(theta) =
# a*b/sqrt((b cos)^2 + (a sin)^2) at a=10, b=5. Computed once at 4096 samples;
# recomputed by `test_ellipse_matches_continuum` so the constants cannot rot.
ELLIPSE_A, ELLIPSE_B = 10, 5


def _ellipse_reference(a, b, n=4096):
    """Continuum ``(a0, m_1..m_6)`` of an ellipse's polar radius function."""
    th = 2.0 * np.pi * np.arange(n) / n
    r = a * b / np.sqrt((b * np.cos(th)) ** 2 + (a * np.sin(th)) ** 2)
    c = np.fft.rfft(r) / n
    return c[0].real, 2.0 * np.abs(c[1:7])


# The backend-parity fixtures, built once. Inlining the dict in each test rebuilt
# all four shapes on every parametrised run to use one of them.
_PARITY = {
    "y_branch": y_branch(9),
    "hairpin": self_touch_hairpin(),
    "annulus": annulus(9, 5),
    "lobed": lobed_cylinder(radius=6, n_lobes=3, amp=1.0, length=20),
}


def _mid(arr):
    """Drop the end caps, where a cross-section is not a cross-section."""
    return arr[5:-5]


# ---------------------------------------------------------------------------
# the Fourier stage, against an analytic cross-section
# ---------------------------------------------------------------------------


def test_circular_cross_section_has_no_odd_harmonics():
    """A plain disc: all the energy in ``a0``, and none of it anywhere else.

    ``m_4`` is not asserted to zero. A disc rasterized on a cubic lattice has
    exactly four-fold symmetry, so the quantization error lands in ``k = 4, 8,
    ...`` - the direction-dependent noise floor the extraction cannot do better
    than. Asserting it away would be asserting a falsehood.
    """
    vox = disc_cylinder(8, 40)
    p = T.tube_profile(vox, axial_skeleton(vox), K=6, n_theta=128, max_radius=60)

    assert np.abs(_mid(p.a0) - 8.0).max() < 0.35
    mag = _mid(p.mag)
    assert mag[:, [0, 1, 2, 4]].max() < 0.05  # k = 1, 2, 3, 5
    assert mag[:, 3].max() < 0.30  # k = 4: the lattice's own signature
    assert _mid(p.residual).max() < 0.6


@pytest.mark.parametrize("n_lobes,amp", [(2, 1.0), (3, 1.5), (4, 2.0), (5, 1.0), (6, 1.2)])
def test_lobe_lands_in_its_own_harmonic(n_lobes, amp):
    """``r = 8 + amp*cos(k theta)`` must come back as ``a0 = 8``, ``m_k = amp``."""
    vox = lobed_cylinder(radius=8, n_lobes=n_lobes, amp=amp, length=40)
    p = T.tube_profile(vox, axial_skeleton(vox), K=8, n_theta=128, max_radius=60)

    mag = _mid(p.mag).mean(axis=0)
    assert np.abs(_mid(p.a0).mean() - 8.0) < 0.15
    assert int(np.argmax(mag)) + 1 == n_lobes
    assert abs(mag[n_lobes - 1] - amp) < 0.2
    assert np.delete(mag, n_lobes - 1).max() < 0.25


def test_ellipse_matches_continuum():
    """Ellipticity is a ``k=2`` phenomenon, to within the rasterization error."""
    a0_ref, mag_ref = _ellipse_reference(ELLIPSE_A, ELLIPSE_B)
    vox = elliptic_cylinder(ELLIPSE_A, ELLIPSE_B, 40)
    p = T.tube_profile(vox, axial_skeleton(vox), K=6, n_theta=128, max_radius=60)

    mag = _mid(p.mag).mean(axis=0)
    assert abs(_mid(p.a0).mean() - a0_ref) < 0.2
    assert abs(mag[1] - mag_ref[1]) < 0.35  # k=2 carries the shape
    assert abs(mag[3] - mag_ref[3]) < 0.2  # k=4 the next non-zero term
    # An ellipse has two-fold symmetry, so every odd harmonic vanishes exactly.
    assert mag[[0, 2, 4]].max() < 1e-6


def test_truncation_residual_is_the_real_error():
    """`residual` must equal the RMS of what truncating at ``K`` actually threw away.

    Checked by fitting the same cross-section at a much higher ``K`` and measuring
    the discarded tail directly, so this tests the Parseval shortcut in `_fourier`
    rather than restating it.
    """
    vox = lobed_cylinder(radius=8, n_lobes=3, amp=1.5, length=40)
    skel = axial_skeleton(vox)
    lo = T.tube_profile(vox, skel, K=2, n_theta=128, max_radius=60)
    hi = T.tube_profile(vox, skel, K=40, n_theta=128, max_radius=60)

    # Everything `lo` dropped is still present in `hi`; weights are 2 off DC.
    dropped = np.sqrt((hi.mag[:, 2:].astype(float) ** 2).sum(axis=1) / 2.0)
    assert np.allclose(lo.residual, np.hypot(dropped, hi.residual), atol=1e-4)
    # Truncating harder can only cost more.
    assert (lo.residual >= hi.residual - 1e-6).all()


# ---------------------------------------------------------------------------
# the raycaster, against a brute-force walk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vox",
    [
        disc_cylinder(5, 12),
        lobed_cylinder(radius=5, n_lobes=3, amp=1.5, length=12),
        elliptic_cylinder(7, 4, 12),
    ],
    ids=["disc", "lobed", "ellipse"],
)
def test_raycast_matches_brute_force(vox):
    """The DDA's sub-voxel exits must match a fine fixed-step walk.

    The oracle overshoots the true face crossing by up to one sampling step, and
    only ever in one direction - so the comparison is one-sided.
    """
    step, n_theta = 0.02, 32
    skel = axial_skeleton(vox)
    p = T.tube_profile(vox, skel, K=8, n_theta=n_theta, max_radius=40)
    u, v, _ = p.frame_vectors()

    for i in range(2, len(skel.nodes) - 2):
        dirs = ring_directions(u[i], v[i], n_theta, np.ones(3))
        origins = np.repeat(skel.nodes[i][None, :], n_theta, axis=0)
        ref = brute_force_radii(vox, origins, dirs, max_dist=40, step=step)
        # Refit the oracle's samples the same way and compare the coefficients.
        a0, mag, _, _ = T._fourier(ref[None, :], p.K)
        assert a0[0] - p.a0[i] > -step  # the oracle never undershoots
        assert a0[0] - p.a0[i] < step * 2
        assert np.abs(mag[0] - p.mag[i]).max() < 4 * step


def test_exit_distance_is_sub_voxel():
    """A radius-`r` disc must read out with sub-voxel precision, not as an integer.

    A voxel-counting raycaster would land on halves; the exact face crossing does
    not, and that is the whole point of using Amanatides-Woo rather than sampling.
    """
    vox = disc_cylinder(6, 12)
    # 4 rays, exactly along +-x/+-y: the disc's extreme voxel is at 6, whose outer
    # face is at 6.5, so this is analytically exact.
    p = T.tube_profile(vox, axial_skeleton(vox), K=1, n_theta=4, max_radius=40)
    assert np.allclose(p.a0[2:-2], 6.5, atol=1e-9)
    assert np.abs(p.mag[2:-2]).max() < 1e-9


def test_rays_that_never_exit_are_flagged_not_hung():
    """A ray fired down an infinite corridor must terminate at `max_radius`."""
    # A slab: rays in the plane of the slab never leave it within the cap.
    xs, ys = np.meshgrid(np.arange(60), np.arange(60))
    slab = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size, int)], axis=1)
    slab = np.vstack([slab, slab + (0, 0, 1)]).astype(np.int64)
    nodes = np.array([[30.0, 30.0, 0.0], [30.0, 30.0, 1.0]])
    skel = Skeleton(nodes, np.array([[0, 1]]), None, None)

    p = T.tube_profile(slab, skel, K=2, n_theta=16, max_radius=5.0)
    assert p.flag("ray_escaped").any()
    assert p.a0.max() <= 5.0 + 1e-9


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def test_quaternion_round_trips_through_every_shepperd_branch():
    rng = np.random.default_rng(0)
    t = T._normalize(rng.normal(size=(20000, 3)))
    u, v = T._seed_frame(t)

    xx, yy, zz = u[:, 0], v[:, 1], t[:, 2]
    case = np.where(xx + yy + zz > 0, 0, 1 + np.argmax(np.stack([xx, yy, zz], 1), 1))
    assert (np.bincount(case, minlength=4) > 0).all(), "a branch went unexercised"

    q = T._frame_to_quat(u, v, t)
    u2, v2, t2 = T._quat_to_frame(q)
    assert np.allclose(u, u2, atol=1e-12)
    assert np.allclose(v, v2, atol=1e-12)
    assert np.allclose(t, t2, atol=1e-12)
    assert (q[:, 3] >= 0).all(), "not canonicalised; delta coding would alternate"


@pytest.mark.parametrize("shape", ["y_branch", "hairpin", "cylinder"])
def test_frames_are_orthonormal(shape):
    vox = {"y_branch": y_branch(9), "hairpin": self_touch_hairpin(),
           "cylinder": solid_cylinder(6, 30)}[shape]
    p = T.tube_profile(vox, teasar_skeletonize(vox), K=4, n_theta=32)
    u, v, t = p.frame_vectors()

    for w in (u, v, t):
        assert np.abs(np.linalg.norm(w, axis=1) - 1.0).max() < 1e-6
    assert np.abs(np.einsum("ij,ij->i", u, v)).max() < 1e-6
    assert np.abs(np.einsum("ij,ij->i", u, t)).max() < 1e-6
    assert np.abs(np.einsum("ij,ij->i", v, t)).max() < 1e-6
    # Right-handed: u x v == t, not -t.
    assert np.abs(np.cross(u, v) - t).max() < 1e-6


def test_rmf_does_not_twist_along_a_straight_branch():
    """On a straight run the rotation-minimizing frame must not rotate at all.

    A Frenet frame is undefined here (zero curvature) and in practice spins or
    flips; that difference is the whole reason `_frames` uses double reflection.
    """
    vox = solid_cylinder(6, 40)
    p = T.tube_profile(vox, axial_skeleton(vox, centre=(6, 6)), K=2, n_theta=32)
    u, _, _ = p.frame_vectors()
    assert np.abs(u - u[0]).max() < 1e-9


def test_tangent_follows_the_branch():
    vox = solid_cylinder(6, 40)
    p = T.tube_profile(vox, axial_skeleton(vox, centre=(6, 6)), K=2, n_theta=32)
    _, _, t = p.frame_vectors()
    assert np.abs(np.abs(t[:, 2]) - 1.0).max() < 1e-9


# ---------------------------------------------------------------------------
# the frame-independence claim the representation rests on
# ---------------------------------------------------------------------------


def test_magnitudes_are_frame_independent_but_phases_are_not():
    """Rotating the object about its own axis must not move ``m_k``.

    This is the property that makes truncating on ``m_k`` legitimate and truncating
    on ``a_k``/``b_k`` meaningless. The residual spread is rasterization - a rotated
    ellipse lands on different voxels - not frame dependence.
    """
    rots = [0.0, 0.4, 0.8, 1.2]
    mags, phases = [], []
    for rot in rots:
        vox = elliptic_cylinder(ELLIPSE_A, ELLIPSE_B, 40, rot=rot)
        p = T.tube_profile(vox, axial_skeleton(vox), K=4, n_theta=128, max_radius=60)
        mags.append(_mid(p.mag).mean(axis=0))
        phases.append(_mid(p.phase).mean(axis=0))
    mags, phases = np.array(mags), np.array(phases)

    m2 = mags[:, 1]
    assert (m2.max() - m2.min()) / m2.mean() < 0.25, "m_2 moved with the frame"
    assert m2.min() > 2.0, "the k=2 signal collapsed at some rotation"

    # ...while phi_2 advances by 2 * rot, the k=2 harmonic's own rate.
    d_phase = np.unwrap(phases[:, 1] - phases[0, 1])
    assert np.abs(d_phase - 2.0 * np.array(rots)).max() < 0.35


# ---------------------------------------------------------------------------
# anisotropy
# ---------------------------------------------------------------------------


def test_spacing_scales_radii_into_physical_units():
    vox = disc_cylinder(8, 30)
    base = T.tube_profile(vox, axial_skeleton(vox), K=4, n_theta=64, max_radius=1e4)
    scaled = T.tube_profile(
        vox, axial_skeleton(vox, spacing=(4, 4, 4)), K=4, n_theta=64, max_radius=1e4
    )
    assert np.allclose(scaled.a0, 4.0 * base.a0, rtol=1e-6)
    assert np.allclose(scaled.mag, 4.0 * base.mag, rtol=1e-6, atol=1e-6)
    assert np.allclose(scaled.residual, 4.0 * base.residual, rtol=1e-5, atol=1e-5)


def test_anisotropy_only_touches_axes_the_cross_section_spans():
    """At 4x4x40, a z-running neurite's cross-section never sees the 40.

    The converse is the artifact worth remembering, and the sharpest statement of
    it: the *same* voxel cylinder read with its axis along z is perfectly circular
    (``m_2 = 0``), and read with its axis in-plane is wildly elliptical
    (``m_2 = 68``), purely because z is then a radial direction. Identical geometry,
    completely different apparent harmonic content - so the noise floor on ``m_k``
    is a function of the neurite's direction, not a constant.
    """
    vox = disc_cylinder(8, 30)
    along_z = T.tube_profile(
        vox, axial_skeleton(vox, spacing=(4, 4, 40)), K=4, n_theta=64, max_radius=1e4
    )
    isotropic = T.tube_profile(
        vox, axial_skeleton(vox, spacing=(4, 4, 4)), K=4, n_theta=64, max_radius=1e4
    )
    assert np.allclose(along_z.a0, isotropic.a0, rtol=1e-9)
    assert along_z.mag[:, 1].max() < 1e-4, "a z-running tube must read as circular"

    # Same object, rotated so its axis runs along +x: now z is a radial direction.
    swapped = vox[:, [2, 1, 0]].copy()
    skel = axial_skeleton(vox, spacing=(4, 4, 40))
    nodes = skel.nodes[:, [2, 1, 0]].copy()
    across = T.tube_profile(
        swapped, Skeleton(nodes, skel.edges, None, np.array([4.0, 4.0, 40.0])),
        K=4, n_theta=64, max_radius=1e4,
    )
    mid = slice(5, -5)
    assert across.mag[mid, 1].mean() > 50.0, "the 10:1 anisotropy vanished"

    # Both readings match the continuum polar-radius expansion of the ellipse the
    # anisotropy makes of the disc, which is what says the scaling is exact rather
    # than merely large: semi-axes 4*a0_iso and 40*a0_iso.
    half = float(isotropic.a0[mid].mean()) / 4.0
    ref_a0, ref_mag = _ellipse_reference(40.0 * half, 4.0 * half)
    assert abs(across.a0[mid].mean() - ref_a0) < 2.0
    assert abs(across.mag[mid, 1].mean() - ref_mag[1]) < 2.0


# ---------------------------------------------------------------------------
# backend parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", list(_PARITY))
def test_backends_agree(shape, monkeypatch):
    """Every raycasting backend must produce identical output.

    Not `allclose` - bit-identical. `"exits"` walks the DDA in Rust, `"sparse"`
    and `"keys"` walk it in numpy over two different membership probes, so any
    discrepancy is a bug in one of the walks, never a tolerance.
    """
    vox = _PARITY[shape]
    skel = teasar_skeletonize(vox)

    fast = T.tube_profile(vox, skel, K=4, n_theta=32, diagnostics=True)
    monkeypatch.setattr(T, "_dijkstra3d_sparse", None)
    slow = T.tube_profile(vox, skel, K=4, n_theta=32, diagnostics=True)

    fields = ("nodes", "edges", "a0", "mag", "phase", "frame", "branch",
              "flags", "residual", "non_star")
    for field in fields:
        np.testing.assert_array_equal(
            getattr(fast, field), getattr(slow, field), err_msg=f"{field} differs"
        )


@pytest.mark.parametrize("shape", list(_PARITY))
@pytest.mark.parametrize("diagnostics", [False, True])
def test_ray_exits_backend_matches_the_numpy_walk(shape, diagnostics):
    """The Rust DDA against the numpy one, on the shapes built to break them.

    `annulus` and `hairpin` are the re-entry cases, so this also pins the second
    `ray_exits` pass - the one that has to reproduce `star_window`'s own deadline
    rather than sharing the escape cap.
    """
    vox = _PARITY[shape]
    skel = teasar_skeletonize(vox)

    # Assert the routing first. A parity test cannot see a backend that silently
    # falls through to the other one - it just compares an implementation to
    # itself and passes.
    assert T._caster(vox, "exits").name == "exits"
    assert T._caster(vox, "sparse").name == "sparse"

    kw = dict(K=4, n_theta=32, diagnostics=diagnostics)
    rust = T.tube_profile(vox, skel, backend="exits", **kw)
    numpy_ = T.tube_profile(vox, skel, backend="sparse", **kw)
    for field in ("a0", "mag", "phase", "flags", "residual", "non_star"):
        np.testing.assert_array_equal(
            getattr(rust, field), getattr(numpy_, field), err_msg=f"{field} differs"
        )


@pytest.mark.parametrize("star_window", [0.1, 1.0, 4.0])
def test_ray_exits_reproduces_the_star_window(star_window):
    """The re-entry deadline is the one place the two casters could silently drift.

    `ray_exits` takes its cap up front, but the deadline is
    ``(1 + star_window) * t_exit`` and so is only known after the first pass.
    Sweeping the window is what proves the second call actually uses it.
    """
    vox = annulus(9, 5)
    skel = teasar_skeletonize(vox)
    kw = dict(K=4, n_theta=64, diagnostics=True, star_window=star_window)
    rust = T.tube_profile(vox, skel, backend="exits", **kw)
    numpy_ = T.tube_profile(vox, skel, backend="sparse", **kw)
    np.testing.assert_array_equal(rust.non_star, numpy_.non_star)
    assert rust.non_star.max() > 0  # the fixture has to actually exercise it


def test_explicit_backend_selection():
    vox = disc_cylinder(5, 12)
    skel = axial_skeleton(vox)
    a = T.tube_profile(vox, skel, K=2, n_theta=32, backend="sparse")
    b = T.tube_profile(vox, skel, K=2, n_theta=32, backend="keys")
    np.testing.assert_array_equal(a.a0, b.a0)
    with pytest.raises(ValueError):
        T.tube_profile(vox, skel, backend="nope")

    # Every name must select the backend it names, not merely produce an answer.
    assert T._caster(vox, "keys").name == "keys"
    assert T._caster(vox, "sparse").name == "sparse"
    assert T._caster(vox, None).name == "exits"


def test_backends_without_the_library(monkeypatch):
    """With the library nulled out - how `test_backends_agree` runs its oracle."""
    monkeypatch.setattr(T, "_dijkstra3d_sparse", None)
    with pytest.raises(RuntimeError, match="dijkstra3d_sparse"):
        T._caster(np.zeros((1, 3), np.int64), "exits")

    vox = disc_cylinder(5, 12)
    assert T._caster(vox, None).name == "keys"


@pytest.mark.parametrize("backend", ["exits", "sparse", "keys"])
def test_duplicate_voxels_are_tolerated(backend):
    """`Graph` rejects duplicates, so the fallback dedup has to actually fire.

    The dedup is no longer run up front - it cost 4.3 s on a 5.5 M-voxel arbor to
    discover there was nothing to remove - so this is the only thing standing
    between a duplicated input and a `ValueError` out of the index build.
    """
    vox = lobed_cylinder(radius=6, n_lobes=3, amp=1.0, length=20)
    skel = axial_skeleton(vox)
    dup = np.concatenate([vox, vox[::3]])  # a third of them, twice

    clean = T.tube_profile(vox, skel, K=4, n_theta=32, backend=backend)
    doubled = T.tube_profile(dup, skel, K=4, n_theta=32, backend=backend)
    np.testing.assert_array_equal(clean.a0, doubled.a0)
    np.testing.assert_array_equal(clean.mag, doubled.mag)


def test_seed_outside_nodes_are_skipped_not_measured():
    """A node outside the object must read as zero, not as half a voxel."""
    vox = disc_cylinder(5, 12)
    skel = axial_skeleton(vox)
    nodes = np.asarray(skel.nodes, dtype=float).copy()
    nodes[3] = (PAD + 60, PAD + 60, nodes[3, 2])  # nowhere near the object
    moved = Skeleton(nodes, np.asarray(skel.edges), None, skel.spacing)

    p = T.tube_profile(vox, moved, K=4, n_theta=32)
    assert p.flag("seed_outside")[3] and not p.flag("seed_outside")[4]
    assert p.a0[3] == 0.0 and p.mag[3].max() == 0.0 and p.residual[3] == 0.0
    assert not p.flag("ray_escaped")[3]  # no rays were cast, so none escaped
    assert p.a0[4] > 4.0  # its neighbours are untouched


# ---------------------------------------------------------------------------
# flags
# ---------------------------------------------------------------------------


def test_junction_flag_covers_the_bifurcation_and_its_neighbours():
    vox = y_branch(9)
    skel = teasar_skeletonize(vox)
    p = T.tube_profile(vox, skel, K=2, n_theta=32)

    deg = skel.node_degrees()
    junction = deg >= 3
    assert junction.any(), "fixture no longer has a bifurcation"
    assert p.flag("junction")[junction].all()
    # ...and their direct neighbours, but nothing further out.
    expected = junction.copy()
    for a, b in skel.edges:
        if junction[a]:
            expected[b] = True
        if junction[b]:
            expected[a] = True
    np.testing.assert_array_equal(p.flag("junction"), expected)


def _parallel_tubes(radius=5, gap=3, length=30):
    """``(one tube, both tubes, skeleton down the first)``.

    Every ray from the first crosses the gap into the second, so this is the
    star-shapedness failure in its purest form. `gap` is a parameter because two
    callers assert on where the neighbour sits relative to the first exit.
    """
    a = disc_cylinder(radius, length)
    b = a.copy()
    b[:, 0] += 2 * radius + gap
    return a, np.vstack([a, b]), axial_skeleton(a)


def _lobed_profile():
    """The shared ``r = 8 + 1.5 cos(3 theta)`` fixture for the LOD/normal tests.

    Rebuilt per call rather than cached: `TubeProfile` is mutable, and these tests
    are cheap enough that sharing one across them would trade a real hazard for a
    fraction of a second.
    """
    vox = lobed_cylinder(radius=8, n_lobes=3, amp=1.5, length=30)
    return T.tube_profile(vox, axial_skeleton(vox), K=6, n_theta=64, max_radius=40)


def test_non_star_shaped_is_detected_only_with_diagnostics():
    """Two parallel tubes: every ray from one crosses the gap into the other.

    This is the star-shapedness failure the handoff calls the hard ceiling on the
    approach, in its purest form - and it costs nothing to spot, because the rays
    are already being walked.
    """
    a, both, skel = _parallel_tubes()

    on = T.tube_profile(both, skel, K=4, n_theta=64, diagnostics=True, max_radius=40)
    off = T.tube_profile(both, skel, K=4, n_theta=64, diagnostics=False, max_radius=40)

    assert on.flag("non_star").all()
    assert not off.flag("non_star").any()
    assert (on.non_star > 0).all() and (off.non_star == 0).all()
    # Diagnostics must not change the profile itself, only what is known about it.
    np.testing.assert_array_equal(on.a0, off.a0)
    np.testing.assert_array_equal(on.mag, off.mag)

    # A single convex tube in isolation has nothing to re-enter.
    alone = T.tube_profile(a, skel, K=4, n_theta=64, diagnostics=True, max_radius=40)
    assert not alone.flag("non_star")[3:-3].any()


def test_star_window_is_independent_of_the_escape_cap():
    """The star-shapedness search must not inherit `max_radius`.

    Widening `max_radius` lets a ray keep going until it reaches some *other* part
    of the object, which is a fact about how densely the arbor is packed, not about
    whether this cross-section is star-shaped. On the neuron fixture the two
    windows disagree by more than a factor of two, so conflating them would have
    reported the approach as far more broken than it is.
    """
    a, both, skel = _parallel_tubes()

    # Rays reach the neighbour at ~13 voxels; the local radius is ~5.
    tight = T.tube_profile(both, skel, K=2, n_theta=64, diagnostics=True,
                           max_radius=40, star_window=0.25)
    wide = T.tube_profile(both, skel, K=2, n_theta=64, diagnostics=True,
                          max_radius=40, star_window=4.0)
    assert not tight.flag("non_star").any(), "a convex tube read as non-star"
    assert wide.flag("non_star").all(), "the neighbour went unseen"

    # The escape cap must not move the answer at a fixed star_window...
    near = T.tube_profile(both, skel, K=2, n_theta=64, diagnostics=True,
                          max_radius=40, star_window=4.0)
    far = T.tube_profile(both, skel, K=2, n_theta=64, diagnostics=True,
                         max_radius=400, star_window=4.0)
    np.testing.assert_array_equal(near.non_star, far.non_star)
    # ...while still doing its own job of bounding a ray that never exits.
    assert far.a0.max() < 40.0


def test_non_star_is_a_fraction_not_just_a_bit():
    """`non_star` must report severity: which fraction of a node's rays re-entered.

    A single grazing ray out of 64 and half the ring folding back are very different
    situations, and a bare flag cannot tell them apart.
    """
    a, both, skel = _parallel_tubes()

    p = T.tube_profile(both, skel, K=2, n_theta=64, diagnostics=True,
                       max_radius=40, star_window=4.0)
    assert p.non_star.dtype == np.float32
    # Only the rays pointing at the neighbour re-enter - a minority of the ring.
    assert 0.0 < p.non_star.mean() < 0.5
    np.testing.assert_array_equal(p.flag("non_star"), p.non_star > 0)


def test_seed_outside_flag():
    """A skeleton node in empty space is reported, not silently profiled."""
    vox = disc_cylinder(4, 10)
    nodes = np.array([[PAD, PAD, 5.0], [PAD, PAD, 6.0], [PAD + 500, PAD, 5.0]])
    skel = Skeleton(nodes, np.array([[0, 1]]), None, None)
    p = T.tube_profile(vox, skel, K=2, n_theta=16, max_radius=20)
    np.testing.assert_array_equal(p.flag("seed_outside"), [False, False, True])


def test_branch_end_flag_and_branch_ids():
    vox = y_branch(9)
    skel = teasar_skeletonize(vox)
    p = T.tube_profile(vox, skel, K=2, n_theta=16)
    deg = skel.node_degrees()

    assert p.flag("branch_end")[deg == 1].all(), "every tip is a branch end"
    assert p.branch.max() + 1 == 3, "a Y has three branches"
    assert p.branch.min() == 0


def test_flag_accessor_rejects_unknown_names():
    vox = line(6, axis=2)
    skel = Skeleton(vox.astype(float), np.stack([np.arange(5), np.arange(1, 6)], 1),
                    None, None)
    p = T.tube_profile(vox, skel, K=1, n_theta=8)
    assert p.flag("junction").shape == (6,)
    with pytest.raises(ValueError):
        p.flag("no_such_flag")


# ---------------------------------------------------------------------------
# evaluation and LOD
# ---------------------------------------------------------------------------


def test_radius_at_matches_the_cartesian_form():
    vox = elliptic_cylinder(9, 5, 20)
    p = T.tube_profile(vox, axial_skeleton(vox), K=4, n_theta=64, max_radius=40)
    th = np.linspace(0, 2 * np.pi, 37)
    a, b = p.coefficients()
    ref = p.a0[:, None].astype(float) + sum(
        a[:, j : j + 1] * np.cos((j + 1) * th[None, :])
        + b[:, j : j + 1] * np.sin((j + 1) * th[None, :])
        for j in range(p.K)
    )
    assert np.allclose(p.radius_at(th), ref, atol=1e-5)
    assert np.allclose(p.radius_at(th, k=0), p.a0[:, None].astype(float))
    with pytest.raises(ValueError):
        p.radius_at(th, k=p.K + 1)


def test_evaluate_reconstructs_the_surface():
    """Ring points must sit on the object's surface, within the truncation budget."""
    p = _lobed_profile()

    pts = p.evaluate()[5:-5]
    assert pts.shape[1:] == (64, 3)
    # Compare each ring point against the truth at *its own* world angle. The
    # frame's theta=0 sits at an arbitrary rotation about the tangent, so
    # comparing index for index would be testing the seed, not the surface.
    offset = pts[:, :, :2] - np.array([PAD, PAD])
    radial = np.linalg.norm(offset, axis=2)
    world = np.arctan2(offset[:, :, 1], offset[:, :, 0])
    assert np.abs(radial - (8.0 + 1.5 * np.cos(3 * world))).mean() < 0.6
    # The rings must stay in their own z plane - the frame is perpendicular.
    assert np.abs(pts[:, :, 2] - p.vertices[5:-5, 2][:, None]).max() < 1e-9


def test_lod_axes_are_independent():
    """Both LOD knobs must work, and truncating harder must cost more, not less."""
    p = _lobed_profile()

    assert p.evaluate(n_theta=16).shape == (len(p.nodes), 16, 3)  # angular sampling
    sub = np.arange(0, len(p.nodes), 3)
    assert p.evaluate(nodes=sub).shape == (len(sub), 64, 3)  # axial
    assert p.evaluate(n_theta=8, k=1, nodes=sub).shape == (len(sub), 8, 3)  # both

    # Dropping a harmonic can only add error - in RMS, which is the norm the
    # harmonics are orthogonal in. Mean absolute error is not monotone and
    # asserting on it would be asserting something untrue.
    th = 2 * np.pi * np.arange(64) / 64
    full = p.radius_at(th)
    err = [
        float(np.sqrt(((p.radius_at(th, k=k) - full) ** 2).mean()))
        for k in range(p.K + 1)
    ]
    assert err[-1] == 0.0
    assert all(err[i] >= err[i + 1] - 1e-12 for i in range(len(err) - 1))
    # And the drop from k to k+1 is exactly that harmonic's own energy.
    dropped = float(np.sqrt((p.mag[:, 0].astype(float) ** 2).mean() / 2.0))
    assert abs(np.sqrt(err[0] ** 2 - err[1] ** 2) - dropped) < 1e-4


# ---------------------------------------------------------------------------
# normals
# ---------------------------------------------------------------------------


def _straight_profile(a0):
    """A tube along +z with an identity frame, built without going near a voxel.

    Lets the normal be checked against a closed form instead of against a
    quantized reading of one.
    """
    m = len(a0)
    z = np.arange(float(m))
    return T.TubeProfile(
        nodes=np.stack([np.zeros(m), np.zeros(m), z], 1).astype(np.float32),
        edges=np.stack([np.arange(m - 1), np.arange(1, m)], 1).astype(np.int32),
        a0=np.asarray(a0, np.float32),
        mag=np.zeros((m, 1), np.float32),
        phase=np.zeros((m, 1), np.float32),
        frame=np.tile(np.array([0, 0, 0, 1], np.float32), (m, 1)),  # u=x, v=y, t=z
        branch=np.zeros(m, np.int32),
        flags=np.zeros(m, np.uint8),
        residual=np.zeros(m, np.float32),
        n_theta=64,
    )


def test_normals_of_a_straight_circular_tube_are_radial():
    p = _straight_profile(np.full(20, 5.0))
    pts, n = p.evaluate(n_theta=8, return_normals=True)

    assert n.shape == pts.shape
    th = 2 * np.pi * np.arange(8) / 8
    want = np.stack([np.cos(th), np.sin(th), np.zeros(8)], 1)
    assert np.abs(n - want[None]).max() < 1e-12


def test_normals_tilt_with_an_axial_taper():
    """The point of differencing the surface rather than reusing the tangent.

    A cone's centreline is straight, so a normal built from the stored tangent
    would come out radial - and be wrong by the taper angle everywhere.
    """
    slope = 0.25
    p = _straight_profile(5.0 + slope * np.arange(20.0))
    _, n = p.evaluate(n_theta=8, return_normals=True)

    th = 2 * np.pi * np.arange(8) / 8
    want = np.stack([np.cos(th), np.sin(th), np.full(8, -slope)], 1)
    want /= np.linalg.norm(want, axis=-1, keepdims=True)
    # Exact at the branch ends too: a one-sided difference of a cone is a cone.
    assert np.abs(n - want[None]).max() < 1e-12


def test_normals_match_the_analytic_surface_normal():
    """Against the continuum shape, not against the code's own formula.

    For r(w) in the plane the outward normal is (r cos w + r' sin w,
    r sin w - r' cos w, 0) - derived from the extruded surface, sharing no
    arithmetic with `evaluate`.
    """
    p = _lobed_profile()
    pts, n = p.evaluate(return_normals=True)
    pts, n = pts[5:-5], n[5:-5]

    assert np.abs(np.linalg.norm(n, axis=-1) - 1.0).max() < 1e-12
    off = pts[:, :, :2] - np.array([PAD, PAD])
    assert ((n[:, :, :2] * off).sum(-1) > 0).all()  # outward, every one
    assert np.abs(n[:, :, 2]).max() < 1e-12  # extruded: no axial component

    w = np.arctan2(off[:, :, 1], off[:, :, 0])
    r, dr = 8.0 + 1.5 * np.cos(3 * w), -4.5 * np.sin(3 * w)
    want = np.stack([r * np.cos(w) + dr * np.sin(w), r * np.sin(w) - dr * np.cos(w),
                     np.zeros_like(w)], -1)
    want /= np.linalg.norm(want, axis=-1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((n * want).sum(-1), -1.0, 1.0)))
    assert ang.mean() < 4.0 and np.percentile(ang, 95) < 8.0


def test_normal_error_grows_with_k_even_where_the_radius_error_does_not():
    """`dr/dtheta` weights harmonic k by k, so K is bounded by the normal.

    On a quantized circle the m_k stop decaying around the staircase floor; the
    radius does not care, but every extra harmonic tilts the normal further.
    """
    vox = lobed_cylinder(radius=8, n_lobes=3, amp=0.0, length=30)  # a circle
    p = T.tube_profile(vox, axial_skeleton(vox), K=8, n_theta=64, max_radius=60)

    th = 2 * np.pi * np.arange(64) / 64
    tilt = []
    for k in range(p.K + 1):
        r, dr = p.radius_at(th, k=k, derivative=True)
        tilt.append(float(np.degrees(np.arctan2(np.abs(dr[5:-5]), r[5:-5])).mean()))
    assert tilt[4] < 4.0  # the shipped K is fine ...
    assert tilt[8] > 2 * tilt[4]  # ... and buying more harmonics makes it worse

    # Which is what `k_normal` exists to undo: a smoother normal on the same
    # surface. The points must not move when only the normal is retruncated.
    pts, n4 = p.evaluate(n_theta=32, k=8, return_normals=True)
    pts1, n1 = p.evaluate(n_theta=32, k=8, return_normals=True, k_normal=1)
    np.testing.assert_array_equal(pts, pts1)
    u, v, _ = p.frame_vectors()
    th32 = 2 * np.pi * np.arange(32) / 32
    e_r = np.cos(th32)[None, :, None] * u[:, None, :] + np.sin(th32)[None, :, None] * v[:, None, :]

    def spread(n):
        return np.degrees(np.arccos(np.clip(np.einsum("ijk,ijk->ij", n, e_r), -1, 1)))[5:-5].mean()

    assert spread(n1) < spread(n4)
    # k_normal=None must mean "the surface's own normal", i.e. no smoothing.
    np.testing.assert_array_equal(n4, p.evaluate(n_theta=32, k=8, return_normals=True,
                                                 k_normal=8)[1])


def test_normals_survive_degenerate_nodes():
    """Nothing here may produce a NaN; the radial direction is the fallback."""
    p = _straight_profile(np.array([5.0]))  # one node, no edges, no axial direction
    _, n = p.evaluate(n_theta=4, return_normals=True)
    th = 2 * np.pi * np.arange(4) / 4
    assert np.abs(n[0] - np.stack([np.cos(th), np.sin(th), np.zeros(4)], 1)).max() < 1e-12

    p = _straight_profile(np.array([0.0, 0.0, 0.0]))  # zero radius everywhere
    _, n = p.evaluate(n_theta=4, return_normals=True)
    assert np.isfinite(n).all()
    assert np.abs(np.linalg.norm(n, axis=-1) - 1.0).max() < 1e-12


def test_normals_respect_the_lod_knobs():
    p = _lobed_profile()

    sub = np.arange(0, len(p.nodes), 3)
    pts, n = p.evaluate(n_theta=8, k=2, nodes=sub, return_normals=True)
    assert pts.shape == n.shape == (len(sub), 8, 3)
    # A boolean selection has to mean the same thing as the integer one it maps to.
    mask = np.zeros(len(p.nodes), bool)
    mask[sub] = True
    np.testing.assert_array_equal(n, p.evaluate(n_theta=8, k=2, nodes=mask, return_normals=True)[1])
    # Positions must not change just because normals were asked for.
    np.testing.assert_array_equal(pts, p.evaluate(n_theta=8, k=2, nodes=sub))


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------


def test_gpu_buffer_layout():
    vox = y_branch(9)
    p = T.tube_profile(vox, teasar_skeletonize(vox, spacing=(2, 2, 3)), K=4, n_theta=32)
    buf, hdr = p.to_gpu_buffer()

    assert buf.dtype == np.float32 and buf.flags["C_CONTIGUOUS"]
    assert buf.shape == (len(p.nodes), 8 + 2 * p.K)
    assert buf.nbytes / len(p.nodes) == 64.0, "K=4 must be 16 floats a node"
    assert hdr == {"K": 4, "n_theta": 32, "n_nodes": len(p.nodes),
                   "stride_floats": 16, "spacing": (2.0, 2.0, 3.0),
                   "form": "cartesian"}

    np.testing.assert_allclose(buf[:, 0:3], p.vertices, atol=1e-4)
    np.testing.assert_array_equal(buf[:, 3:7], p.frame)
    np.testing.assert_array_equal(buf[:, 7], p.a0)
    # Cartesian by default - a shader evaluates a_k cos + b_k sin by angle
    # addition, and would otherwise need a cos/sin of phi_k per harmonic.
    a, b = p.coefficients()
    np.testing.assert_allclose(buf[:, 8 : 8 + p.K], a, atol=1e-6)
    np.testing.assert_allclose(buf[:, 8 + p.K :], b, atol=1e-6)


def test_gpu_buffer_polar_form():
    vox = y_branch(9)
    p = T.tube_profile(vox, teasar_skeletonize(vox), K=4, n_theta=32)
    buf, hdr = p.to_gpu_buffer(form="polar")

    assert hdr["form"] == "polar"
    np.testing.assert_array_equal(buf[:, 8 : 8 + p.K], p.mag)
    np.testing.assert_array_equal(buf[:, 8 + p.K :], p.phase)
    # Everything outside the coefficient block is the same either way.
    other, _ = p.to_gpu_buffer()
    np.testing.assert_array_equal(buf[:, :8], other[:, :8])

    with pytest.raises(ValueError):
        p.to_gpu_buffer(form="nope")


@pytest.mark.parametrize("form", ["cartesian", "polar"])
def test_gpu_buffer_evaluates_to_the_same_surface(form):
    """Both forms must reconstruct `radius_at` - they are one series, two spellings.

    Evaluated the way the shader would, straight out of the flat buffer, so this
    also pins the offsets the WGSL indexes with.
    """
    vox = lobed_cylinder(radius=8, n_lobes=3, amp=1.5, length=20)
    p = T.tube_profile(vox, axial_skeleton(vox), K=5, n_theta=64, max_radius=40)
    buf, hdr = p.to_gpu_buffer(form=form)
    flat = buf.ravel()  # what actually gets bound
    K, stride = hdr["K"], hdr["stride_floats"]

    th = np.linspace(0, 2 * np.pi, 23)
    for i in range(0, len(p.nodes), 3):
        base = i * stride
        r = np.full_like(th, flat[base + 7])
        for k in range(1, K + 1):
            first = flat[base + 7 + k]
            second = flat[base + 7 + K + k]
            if form == "cartesian":
                r += first * np.cos(k * th) + second * np.sin(k * th)
            else:
                r += first * np.cos(k * th - second)
        np.testing.assert_allclose(r, p.radius_at(th)[i], atol=1e-4)


@pytest.mark.parametrize("compress", [True, False])
def test_npz_round_trip(tmp_path, compress):
    vox = y_branch(11)
    p = T.tube_profile(vox, teasar_skeletonize(vox, spacing=(2, 2, 3)),
                       K=4, n_theta=32, diagnostics=True)
    path = tmp_path / "profile.npz"
    p.save_npz(path, compress=compress)
    back = T.TubeProfile.load_npz(path)

    for field in ("nodes", "edges", "mag", "frame", "branch", "flags", "residual",
                  "non_star"):
        np.testing.assert_array_equal(getattr(p, field), getattr(back, field))
    np.testing.assert_allclose(p.a0, back.a0, atol=1e-4)
    np.testing.assert_allclose(p.phase, back.phase, atol=1e-4)
    np.testing.assert_array_equal(p.spacing, back.spacing)
    assert back.n_theta == p.n_theta and back.K == p.K


def test_delta_coding_restarts_at_every_branch():
    """A delta must never span a junction - across one the frame is discontinuous."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 2)).astype(np.float32)
    runs = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0], dtype=bool)

    enc = T._delta_encode(x, runs)
    np.testing.assert_allclose(T._delta_decode(enc, runs), x, atol=1e-5)
    # The first row of each run is stored verbatim, so a run is decodable alone.
    np.testing.assert_allclose(enc[runs], x[runs], atol=1e-6)

    for one_d in (x[:, 0], np.arange(12, dtype=np.float32)):
        np.testing.assert_allclose(
            T._delta_decode(T._delta_encode(one_d, runs), runs), one_d, atol=1e-5
        )
    # A single run, and a run per row, are the two boundary cases.
    for edge in (np.array([True] + [False] * 11), np.ones(12, bool)):
        np.testing.assert_allclose(
            T._delta_decode(T._delta_encode(x, edge), edge), x, atol=1e-5
        )


def test_to_skeleton_and_swc():
    vox = y_branch(9)
    p = T.tube_profile(vox, teasar_skeletonize(vox, spacing=(2, 2, 2)), K=2, n_theta=16)
    skel = p.to_skeleton()
    assert isinstance(skel, Skeleton)
    np.testing.assert_allclose(skel.radii, p.a0, rtol=1e-6)

    swc = p.to_swc()
    assert swc.shape == (len(p.nodes), 7)
    np.testing.assert_allclose(swc[:, 5], p.a0, rtol=1e-5)
    assert (swc[:, 6] == -1).sum() == 1, "one root"


# ---------------------------------------------------------------------------
# topology helpers
# ---------------------------------------------------------------------------


def test_branches_of_a_pure_cycle():
    """A loop has no degree!=2 node to seed a walk, so it needs its own path."""
    edges = np.array([[i, (i + 1) % 8] for i in range(8)])  # an 8-node ring
    flat, starts, lengths = T._branches(edges, 8)
    assert len(lengths) == 1
    assert set(flat.tolist()) == set(range(8))
    assert flat[0] == flat[-1], "the cycle is closed back onto its seed"


def test_branches_cover_every_node():
    for vox in (y_branch(9), self_touch_hairpin(), annulus(9, 5)):
        skel = teasar_skeletonize(vox)
        m = len(skel.nodes)
        flat, starts, lengths = T._branches(np.asarray(skel.edges), m)
        assert set(flat.tolist()) == set(range(m)), "a node fell out of every branch"
        assert (np.diff(lengths) <= 0).all(), "branches must be sorted by length"
        assert lengths.sum() == len(flat)


def test_isolated_nodes_get_a_frame():
    vox = np.array([[5, 5, 5], [50, 50, 50]], dtype=np.int64)
    skel = Skeleton(vox.astype(float), np.empty((0, 2), np.int64), None, None)
    p = T.tube_profile(vox, skel, K=2, n_theta=16)
    u, v, t = p.frame_vectors()
    assert np.abs(np.linalg.norm(t, axis=1) - 1.0).max() < 1e-9
    assert np.abs(np.einsum("ij,ij->i", u, v)).max() < 1e-9


# ---------------------------------------------------------------------------
# API surface, validation and degenerate inputs
# ---------------------------------------------------------------------------


def test_public_exports():
    assert sc.tube_coefficients is T.tube_coefficients
    assert sc.TubeProfile is T.TubeProfile
    assert sc.measure.tube_profile is T.tube_profile
    assert "tube_profile" in sc.measure.__all__
    assert {"tube_coefficients", "TubeProfile"} <= set(sc.__all__)


@pytest.mark.parametrize("method", ["teasar", "wavefront", "thin"])
def test_tube_coefficients_pipeline(method):
    vox = solid_cylinder(6, 30)
    p = sc.tube_coefficients(vox, method=method, K=4, n_theta=32, spacing=(1, 1, 1))
    assert len(p.nodes) > 0
    assert p.K == 4 and p.n_theta == 32
    assert np.isfinite(p.a0).all() and (p.a0 > 0).all()
    assert p.mag.shape == (len(p.nodes), 4)
    with pytest.raises(ValueError):
        sc.tube_coefficients(vox, method="nope")


def test_tube_coefficients_forwards_skeleton_kwargs():
    vox = y_branch(11)
    loose = sc.tube_coefficients(vox, method="teasar", n_theta=16, K=2)
    pruned = sc.tube_coefficients(vox, method="teasar", n_theta=16, K=2,
                                  min_branch_length=100)
    assert len(pruned.nodes) < len(loose.nodes)


@pytest.mark.parametrize("kwargs", [{"K": 32, "n_theta": 64}, {"n_theta": 3}, {"K": -1}])
def test_invalid_parameters(kwargs):
    vox = line(6, axis=2)
    skel = Skeleton(vox.astype(float), np.stack([np.arange(5), np.arange(1, 6)], 1),
                    None, None)
    with pytest.raises(ValueError):
        T.tube_profile(vox, skel, **kwargs)


def test_rejects_a_non_skeleton():
    with pytest.raises(TypeError):
        T.tube_profile(line(6), "not a skeleton")


def test_empty_and_single_node():
    empty_skel = Skeleton(np.empty((0, 3)), np.empty((0, 2), np.int64), None, None)
    p = T.tube_profile(np.empty((0, 3), np.int64), empty_skel, K=3)
    assert p.nodes.shape == (0, 3) and p.mag.shape == (0, 3) and p.K == 3
    assert p.to_gpu_buffer()[0].shape == (0, 14)

    p = T.tube_profile(np.array([[5, 5, 5]], np.int64), empty_skel, K=3)
    assert len(p.nodes) == 0

    one = Skeleton(np.array([[5.0, 5.0, 5.0]]), np.empty((0, 2), np.int64), None, None)
    p = T.tube_profile(np.array([[5, 5, 5]], np.int64), one, K=2, n_theta=16)
    assert p.a0.shape == (1,) and np.isfinite(p.a0).all()
    assert p.flag("branch_end")[0]


def test_sparse_array_input():
    """The `sparse_aware` contract: a 3-D coo_array works wherever voxels do."""
    sparse = pytest.importorskip("scipy.sparse")
    vox = disc_cylinder(5, 12)
    skel = axial_skeleton(vox)
    dense = T.tube_profile(vox, skel, K=2, n_theta=32, max_radius=40)

    coo = sparse.coo_array(
        (np.ones(len(vox), np.uint8), tuple(vox[:, i] for i in range(3))),
        shape=tuple(vox.max(axis=0) + 1),
    )
    np.testing.assert_array_equal(T.tube_profile(coo, skel, K=2, n_theta=32,
                                                max_radius=40).a0, dense.a0)


def test_spacing_defaults_to_the_skeletons():
    vox = disc_cylinder(5, 12)
    inherited = T.tube_profile(vox, axial_skeleton(vox, spacing=(3, 3, 3)),
                               K=2, n_theta=32, max_radius=1e4)
    explicit = T.tube_profile(vox, axial_skeleton(vox), spacing=(3, 3, 3),
                              K=2, n_theta=32, max_radius=1e4)
    np.testing.assert_allclose(inherited.a0, explicit.a0)
    assert inherited.spacing is not None


# ---------------------------------------------------------------------------
# the real neurite: cost has to stay bounded
# ---------------------------------------------------------------------------

_NEURON = os.path.join(os.path.dirname(__file__), "10075_scale3.npy")


@pytest.mark.skipif(not os.path.exists(_NEURON), reason="large fixture not present")
def test_large_neuron_stays_bounded():
    """Peak memory must track the ray block, not the node count or the bbox.

    The bound is the point: a 5.6M-voxel neurite in a 14.8-billion-cell box is
    exactly the case the library exists for, and a raycaster that materialised all
    its rays at once would need tens of gigabytes.
    """
    import time

    vox = np.load(_NEURON)
    skel = sc.wavefront_skeletonize(vox, spacing=(16, 16, 16))

    tracemalloc.start()
    t0 = time.perf_counter()
    p = T.tube_profile(vox, skel, K=4, n_theta=32)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(p.nodes) == len(skel.nodes)
    assert np.isfinite(p.a0).all()
    assert peak < 1.5 * 1024**3, f"peak {peak / 1024**3:.2f} GB"
    assert dt < 180, f"took {dt:.1f}s"
