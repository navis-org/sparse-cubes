"""Shared mesh fixtures for the voxelization tests.

`_shapes.py` holds voxel sets; these are the *mesh* side, including the
deliberately broken ones. They live here rather than in a test module because
both `test_voxelize.py` and `test_binary.py` need the same defects - and the
same pitch, since the assertions on both sides are resolution-dependent.
"""

import warnings

import numpy as np
import trimesh as tm

import sparsecubes as sc

# Fine enough that a defect spans several voxels, coarse enough to stay quick.
PITCH = 0.08


def sphere():
    """The unit icosphere every defect below is derived from."""
    return tm.creation.icosphere(subdivisions=3, radius=1.0)


def _dup_faces(s):
    """200 faces listed twice - keeps the crossing count even, the winding not."""
    return tm.Trimesh(s.vertices, np.vstack([s.faces, s.faces[:200]]), process=False)


def _flipped(s):
    """Half the faces wound the other way: inconsistent normals."""
    f = s.faces.copy()
    flip = np.random.default_rng(1).random(len(f)) < 0.5
    f[flip] = f[flip][:, ::-1]
    return tm.Trimesh(s.vertices, f, process=False)


def _overlap(s):
    """Two solid spheres interpenetrating - a self-intersection."""
    a, b = sphere(), sphere()
    b.apply_translation([1.2, 0, 0])
    return tm.util.concatenate([a, b])


def _nested(s):
    """A second closed shell entirely inside the first."""
    return tm.util.concatenate([s, tm.creation.icosphere(subdivisions=3, radius=0.5)])


def _cap_off(s):
    """One big hole: every face above z = 0.75 removed."""
    return tm.Trimesh(s.vertices, s.faces[s.triangles_center[:, 2] < 0.75], process=False)


def _shredded(s):
    """10% of faces deleted at random - many small holes."""
    keep = np.random.default_rng(0).random(len(s.faces)) > 0.10
    return tm.Trimesh(s.vertices, s.faces[keep], process=False)


DEFECTS = {
    "dup_faces": _dup_faces,
    "flipped": _flipped,
    "overlap": _overlap,
    "nested": _nested,
    "cap_off": _cap_off,
    "shredded": _shredded,
}


def broken(kind):
    """A unit icosphere with one named defect introduced."""
    return DEFECTS[kind](sphere())


def quiet_voxelize(mesh, spacing=PITCH, **kwargs):
    """`sc.voxelize` with the broken-mesh warnings suppressed.

    Every defect here warns by design; the tests that assert on the *warnings*
    capture them explicitly instead of going through this.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sc.voxelize(mesh, spacing, **kwargs)
