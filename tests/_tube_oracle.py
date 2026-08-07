"""A deliberately naive first-exit ray oracle for `sparsecubes.tube`.

`tube._raycast` is an Amanatides-Woo DDA: it never evaluates a point on the ray,
only the parametric distances at which the ray crosses cell faces. This oracle
does the opposite - it walks the ray in tiny fixed steps, rounds each sample to a
cell and looks it up in a plain Python ``set`` of coordinate tuples. It shares no
arithmetic, no packing scheme and no data structure with the implementation, so
agreement between the two is evidence rather than tautology (the same standard
`_voxel_oracle.py` holds `voxelize` to).

Being a sampled walk, it brackets rather than matches: the exit it reports is the
first sample *outside* the object, so it overshoots the true face crossing by at
most one step. Compare with ``atol`` of roughly `step`.
"""

import numpy as np


def brute_force_radii(voxels, origins, dirs, max_dist, step=0.02):
    """First-exit distance of each ray, by fine fixed-step sampling.

    Parameters
    ----------
    voxels :    (N, 3) integer voxel coordinates.
    origins :   (R, 3) float ray origins, index space.
    dirs :      (R, 3) float ray directions, index space. The returned distance is
                in units of the *parameter*, i.e. of ``|dirs|``, matching
                `tube._raycast`.
    max_dist :  float, ceiling on the parameter.
    step :      float, sampling interval along the parameter.

    Returns
    -------
    (R,) float array. Rays that never leave the object report `max_dist`.
    """
    occupied = {tuple(int(c) for c in row) for row in np.asarray(voxels)}
    origins = np.asarray(origins, dtype=float)
    dirs = np.asarray(dirs, dtype=float)

    out = np.full(len(origins), float(max_dist))
    n_steps = int(np.ceil(max_dist / step)) + 1
    for i in range(len(origins)):
        p0, d = origins[i], dirs[i]
        for j in range(n_steps):
            t = j * step
            p = p0 + t * d
            cell = (int(round(p[0])), int(round(p[1])), int(round(p[2])))
            if cell not in occupied:
                out[i] = t
                break
    return out


def ring_directions(u, v, n_theta, inv_spacing):
    """The `n_theta` index-space ray directions of one node's cross-section.

    Mirrors what `tube.tube_profile` hands `_raycast`: a unit direction in physical
    space, divided by the spacing to get into index space.
    """
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    d = np.cos(theta)[:, None] * np.asarray(u)[None, :] + np.sin(theta)[:, None] * np.asarray(v)[None, :]
    return d * np.asarray(inv_spacing)
