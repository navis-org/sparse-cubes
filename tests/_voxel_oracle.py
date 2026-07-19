"""Brute-force reference implementations for checking `sparsecubes.voxelize`.

Deliberately written with *different* algorithms than the module under test, so
that agreement is evidence rather than a tautology:

  - `sat_hit`     : the classic 13-axis Akenine-Moller triangle/AABB separating
                    axis test, vs. the module's 1-plane + 3x3-projection
                    reformulation.
  - `inside_mesh` : Moller-Trumbore ray/triangle intersection plus parity, vs.
                    the module's 2D scanline rasterization.

Both are exact (no sampling, no tolerances) and brute force over every triangle,
so they are only usable on small meshes.

Everything works in *cell space*: vertices scaled by ``V / spacing + 0.5`` so that
voxel ``i`` occupies the unit cell ``[i, i + 1)``, matching `voxelize`'s
convention. Comparing in physical space instead reintroduces rounding (``grid *
spacing``) that shifts the reference cubes by an ulp and produces spurious
disagreements on grazing contacts.
"""

import numpy as np


def sat_hit(centers, tris, half=0.5):
    """Exact triangle vs AABB overlap. centers (P,3), tris (T,3,3) -> (P,) bool."""
    hit = np.zeros(len(centers), bool)
    h = np.array([half, half, half])
    for i, c in enumerate(centers):
        u = tris - c  # box now centred at the origin
        u0, u1, u2 = u[:, 0], u[:, 1], u[:, 2]
        f = [u1 - u0, u2 - u1, u0 - u2]

        ok = np.ones(len(tris), bool)
        for k in range(3):  # 3 box face normals
            ok &= ~((u[:, :, k].min(1) > h[k]) | (u[:, :, k].max(1) < -h[k]))
        for fv in f:  # 9 edge-cross axes
            for k in range(3):
                a = np.zeros_like(fv)
                p, q = (k + 1) % 3, (k + 2) % 3
                a[:, p] = -fv[:, q]
                a[:, q] = fv[:, p]
                proj = np.stack([(a * u0).sum(1), (a * u1).sum(1), (a * u2).sum(1)], 1)
                r = h[p] * np.abs(a[:, p]) + h[q] * np.abs(a[:, q])
                ok &= ~((proj.min(1) > r) | (proj.max(1) < -r))
        n = np.cross(f[0], f[1])  # triangle plane vs box
        ok &= np.abs((n * u0).sum(1)) <= (np.abs(n) * h).sum(1)
        hit[i] = ok.any()
    return hit


def inside_mesh(pts, tris):
    """Moller-Trumbore +Z ray parity. pts (P,3), tris (T,3,3) -> (P,) bool.

    Rays are nudged by an irrational-ish epsilon in X/Y so they never strike a
    shared triangle edge exactly; without that, a ray through a face diagonal is
    counted by both adjacent triangles and the parity flips. (The module under
    test solves the same problem exactly, by symbolic perturbation.)
    """
    pts = pts + np.array([3.13e-7, 7.71e-7, 0.0])
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    d = np.array([0.0, 0.0, 1.0])
    h = np.cross(d, e2)
    a = (e1 * h).sum(1)
    par = np.abs(a) < 1e-12
    f = 1.0 / np.where(par, 1.0, a)
    out = np.zeros(len(pts), bool)
    for i, p in enumerate(pts):
        s = p - v0
        u = f * (s * h).sum(1)
        q = np.cross(s, e1)
        v = f * (q * d).sum(1)
        t = f * (q * e2).sum(1)
        ok = (~par) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        out[i] = (ok.sum() % 2) == 1
    return out


def reference_bounds(mesh, spacing):
    """Exact bracket on the correct solid voxelization of `mesh`.

    Returns `(lower, upper)` sets of XYZ index tuples. `lower` uses a marginally
    shrunk cube and so is what any correct result *must* contain; `upper` uses the
    closed cube and so is what it may not exceed. They differ only for cells the
    surface merely grazes, where inclusion is genuinely a matter of convention.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces)
    spacing = np.asarray(spacing, dtype=np.float64).ravel()
    if spacing.size == 1:
        spacing = np.repeat(spacing, 3)

    u = V / spacing + 0.5  # cell space: voxel i occupies [i, i + 1)
    tris = u[F]
    lo = np.floor(u.min(0) - 2).astype(int)
    hi = np.ceil(u.max(0) + 2).astype(int)
    grid = np.stack(
        np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi)], indexing="ij"), -1
    ).reshape(-1, 3)
    ctr = grid + 0.5  # centre of the unit cell [i, i + 1)

    inside = inside_mesh(ctr, tris)
    upper = grid[sat_hit(ctr, tris, 0.5) | inside]
    lower = grid[sat_hit(ctr, tris, 0.5 - 1e-9) | inside]
    as_set = lambda a: set(map(tuple, a.tolist()))
    return as_set(lower), as_set(upper)
