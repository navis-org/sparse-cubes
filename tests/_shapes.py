"""Shared fixtures and topology helpers for the thinning / skeleton tests."""

import numpy as np

from sparsecubes.skeleton import _edges_26


# --- synthetic voxel shapes ------------------------------------------------

def line(n=8, axis=0):
    """A straight 1-voxel-wide line of length `n` along `axis`."""
    v = np.zeros((n, 3), dtype=np.int64)
    v[:, axis] = np.arange(n)
    return v


def l_shape():
    """A right-angle bend (two straight arms sharing a corner)."""
    a = line(8, axis=0)
    b = line(8, axis=1)
    return np.unique(np.vstack([a, b]), axis=0)


def y_branch(arm=7):
    """Three straight arms meeting at the origin (a clean degree-3 junction)."""
    arms = [np.zeros((arm, 3), dtype=np.int64) for _ in range(3)]
    for ax in range(3):
        arms[ax][:, ax] = np.arange(1, arm + 1)
    junction = np.zeros((1, 3), dtype=np.int64)
    return np.unique(np.vstack([junction, *arms]), axis=0)


def solid_cube(k):
    """A solid k x k x k cube."""
    g = np.indices((k, k, k)).reshape(3, -1).T
    return g.astype(np.int64)


def slab(k):
    """A flat k x k x 1 slab in the z=0 plane."""
    xs, ys = np.meshgrid(np.arange(k), np.arange(k))
    return np.stack([xs.ravel(), ys.ravel(), np.zeros(k * k, dtype=int)], axis=1).astype(
        np.int64
    )


def solid_cylinder(radius=4, length=20):
    """A solid cylinder of the given radius running along +z."""
    r = radius
    rng = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(rng, rng)
    disk = np.stack([xx.ravel(), yy.ravel()], axis=1)
    disk = disk[(disk[:, 0] ** 2 + disk[:, 1] ** 2) <= r * r]
    out = []
    for z in range(length):
        col = np.column_stack([disk[:, 0] + r, disk[:, 1] + r, np.full(len(disk), z)])
        out.append(col)
    return np.vstack(out).astype(np.int64)


def annulus(outer=9, inner=5):
    """A flat ring (annulus) in the z=0 plane - a shape with one loop."""
    rng = np.arange(-outer, outer + 1)
    xx, yy = np.meshgrid(rng, rng)
    d2 = xx.ravel() ** 2 + yy.ravel() ** 2
    keep = (d2 <= outer * outer) & (d2 >= inner * inner)
    pts = np.stack([xx.ravel()[keep] + outer, yy.ravel()[keep] + outer], axis=1)
    return np.column_stack([pts, np.zeros(len(pts), dtype=int)]).astype(np.int64)


# --- topology helpers ------------------------------------------------------

def _union_find(m, edges):
    parent = list(range(m))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    roots = {find(i) for i in range(m)}
    return len(roots)


def betti(nodes, edges):
    """Return (n_components, first_betti) of the 26-connectivity graph."""
    m = len(nodes)
    if m == 0:
        return 0, 0
    c = _union_find(m, edges)
    return c, len(edges) - m + c


def n_components(voxels):
    """Number of 26-connected components of a raw voxel set."""
    voxels = np.asarray(voxels)
    if len(voxels) == 0:
        return 0
    nodes, edges = _edges_26(voxels.astype(np.int64))
    return betti(nodes, edges)[0]


def has_2x2x2_block(voxels):
    """True if the voxel set contains a fully occupied 2x2x2 block."""
    s = set(map(tuple, np.asarray(voxels).tolist()))
    for x, y, z in s:
        if all(
            (x + dx, y + dy, z + dz) in s
            for dx in (0, 1)
            for dy in (0, 1)
            for dz in (0, 1)
        ):
            return True
    return False


def is_subset(sub, full):
    """True if every row of `sub` is a row of `full`."""
    fs = set(map(tuple, np.asarray(full).tolist()))
    return all(tuple(r) in fs for r in np.asarray(sub).tolist())
