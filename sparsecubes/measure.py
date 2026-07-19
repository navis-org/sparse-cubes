"""Measurements and labelling on sparse ``(N, 3)`` voxel clouds.

Everything in this module reduces a voxel set to numbers or per-voxel labels.
The companion `sparsecubes.binary` holds the operations that map voxel set(s)
back to a voxel set.

As everywhere else in the library, no dense grid is allocated: component
labelling runs on the packed-key adjacency graph, and the distance transform
uses the sparse background shell rather than a densified EDT.
"""

import numpy as np

from ._keys import (
    as_spacing,
    key_deltas,
    neighbor_offsets,
    sorted_hit,
    to_keys,
    unique,
    validate,
)
from .core import _component_labels, boundary_shell, pack

from ._sparse import sparse_aware

__all__ = [
    "connected_components",
    "largest_component",
    "remove_small_objects",
    "volume",
    "surface_area",
    "bounding_box",
    "centroid",
    "distance_transform",
]


def _labels_for_rows(voxels, connectivity):
    """``(n_components, labels)`` with `labels` aligned to the caller's rows.

    Components are computed on the deduplicated key set, then mapped back onto
    the input row order so duplicate rows all receive the same label.
    """
    if connectivity not in (6, 26):
        raise ValueError(
            f"connectivity must be 6 or 26 for component labelling, got "
            f"{connectivity!r} (18 is not supported by the underlying routine)"
        )
    keys, shift = to_keys(voxels, margin=1)
    if len(keys) == 0:
        return 0, np.zeros(len(voxels), dtype=np.int64)
    n_comp, labels = _component_labels(keys, connectivity)
    labels = np.asarray(labels, dtype=np.int64)
    own = pack(voxels.astype(np.int64, copy=False) - shift)
    return n_comp, labels[np.searchsorted(keys, own)]


@sparse_aware
def connected_components(voxels, connectivity=26):
    """Label the connected components of a voxel set.

    Parameters
    ----------
    voxels :        (N, 3) integer array of XYZ voxel coordinates.
    connectivity :  {6, 26}, optional
                    26 (default) treats corner-touching voxels as connected;
                    6 requires a shared face.

    Returns
    -------
    n_components :  int
    labels :        (N,) int64, aligned with `voxels` row for row (so duplicate
                    rows share a label). Values are ``0 .. n_components - 1``.

    Notes
    -----
    Needs `scipy` or the optional `dijkstra3d_sparse` accelerator, which is used
    preferentially when installed.
    """
    n_comp, labels = _labels_for_rows(voxels, connectivity)
    return n_comp, labels


@sparse_aware(mirror=True)
def largest_component(voxels, connectivity=26):
    """The voxels of the single largest connected component.

    Ties are broken toward the lowest label. Returns an empty ``(0, 3)`` array
    for empty input. See `connected_components` for the parameters.
    """
    n_comp, labels = _labels_for_rows(voxels, connectivity)
    if n_comp == 0:
        return voxels[:0]
    keep = labels == np.argmax(np.bincount(labels, minlength=n_comp))
    return voxels[keep]


@sparse_aware(mirror=True)
def remove_small_objects(voxels, min_size, connectivity=26):
    """Drop connected components smaller than `min_size` voxels.

    The sparse counterpart of `skimage.morphology.remove_small_objects` - the
    standard way to despeckle a segmentation before meshing or skeletonizing.

    Parameters
    ----------
    voxels :        (N, 3) integer array of XYZ voxel coordinates.
    min_size :      int. Components with *fewer* than this many voxels are
                    removed; components of exactly `min_size` are kept.
    connectivity :  {6, 26}, optional. See `connected_components`.

    Returns
    -------
    (M, 3) array of the input dtype, ``M <= N``, in the input's row order.
    """
    min_size = int(min_size)
    n_comp, labels = _labels_for_rows(voxels, connectivity)
    if n_comp == 0:
        return voxels[:0]
    sizes = np.bincount(labels, minlength=n_comp)
    return voxels[sizes[labels] >= min_size]


@sparse_aware
def volume(voxels, spacing=None):
    """Total occupied volume: the number of *unique* voxels times a cell's volume.

    Parameters
    ----------
    voxels :    (N, 3) integer array of XYZ voxel coordinates.
    spacing :   scalar or length-3, optional. Physical size of a voxel. Without
                it the result is a plain voxel count.

    Returns
    -------
    float
    """
    validate(voxels)
    n = len(unique(voxels, axis=0)) if len(voxels) else 0
    s = as_spacing(spacing)
    return float(n) * (float(np.prod(s)) if s is not None else 1.0)


@sparse_aware
def surface_area(voxels, spacing=None):
    """Area of the object's exposed surface (its blocky, staircase surface).

    Counts the voxel faces with no occupied neighbour behind them - the same
    faces `culled_faces` would mesh - and weights each by its physical area. Note
    this is the *blocky* area, so a sphere measures above its analytic surface
    area (the usual staircase overestimate); it is not the area of the smoothed
    `surface_nets` mesh.

    Parameters
    ----------
    voxels :    (N, 3) integer array of XYZ voxel coordinates.
    spacing :   scalar or length-3, optional. Physical size of a voxel. Without
                it the result is a plain count of exposed faces.

    Returns
    -------
    float
    """
    keys, _ = to_keys(voxels, margin=1)
    if len(keys) == 0:
        return 0.0
    s = as_spacing(spacing)
    sx, sy, sz = (1.0, 1.0, 1.0) if s is None else tuple(float(v) for v in s)
    # A face normal to x has area sy*sz, and so on.
    axis_area = np.array([sy * sz, sx * sz, sx * sy])

    total = 0.0
    for off, delta in zip(neighbor_offsets(6), key_deltas(6)):
        axis = int(np.argmax(np.abs(off)))
        exposed = int((~sorted_hit(keys + delta, keys)).sum())
        total += exposed * axis_area[axis]
    return float(total)


@sparse_aware
def bounding_box(voxels):
    """Inclusive index bounds of the voxel set.

    Returns
    -------
    (2, 3) integer array ``[[xmin, ymin, zmin], [xmax, ymax, zmax]]``, both ends
    inclusive (i.e. index space, not physical extents). Raises on empty input,
    which has no meaningful bounds.
    """
    validate(voxels)
    if len(voxels) == 0:
        raise ValueError("bounding_box() is undefined for an empty voxel set")
    return np.stack([voxels.min(axis=0), voxels.max(axis=0)]).astype(voxels.dtype)


@sparse_aware
def centroid(voxels, spacing=None):
    """Centre of mass of the *unique* voxels, scaled by `spacing` when given.

    Returns
    -------
    (3,) float array. Raises on empty input.
    """
    validate(voxels)
    if len(voxels) == 0:
        raise ValueError("centroid() is undefined for an empty voxel set")
    c = unique(voxels, axis=0).astype(float).mean(axis=0)
    s = as_spacing(spacing)
    return c * s if s is not None else c


@sparse_aware
def distance_transform(voxels, spacing=None, workers=-1):
    """Exact distance from each voxel to the nearest background voxel.

    The sparse Euclidean distance transform: `core.boundary_shell` gives the
    empty cells hugging the object's surface, and a KD-tree returns each voxel's
    distance to the nearest one. Memory tracks the surface plus the voxel count -
    never the bounding-box volume a dense EDT (`scipy.ndimage.distance_transform_edt`)
    would need.

    A surface voxel sits one voxel from its shell, so the minimum is ``~1``
    (scaled by `spacing`) rather than 0 - even a one-voxel-thick neurite gets a
    positive radius. This is the same field `teasar_skeletonize` uses for radii.

    Parameters
    ----------
    voxels :    (N, 3) integer array of XYZ voxel coordinates.
    spacing :   scalar or length-3, optional. Anisotropy is applied by scaling
                coordinates before the query.
    workers :   int, optional. Threads for the KD-tree query, which dominates the
                call. ``-1`` (default) uses every core; ``1`` disables threading -
                set that when parallelizing over objects yourself. Speed only, the
                distances are identical. Needs scipy >= 1.6.

    Returns
    -------
    (N,) float array, aligned with `voxels` row for row.

    Notes
    -----
    Needs `scipy`; install ``sparse-cubes[skeleton]``.
    """
    validate(voxels)
    if len(voxels) == 0:
        return np.zeros(0, dtype=float)
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError(
            "distance_transform() requires scipy; install `sparse-cubes[skeleton]` "
            "or `pip install scipy`."
        ) from exc

    s = as_spacing(spacing)
    shell = boundary_shell(voxels).astype(float)
    pts = voxels.astype(float)
    if s is not None:
        shell = shell * s
        pts = pts * s
    tree = cKDTree(shell)
    try:  # `workers` (scipy >= 1.6) is a large win - the query dominates the cost
        dist, _ = tree.query(pts, k=1, workers=workers)
    except TypeError:  # pragma: no cover - older scipy
        dist, _ = tree.query(pts, k=1)
    return np.asarray(dist, dtype=float)
