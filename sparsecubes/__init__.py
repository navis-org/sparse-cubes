
from . import binary, measure
from .core import (
    mesh,
    surface_nets,
    culled_faces,
    greedy_faces,
    dual_contour,  # deprecated alias
    marching_cubes,  # deprecated alias
)
from .voxelization import voxelize
from .skeleton import thin_skeletonize, centerline, Skeleton
from .teasar import teasar_skeletonize
from .wavefront import wavefront_skeletonize
from .downsample import downsample, downsample_graph
from .graph import edges
from .__version__ import __version__

# The top level carries the end-to-end pipelines, plus the operations that change
# what the voxel set *is* rather than which voxels are in it - `edges` (voxels ->
# graph) and `downsample`/`downsample_graph` (voxels -> a coarser lattice). The
# primitives that stay on one lattice live in the two submodules:
#   sparsecubes.binary  - voxel set(s) -> voxel set (morphology, set algebra)
#   sparsecubes.measure - voxel set -> numbers / labels
__all__ = [
    "binary",
    "measure",
    "mesh",
    "surface_nets",
    "culled_faces",
    "greedy_faces",
    "dual_contour",
    "marching_cubes",
    "voxelize",
    "thin_skeletonize",
    "centerline",
    "teasar_skeletonize",
    "wavefront_skeletonize",
    "edges",
    "downsample",
    "downsample_graph",
    "Skeleton",
    "__version__",
]

# `thin` and `fill_cavities` moved to `sparsecubes.binary` in 0.4.0 - they are
# primitives, not pipelines. Point the old spellings at the new home rather than
# letting them fail with a bare AttributeError.
_MOVED = {"thin": "binary.thin", "fill_cavities": "binary.fill_cavities"}


def __getattr__(name):
    if name in _MOVED:
        raise AttributeError(
            f"`sparsecubes.{name}` moved to `sparsecubes.{_MOVED[name]}` in 0.4.0. "
            f"Use `sc.{_MOVED[name]}(...)`."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
