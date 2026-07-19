"""Transparent interop with 3-D sparse arrays (`scipy.sparse.coo_array`).

A 3-D sparse array and this library's ``(N, 3)`` index arrays are the same thing
wearing different clothes - a COO volume *is* a list of occupied coordinates. The
`sparse_aware` decorator makes that swap invisible: hand any voxel-taking function
a 3-D `scipy.sparse` array and it works, and the operations that return a voxel
set hand you one back in the same form.

**scipy is never imported here.** It stays an optional, lazily-imported
dependency, so detection is pure duck-typing on the type's module name, and the
extraction (`.tocoo()`, `.coords`, `.data`) only ever calls methods on the object
the caller already handed us. When we do need the module to *build* a result, we
read it out of `sys.modules` - a sparse argument is itself proof that scipy is
already imported, so that lookup can never trigger one.

Scope: scipy supports 3-D in the COO format only (CSR/DOK/LIL remain 2-D as of
scipy 1.15), so a 3-D input is always COO and results come back as `coo_array`.
Two-dimensional matrices are rejected with a pointed message - they cannot
represent a volume.
"""

import functools
import sys

import numpy as np

__all__ = ["sparse_aware", "is_sparse", "to_voxels", "to_sparse_like"]

# Type-module prefixes we recognise as sparse containers. Matching on the name
# keeps this import-free; `scipy.sparse.coo_array` lives in `scipy.sparse._coo`.
_SPARSE_MODULES = ("scipy.sparse",)


def is_sparse(obj):
    """Does `obj` look like a sparse array? Duck-typed; never imports scipy.

    Deliberately conservative: an ndarray is never sparse, and we require the
    `tocoo` method we actually rely on, so a false positive would have to be a
    scipy.sparse-module object that already behaves like one.
    """
    if obj is None or isinstance(obj, np.ndarray):
        return False
    module = getattr(type(obj), "__module__", "") or ""
    return module.startswith(_SPARSE_MODULES) and hasattr(obj, "tocoo")


def to_voxels(obj):
    """Extract the ``(N, 3)`` int64 coordinates of a 3-D sparse array's occupied cells.

    Stored-but-zero entries are dropped, so occupancy follows the array's values
    rather than its storage pattern. Duplicate coordinates are left alone - every
    consumer in this library deduplicates anyway.
    """
    coo = obj.tocoo()
    ndim = getattr(coo, "ndim", None)
    if ndim != 3:
        raise TypeError(
            f"Expected a 3-D sparse array (a volume), got {ndim}-D "
            f"{type(obj).__name__}. scipy's 2-D matrix formats (csr/csc/coo_matrix) "
            "cannot represent a volume - build a 3-D `scipy.sparse.coo_array`, or "
            "pass an (N, 3) array of voxel indices directly."
        )
    coords = np.stack([np.asarray(c) for c in coo.coords], axis=1)
    data = np.asarray(coo.data)
    if data.size:
        nonzero = data != 0
        if not nonzero.all():
            coords = coords[nonzero]
    return np.ascontiguousarray(coords, dtype=np.int64)


def to_sparse_like(voxels, template):
    """Rebuild an ``(N, 3)`` voxel array as a sparse array mirroring `template`.

    The template's `shape` acts as a *floor*, not a clamp: coordinates are absolute
    and are never shifted or dropped, so an operation that grew the object past the
    original bounds (a `dilate`, say) widens the shape to fit rather than silently
    truncating. Values are ones in the template's dtype.

    Negative coordinates have no representation in a sparse array and raise -
    those only arise from growing an object that touched index 0, and shifting the
    frame to accommodate them would silently invalidate every other coordinate.
    """
    module = sys.modules.get("scipy.sparse")
    if module is None:  # pragma: no cover - the template proves scipy is imported
        raise RuntimeError(
            "Cannot rebuild a sparse result: `scipy.sparse` is not imported, yet a "
            "sparse input came from it. This should not happen."
        )

    if len(voxels) == 0:
        empty = np.empty(0, dtype=np.int64)
        return module.coo_array(
            (np.empty(0, dtype=template.dtype), (empty, empty, empty)),
            shape=tuple(template.shape),
        )

    low = voxels.min(axis=0)
    if (low < 0).any():
        raise ValueError(
            f"Result contains negative coordinates (min {tuple(int(v) for v in low)}), "
            "which a sparse array cannot represent. This happens when an operation "
            "grows an object that touches index 0. Pass an (N, 3) index array "
            "instead - those are unbounded - or pad your sparse array first."
        )

    shape = tuple(
        int(max(s, m + 1)) for s, m in zip(template.shape, voxels.max(axis=0))
    )
    data = np.ones(len(voxels), dtype=template.dtype)
    coords = tuple(np.ascontiguousarray(voxels[:, i]) for i in range(3))
    out = module.coo_array((data, coords), shape=shape)

    # Mirror the input's storage format where scipy can (3-D is COO-only today, so
    # this is a no-op now and a free upgrade if that changes).
    fmt = getattr(template, "format", None)
    if fmt and fmt != "coo":
        try:
            out = out.asformat(fmt)
        except (ValueError, TypeError, NotImplementedError):
            pass
    return out


def _looks_like_voxels(result):
    """Is `result` an ``(N, 3)`` integer array, i.e. something we can re-wrap?"""
    return (
        isinstance(result, np.ndarray)
        and result.ndim == 2
        and result.shape[1] == 3
        and np.issubdtype(result.dtype, np.integer)
    )


def sparse_aware(func=None, *, mirror=False):
    """Let a voxel-taking function accept 3-D sparse arrays.

    Every argument that looks sparse is converted to ``(N, 3)`` indices before the
    call, so variadic and multi-input functions (`union`, `difference`, ...) are
    handled without naming their parameters. Non-sparse arguments pass through
    untouched, and a call with no sparse argument costs one `isinstance` per
    argument.

    Parameters
    ----------
    mirror :    bool
                If True *and* at least one argument was sparse, an ``(N, 3)``
                integer result is rebuilt as a sparse array modelled on the first
                sparse argument (see `to_sparse_like`). Set it on the operations
                that return a voxel set; leave it off for those returning meshes,
                skeletons, labels or scalars, which have no sparse form.
    """

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            template = None

            def convert(value):
                nonlocal template
                if is_sparse(value):
                    if template is None:
                        template = value
                    return to_voxels(value)
                return value

            args = tuple(convert(a) for a in args)
            kwargs = {k: convert(v) for k, v in kwargs.items()}
            result = fn(*args, **kwargs)

            if mirror and template is not None and _looks_like_voxels(result):
                return to_sparse_like(result, template)
            return result

        return wrapper

    return decorate if func is None else decorate(func)
