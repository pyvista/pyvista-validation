"""Core type aliases."""

from __future__ import annotations

from pyvista_validation import _lazy_import

from ._array_like import _ArrayLike
from ._array_like import _ArrayLike1D
from ._array_like import _ArrayLike2D
from ._array_like import _Scalar

VectorLike = _ArrayLike1D
MatrixLike = _ArrayLike2D
ArrayLike = _ArrayLike

RotationLike = MatrixLike | _lazy_import.vtkMatrix3x3 | _lazy_import.Rotation
TransformLike = RotationLike | _lazy_import.vtkMatrix4x4 | _lazy_import.vtkTransform

# A scalar or any array-like.
_ArrayLikeOrScalar = float | _Scalar | ArrayLike
