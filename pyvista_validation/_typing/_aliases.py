"""Core type aliases."""

from __future__ import annotations

from typing import Union

from pyvista_validation import _lazy_import

from ._array_like import NumberType
from ._array_like import _ArrayLike
from ._array_like import _ArrayLike1D
from ._array_like import _ArrayLike2D

Number = Union[int, float]
VectorLike = _ArrayLike1D[NumberType]
MatrixLike = _ArrayLike2D[NumberType]
ArrayLike = _ArrayLike[NumberType]

RotationLike = Union[MatrixLike[float], _lazy_import.vtkMatrix3x3, _lazy_import.Rotation]
TransformLike = Union[RotationLike, _lazy_import.vtkMatrix4x4, _lazy_import.vtkTransform]

# Undocumented alias - should be expanded in docs
_ArrayLikeOrScalar = Union[NumberType, ArrayLike[NumberType]]
