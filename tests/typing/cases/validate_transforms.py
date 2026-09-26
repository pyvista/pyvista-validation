"""Typing cases for the transforms functions."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from type_assert import assert_types
from vtkmodules.vtkCommonMath import vtkMatrix3x3
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform

from pyvista_validation import validate_axes
from pyvista_validation import validate_rotation
from pyvista_validation import validate_transform3x3
from pyvista_validation import validate_transform4x4
from pyvista_validation._typing import RotationLike
from pyvista_validation._typing import TransformLike
from pyvista_validation._typing import _Array2D
from pyvista_validation._typing import _Scalar


def flag() -> bool:
    """Return a bool no type checker can narrow to a literal."""
    return False


def rotation_like() -> RotationLike:
    """Return a rotation typed as broadly as the parameter that accepts it."""
    return np.eye(3)


def transform_like() -> TransformLike:
    """Return a transform typed as broadly as the parameter that accepts it."""
    return np.eye(4)


assert_types(validate_axes(np.eye(3, dtype=np.int64)), _Array2D[np.float64])
assert_types(validate_axes(np.eye(3, dtype=np.int64), normalize=False), _Array2D[_Scalar])
assert_types(validate_axes([1, 0, 0], [0, 1, 0], normalize=flag()), _Array2D[_Scalar])
assert_types(validate_axes([1, 0, 0], [0, 1, 0]), _Array2D[np.float64])
assert_types(validate_axes(np.eye(3), must_have_orientation=None), _Array2D[np.float64])
assert_types(validate_rotation(np.eye(3, dtype=np.float32)), _Array2D[np.float32])
assert_types(validate_rotation([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), _Array2D[np.int64])
assert_types(
    validate_rotation(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    _Array2D[np.float64],
)
assert_types(validate_rotation(vtkMatrix3x3()), _Array2D[np.float64])
assert_types(validate_rotation(Rotation.identity()), _Array2D[np.float64])
assert_types(validate_rotation(rotation_like()), _Array2D[_Scalar])
assert_types(validate_rotation(np.eye(3), 'right'), _Array2D[np.float64])
assert_types(validate_transform3x3(np.eye(3, dtype=np.float32)), _Array2D[np.float32])
assert_types(validate_transform3x3([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), _Array2D[np.int64])
assert_types(
    validate_transform3x3(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    _Array2D[np.float64],
)
assert_types(validate_transform3x3(vtkMatrix3x3()), _Array2D[np.float64])
assert_types(validate_transform3x3(Rotation.identity()), _Array2D[np.float64])
assert_types(validate_transform3x3(rotation_like()), _Array2D[_Scalar])
assert_types(
    validate_transform4x4(np.eye(4, dtype=np.float32)),
    _Array2D[np.float32 | np.float64],
)
assert_types(
    validate_transform4x4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    _Array2D[np.float64],
)
assert_types(
    validate_transform4x4(((1.0, 0, 0, 0), (0, 1.0, 0, 0), (0, 0, 1.0, 0), (0, 0, 0, 1.0))),
    _Array2D[np.float64],
)
assert_types(validate_transform4x4(vtkMatrix3x3()), _Array2D[np.float64])
assert_types(validate_transform4x4(vtkMatrix4x4()), _Array2D[np.float64])
assert_types(validate_transform4x4(vtkTransform()), _Array2D[np.float64])
assert_types(validate_transform4x4(Rotation.identity()), _Array2D[np.float64])
assert_types(validate_transform4x4(transform_like()), _Array2D[_Scalar])
assert_types(
    validate_transform4x4(np.eye(3, dtype=np.float32)),
    _Array2D[np.float32 | np.float64],
)
assert_types(
    validate_transform4x4([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    _Array2D[np.float64],
)
