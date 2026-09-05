"""Typing cases for validate_dimensionality."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

from pyvista_validation import validate_dimensionality
from pyvista_validation._typing import _Scalar

_DimensionalityOut = bool | int | float | npt.NDArray[_Scalar]


def flag() -> bool:
    """Return a bool no type checker can narrow to a literal."""
    return False


assert_types(validate_dimensionality(2), int)
assert_types(validate_dimensionality(2, dtype_out=bool), bool)
assert_types(validate_dimensionality(2, dtype_out=np.int32), int)
assert_types(validate_dimensionality(2, dtype_out=float), float)
assert_types(validate_dimensionality(2, dtype_out='int32'), bool | int | float)
assert_types(validate_dimensionality(2, to_list=False), npt.NDArray[np.int64])
assert_types(validate_dimensionality(2, dtype_out=np.int16, to_list=False), npt.NDArray[np.int16])
assert_types(validate_dimensionality(2, dtype_out='int32', to_list=False), npt.NDArray[_Scalar])
assert_types(validate_dimensionality(2, to_list=flag()), _DimensionalityOut)
assert_types(validate_dimensionality('2D'), int)
assert_types(validate_dimensionality([2]), int)
assert_types(validate_dimensionality(np.array([2])), int)
assert_types(validate_dimensionality(3, reshape=False), int)
assert_types(validate_dimensionality(1, to_tuple=True), int)
