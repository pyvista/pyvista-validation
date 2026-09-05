"""Typing cases for validate_array3."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

from pyvista_validation import validate_array3
from pyvista_validation._typing import _Scalar

_Array3Out = (
    npt.NDArray[_Scalar]
    | list[bool]
    | list[int]
    | list[float]
    | tuple[bool, bool, bool]
    | tuple[int, int, int]
    | tuple[float, float, float]
)


def flag() -> bool:
    """Return a bool no type checker can narrow to a literal."""
    return False


assert_types(validate_array3(np.zeros(3, dtype=np.float32)), npt.NDArray[np.float32])
assert_types(validate_array3([True, False, True], must_be_real=False), npt.NDArray[np.bool_])
assert_types(validate_array3([1, 2, 3]), npt.NDArray[np.int64])
assert_types(validate_array3([1.5, 2.5, 3.5]), npt.NDArray[np.float64])
assert_types(validate_array3([1, 2, 3], dtype_out=np.float32), npt.NDArray[np.float32])
assert_types(validate_array3([1, 2, 3], dtype_out=bool), npt.NDArray[np.bool_])
assert_types(validate_array3([1, 2, 3], dtype_out=int), npt.NDArray[np.int64])
assert_types(validate_array3([1, 2, 3], dtype_out=float), npt.NDArray[np.float64])
assert_types(validate_array3([1, 2, 3], dtype_out='float32'), npt.NDArray[_Scalar])
assert_types(validate_array3([True, False, True], must_be_real=False, to_list=True), list[bool])
assert_types(validate_array3([1, 2, 3], to_list=True), list[int])
assert_types(validate_array3([1.5, 2.5, 3.5], to_list=True), list[float])
assert_types(validate_array3([1, 2, 3], dtype_out=bool, to_list=True), list[bool])
assert_types(validate_array3([1, 2, 3], dtype_out=int, to_list=True), list[int])
assert_types(validate_array3([1, 2, 3], dtype_out=float, to_list=True), list[float])
assert_types(
    validate_array3([1, 2, 3], dtype_out='float32', to_list=True),
    list[bool] | list[int] | list[float],
)
assert_types(
    validate_array3([True, False, True], must_be_real=False, to_tuple=True),
    tuple[bool, bool, bool],
)
assert_types(validate_array3([1, 2, 3], to_tuple=True), tuple[int, int, int])
assert_types(validate_array3([1.5, 2.5, 3.5], to_tuple=True), tuple[float, float, float])
assert_types(validate_array3([1, 2, 3], dtype_out=bool, to_tuple=True), tuple[bool, bool, bool])
assert_types(validate_array3([1, 2, 3], dtype_out=int, to_tuple=True), tuple[int, int, int])
assert_types(
    validate_array3([1, 2, 3], dtype_out=float, to_tuple=True),
    tuple[float, float, float],
)
assert_types(
    validate_array3([1, 2, 3], dtype_out='float32', to_tuple=True),
    tuple[bool, bool, bool] | tuple[int, int, int] | tuple[float, float, float],
)
assert_types(validate_array3([1, 2, 3], to_list=flag()), _Array3Out)
assert_types(validate_array3(1, broadcast=True), npt.NDArray[np.int64])
assert_types(validate_array3([[1, 2, 3]]), npt.NDArray[np.int64])
assert_types(validate_array3((1.5, 2.5, 3.5), to_tuple=True), tuple[float, float, float])
