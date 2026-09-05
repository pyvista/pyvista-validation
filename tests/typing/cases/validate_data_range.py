"""Typing cases for validate_data_range."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

from pyvista_validation import validate_data_range
from pyvista_validation._typing import _Scalar

_DataRangeOut = (
    npt.NDArray[_Scalar]
    | list[bool]
    | list[int]
    | list[float]
    | tuple[bool, bool]
    | tuple[int, int]
    | tuple[float, float]
)


def flag() -> bool:
    """Return a bool no type checker can narrow to a literal."""
    return False


assert_types(validate_data_range([False, True], must_be_real=False), tuple[bool, bool])
assert_types(validate_data_range([0, 1]), tuple[int, int])
assert_types(validate_data_range([0.0, 1.0]), tuple[float, float])
assert_types(validate_data_range([0, 1], dtype_out=bool), tuple[bool, bool])
assert_types(validate_data_range([0, 1], dtype_out=int), tuple[int, int])
assert_types(validate_data_range([0, 1], dtype_out=float), tuple[float, float])
assert_types(
    validate_data_range([0, 1], dtype_out='float32'),
    tuple[bool, bool] | tuple[int, int] | tuple[float, float],
)
assert_types(
    validate_data_range(np.array([0.0, 1.0], dtype=np.float32), to_list=False, to_tuple=False),
    npt.NDArray[np.float32],
)
assert_types(
    validate_data_range([False, True], must_be_real=False, to_list=False, to_tuple=False),
    npt.NDArray[np.bool_],
)
assert_types(validate_data_range([0, 1], to_list=False, to_tuple=False), npt.NDArray[np.int64])
assert_types(
    validate_data_range([0.0, 1.0], to_list=False, to_tuple=False),
    npt.NDArray[np.float64],
)
assert_types(
    validate_data_range([0, 1], dtype_out=np.float32, to_list=False, to_tuple=False),
    npt.NDArray[np.float32],
)
assert_types(
    validate_data_range([0, 1], dtype_out=bool, to_list=False, to_tuple=False),
    npt.NDArray[np.bool_],
)
assert_types(
    validate_data_range([0, 1], dtype_out=int, to_list=False, to_tuple=False),
    npt.NDArray[np.int64],
)
assert_types(
    validate_data_range([0, 1], dtype_out=float, to_list=False, to_tuple=False),
    npt.NDArray[np.float64],
)
assert_types(
    validate_data_range([0, 1], dtype_out='float32', to_list=False, to_tuple=False),
    npt.NDArray[_Scalar],
)
assert_types(validate_data_range([False, True], must_be_real=False, to_list=True), list[bool])
assert_types(validate_data_range([0, 1], to_list=True), list[int])
assert_types(validate_data_range([0.0, 1.0], to_list=True), list[float])
assert_types(validate_data_range([0, 1], dtype_out=bool, to_list=True), list[bool])
assert_types(validate_data_range([0, 1], dtype_out=int, to_list=True), list[int])
assert_types(validate_data_range([0, 1], dtype_out=float, to_list=True), list[float])
assert_types(
    validate_data_range([0, 1], dtype_out='float32', to_list=True),
    list[bool] | list[int] | list[float],
)
assert_types(validate_data_range([0, 1], to_list=flag()), _DataRangeOut)
assert_types(validate_data_range((0, 1), to_tuple=True), tuple[int, int])
assert_types(validate_data_range([0, 1], to_list=False), _DataRangeOut)
