"""Typing cases for validate_data_range."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

if TYPE_CHECKING:
    from collections.abc import Sequence

from pyvista_validation import validate_data_range
from pyvista_validation._typing import _AnyScalar
from pyvista_validation._typing import _Array1D
from pyvista_validation._typing import _Scalar

_DataRangeOut = (
    _Array1D[_Scalar]
    | list[bool]
    | list[int]
    | list[float]
    | tuple[bool, bool]
    | tuple[int, int]
    | tuple[float, float]
)
_DataRangeAnyOut = (
    _Array1D[_AnyScalar]
    | list[bool]
    | list[int]
    | list[float]
    | list[str]
    | tuple[bool, bool]
    | tuple[int, int]
    | tuple[float, float]
    | tuple[str, str]
)


def flag() -> bool:
    """Return a bool no type checker can narrow to a literal."""
    return False


assert_types(validate_data_range([False, True], must_be_real=False), tuple[bool, bool])
assert_types(validate_data_range([0, 1]), tuple[int, int])
assert_types(validate_data_range([0.0, 1.0]), tuple[float, float])
assert_types(validate_data_range(['a', 'b'], must_be_real=False), tuple[str, str])
assert_types(validate_data_range([0, 1], dtype_out=bool), tuple[bool, bool])
assert_types(validate_data_range([0, 1], dtype_out=int), tuple[int, int])
assert_types(validate_data_range([0, 1], dtype_out=float), tuple[float, float])
assert_types(
    validate_data_range([0, 1], dtype_out='float32'),
    tuple[bool, bool] | tuple[int, int] | tuple[float, float],
)
assert_types(
    validate_data_range(['a', 'b'], must_be_real=False, dtype_out='U1'),
    tuple[bool, bool] | tuple[int, int] | tuple[float, float] | tuple[str, str],
)
assert_types(
    validate_data_range(np.array([0.0, 1.0], dtype=np.float32), to_list=False, to_tuple=False),
    _Array1D[np.float32],
)
assert_types(
    validate_data_range(np.zeros(2) > 0, must_be_real=False, to_list=False, to_tuple=False),
    _Array1D[np.bool_],
)
assert_types(
    validate_data_range([False, True], must_be_real=False, to_list=False, to_tuple=False),
    _Array1D[np.bool_],
)
assert_types(validate_data_range([0, 1], to_list=False, to_tuple=False), _Array1D[np.int64])
assert_types(
    validate_data_range([0.0, 1.0], to_list=False, to_tuple=False),
    _Array1D[np.float64],
)
assert_types(
    validate_data_range(['a', 'b'], must_be_real=False, to_list=False, to_tuple=False),
    _Array1D[np.str_],
)
assert_types(
    validate_data_range([0, 1], dtype_out=np.float32, to_list=False, to_tuple=False),
    _Array1D[np.float32],
)
assert_types(
    validate_data_range([0, 1], dtype_out=bool, to_list=False, to_tuple=False),
    _Array1D[np.bool_],
)
assert_types(
    validate_data_range([0, 1], dtype_out=int, to_list=False, to_tuple=False),
    _Array1D[np.int64],
)
assert_types(
    validate_data_range([0, 1], dtype_out=float, to_list=False, to_tuple=False),
    _Array1D[np.float64],
)
assert_types(
    validate_data_range([0, 1], dtype_out='float32', to_list=False, to_tuple=False),
    _Array1D[_Scalar],
)
assert_types(
    validate_data_range(
        ['a', 'b'], must_be_real=False, dtype_out='U1', to_list=False, to_tuple=False
    ),
    _Array1D[_AnyScalar],
)
assert_types(validate_data_range([False, True], must_be_real=False, to_list=True), list[bool])
assert_types(validate_data_range([0, 1], to_list=True), list[int])
assert_types(validate_data_range([0.0, 1.0], to_list=True), list[float])
assert_types(validate_data_range(['a', 'b'], must_be_real=False, to_list=True), list[str])
assert_types(validate_data_range([0, 1], dtype_out=bool, to_list=True), list[bool])
assert_types(validate_data_range([0, 1], dtype_out=int, to_list=True), list[int])
assert_types(validate_data_range([0, 1], dtype_out=float, to_list=True), list[float])
assert_types(
    validate_data_range([0, 1], dtype_out='float32', to_list=True),
    list[bool] | list[int] | list[float],
)
assert_types(
    validate_data_range(['a', 'b'], must_be_real=False, dtype_out='U1', to_list=True),
    list[bool] | list[int] | list[float] | list[str],
)
assert_types(validate_data_range([0, 1], to_list=flag()), _DataRangeOut)
assert_types(validate_data_range(['a', 'b'], must_be_real=False, to_list=flag()), _DataRangeAnyOut)
assert_types(validate_data_range((0, 1), to_tuple=True), tuple[int, int])
assert_types(validate_data_range([0, 1], to_list=False), _DataRangeOut)


def float32_range() -> npt.NDArray[np.float32] | Sequence[np.float32]:
    """Return a value typed as either an array or a sequence of the same scalar."""
    return np.array([0.0, 1.0], dtype=np.float32)


assert_types(
    validate_data_range(float32_range(), to_list=False, to_tuple=False), _Array1D[np.float32]
)
