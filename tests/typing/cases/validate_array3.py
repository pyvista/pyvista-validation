"""Typing cases for validate_array3."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

if TYPE_CHECKING:
    from collections.abc import Sequence

from pyvista_validation import validate_array3
from pyvista_validation._typing import _AnyScalar
from pyvista_validation._typing import _Array1D
from pyvista_validation._typing import _Scalar

_Array3Out = (
    _Array1D[_Scalar]
    | list[bool]
    | list[int]
    | list[float]
    | tuple[bool, bool, bool]
    | tuple[int, int, int]
    | tuple[float, float, float]
)
_Array3AnyOut = (
    _Array1D[_AnyScalar]
    | list[bool]
    | list[int]
    | list[float]
    | list[str]
    | tuple[bool, bool, bool]
    | tuple[int, int, int]
    | tuple[float, float, float]
    | tuple[str, str, str]
)


def flag() -> bool:
    """Return a bool no type checker can narrow to a literal."""
    return False


assert_types(validate_array3(np.zeros(3, dtype=np.float32)), _Array1D[np.float32])
assert_types(validate_array3(np.zeros(3) > 0, must_be_real=False), _Array1D[np.bool_])
assert_types(validate_array3([True, False, True], must_be_real=False), _Array1D[np.bool_])
assert_types(validate_array3([1, 2, 3]), _Array1D[np.int64])
assert_types(validate_array3([1.5, 2.5, 3.5]), _Array1D[np.float64])
assert_types(validate_array3(['a', 'b', 'c'], must_be_real=False), _Array1D[np.str_])
assert_types(validate_array3([1, 2, 3], dtype_out=np.float32), _Array1D[np.float32])
assert_types(validate_array3([1, 2, 3], dtype_out=bool), _Array1D[np.bool_])
assert_types(validate_array3([1, 2, 3], dtype_out=int), _Array1D[np.int64])
assert_types(validate_array3([1, 2, 3], dtype_out=float), _Array1D[np.float64])
assert_types(validate_array3([1, 2, 3], dtype_out='float32'), _Array1D[_Scalar])
assert_types(
    validate_array3(['a', 'b', 'c'], must_be_real=False, dtype_out='U1'),
    _Array1D[_AnyScalar],
)
assert_types(validate_array3([True, False, True], must_be_real=False, to_list=True), list[bool])
assert_types(validate_array3([1, 2, 3], to_list=True), list[int])
assert_types(validate_array3([1.5, 2.5, 3.5], to_list=True), list[float])
assert_types(validate_array3(['a', 'b', 'c'], must_be_real=False, to_list=True), list[str])
assert_types(validate_array3([1, 2, 3], dtype_out=bool, to_list=True), list[bool])
assert_types(validate_array3([1, 2, 3], dtype_out=int, to_list=True), list[int])
assert_types(validate_array3([1, 2, 3], dtype_out=float, to_list=True), list[float])
assert_types(
    validate_array3([1, 2, 3], dtype_out='float32', to_list=True),
    list[bool] | list[int] | list[float],
)
assert_types(
    validate_array3(['a', 'b', 'c'], must_be_real=False, dtype_out='U1', to_list=True),
    list[bool] | list[int] | list[float] | list[str],
)
assert_types(
    validate_array3([True, False, True], must_be_real=False, to_tuple=True),
    tuple[bool, bool, bool],
)
assert_types(validate_array3([1, 2, 3], to_tuple=True), tuple[int, int, int])
assert_types(validate_array3([1.5, 2.5, 3.5], to_tuple=True), tuple[float, float, float])
assert_types(
    validate_array3(['a', 'b', 'c'], must_be_real=False, to_tuple=True),
    tuple[str, str, str],
)
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
assert_types(
    validate_array3(['a', 'b', 'c'], must_be_real=False, dtype_out='U1', to_tuple=True),
    (
        tuple[bool, bool, bool]
        | tuple[int, int, int]
        | tuple[float, float, float]
        | tuple[str, str, str]
    ),
)
assert_types(validate_array3([1, 2, 3], to_list=flag()), _Array3Out)
assert_types(validate_array3(['a', 'b', 'c'], must_be_real=False, to_list=flag()), _Array3AnyOut)
assert_types(validate_array3(1, broadcast=True), _Array1D[np.int64])
assert_types(validate_array3([[1, 2, 3]]), _Array1D[np.int64])
assert_types(validate_array3((1.5, 2.5, 3.5), to_tuple=True), tuple[float, float, float])


def float32_vector() -> npt.NDArray[np.float32] | Sequence[np.float32]:
    """Return a value typed as either an array or a sequence of the same scalar."""
    return np.ones(3, dtype=np.float32)


assert_types(validate_array3(float32_vector()), _Array1D[np.float32])
