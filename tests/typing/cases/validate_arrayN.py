"""Typing cases for the arrayN functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

from pyvista_validation import validate_arrayN
from pyvista_validation import validate_arrayN_unsigned
from pyvista_validation._typing import _Integer
from pyvista_validation._typing import _Scalar

_ArrayNOut = (
    npt.NDArray[_Scalar]
    | list[bool]
    | list[int]
    | list[float]
    | tuple[bool, ...]
    | tuple[int, ...]
    | tuple[float, ...]
)
_ArrayNUnsignedOut = npt.NDArray[_Integer] | list[int] | tuple[int, ...]


def flag() -> bool:
    """Return a bool no type checker can narrow to a literal."""
    return False


assert_types(validate_arrayN(np.zeros(3, dtype=np.float32)), npt.NDArray[np.float32])
assert_types(validate_arrayN([]), npt.NDArray[np.float64])
assert_types(validate_arrayN([True, False], must_be_real=False), npt.NDArray[np.bool_])
assert_types(validate_arrayN([1, 2]), npt.NDArray[np.int64])
assert_types(validate_arrayN([1.5, 2.5]), npt.NDArray[np.float64])
assert_types(validate_arrayN([1, 2], dtype_out=np.float32), npt.NDArray[np.float32])
assert_types(validate_arrayN([1, 2], dtype_out=bool), npt.NDArray[np.bool_])
assert_types(validate_arrayN([1, 2], dtype_out=int), npt.NDArray[np.int64])
assert_types(validate_arrayN([1, 2], dtype_out=float), npt.NDArray[np.float64])
assert_types(validate_arrayN([1, 2], dtype_out='float32'), npt.NDArray[_Scalar])
assert_types(validate_arrayN([], to_list=True), list[float])
assert_types(validate_arrayN([True, False], must_be_real=False, to_list=True), list[bool])
assert_types(validate_arrayN([1, 2], to_list=True), list[int])
assert_types(validate_arrayN([1.5, 2.5], to_list=True), list[float])
assert_types(validate_arrayN([1, 2], dtype_out=bool, to_list=True), list[bool])
assert_types(validate_arrayN([1, 2], dtype_out=int, to_list=True), list[int])
assert_types(validate_arrayN([1, 2], dtype_out=float, to_list=True), list[float])
assert_types(
    validate_arrayN([1, 2], dtype_out='float32', to_list=True),
    list[bool] | list[int] | list[float],
)
assert_types(validate_arrayN([], to_tuple=True), tuple[float, ...])
assert_types(validate_arrayN([True, False], must_be_real=False, to_tuple=True), tuple[bool, ...])
assert_types(validate_arrayN([1, 2], to_tuple=True), tuple[int, ...])
assert_types(validate_arrayN([1.5, 2.5], to_tuple=True), tuple[float, ...])
assert_types(validate_arrayN([1, 2], dtype_out=bool, to_tuple=True), tuple[bool, ...])
assert_types(validate_arrayN([1, 2], dtype_out=int, to_tuple=True), tuple[int, ...])
assert_types(validate_arrayN([1, 2], dtype_out=float, to_tuple=True), tuple[float, ...])
assert_types(
    validate_arrayN([1, 2], dtype_out='float32', to_tuple=True),
    tuple[bool, ...] | tuple[int, ...] | tuple[float, ...],
)
assert_types(validate_arrayN([1, 2], to_list=flag()), _ArrayNOut)
assert_types(validate_arrayN(1), npt.NDArray[np.int64])
assert_types(validate_arrayN([[1, 2]]), npt.NDArray[np.int64])
assert_types(validate_arrayN(np.float32(1.5)), npt.NDArray[np.float32])
assert_types(validate_arrayN([1, 2], reshape=False), npt.NDArray[np.int64])
assert_types(validate_arrayN_unsigned([1, 2]), npt.NDArray[np.int64])
assert_types(validate_arrayN_unsigned([1, 2], dtype_out=np.uint8), npt.NDArray[np.uint8])
assert_types(validate_arrayN_unsigned([1, 2], dtype_out='int32'), npt.NDArray[_Integer])
assert_types(validate_arrayN_unsigned([1, 2], to_list=True), list[int])
assert_types(validate_arrayN_unsigned([1, 2], to_tuple=True), tuple[int, ...])
assert_types(validate_arrayN_unsigned([1, 2], to_list=flag()), _ArrayNUnsignedOut)
assert_types(validate_arrayN_unsigned(np.array([1, 2], dtype=np.uint16)), npt.NDArray[np.int64])
assert_types(validate_arrayN_unsigned([1.0, 2.0], reshape=False), npt.NDArray[np.int64])
