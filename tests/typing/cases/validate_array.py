"""Typing cases for validate_array."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

from pyvista_validation import validate_array
from pyvista_validation._typing import _Scalar
from pyvista_validation._typing import _ToList
from pyvista_validation._typing import _ToListBool
from pyvista_validation._typing import _ToListFloat
from pyvista_validation._typing import _ToListInt
from pyvista_validation._typing import _ToTuple
from pyvista_validation._typing import _ToTupleBool
from pyvista_validation._typing import _ToTupleFloat
from pyvista_validation._typing import _ToTupleInt

_ArrayOut = npt.NDArray[_Scalar] | _ToList | _ToTuple


def flag() -> bool:
    """Return a bool no type checker can narrow to a literal."""
    return False


assert_types(validate_array(np.array([1.5], dtype=np.float32)), npt.NDArray[np.float32])
assert_types(validate_array([]), npt.NDArray[np.float64])
assert_types(validate_array([True, False], must_be_real=False), npt.NDArray[np.bool_])
assert_types(validate_array([1, 2]), npt.NDArray[np.int64])
assert_types(validate_array([1.5, 2.5]), npt.NDArray[np.float64])
assert_types(validate_array([1, 2], dtype_out=np.float32), npt.NDArray[np.float32])
assert_types(validate_array([1, 2], dtype_out=bool), npt.NDArray[np.bool_])
assert_types(validate_array([1, 2], dtype_out=int), npt.NDArray[np.int64])
assert_types(validate_array([1, 2], dtype_out=float), npt.NDArray[np.float64])
assert_types(validate_array([1, 2], dtype_out='float32'), npt.NDArray[_Scalar])
assert_types(validate_array([], to_list=True), _ToListFloat)
assert_types(validate_array([True, False], must_be_real=False, to_list=True), _ToListBool)
assert_types(validate_array([1, 2], to_list=True), _ToListInt)
assert_types(validate_array([1.5, 2.5], to_list=True), _ToListFloat)
assert_types(validate_array([1, 2], dtype_out=bool, to_list=True), _ToListBool)
assert_types(validate_array([1, 2], dtype_out=int, to_list=True), _ToListInt)
assert_types(validate_array([1, 2], dtype_out=float, to_list=True), _ToListFloat)
assert_types(validate_array([1, 2], dtype_out='float32', to_list=True), _ToList)
assert_types(validate_array([], to_tuple=True), _ToTupleFloat)
assert_types(validate_array([True, False], must_be_real=False, to_tuple=True), _ToTupleBool)
assert_types(validate_array([1, 2], to_tuple=True), _ToTupleInt)
assert_types(validate_array([1.5, 2.5], to_tuple=True), _ToTupleFloat)
assert_types(validate_array([1, 2], dtype_out=bool, to_tuple=True), _ToTupleBool)
assert_types(validate_array([1, 2], dtype_out=int, to_tuple=True), _ToTupleInt)
assert_types(validate_array([1, 2], dtype_out=float, to_tuple=True), _ToTupleFloat)
assert_types(validate_array([1, 2], dtype_out='float32', to_tuple=True), _ToTuple)
assert_types(validate_array([1, 2], to_list=flag()), _ArrayOut)
