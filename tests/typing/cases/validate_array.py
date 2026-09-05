"""Typing cases for validate_array."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

if TYPE_CHECKING:
    from collections.abc import Sequence

from pyvista_validation import validate_array
from pyvista_validation._typing import _AnyScalar
from pyvista_validation._typing import _Floating
from pyvista_validation._typing import _Scalar
from pyvista_validation._typing import _ToAnyList
from pyvista_validation._typing import _ToAnyTuple
from pyvista_validation._typing import _ToList
from pyvista_validation._typing import _ToListBool
from pyvista_validation._typing import _ToListFloat
from pyvista_validation._typing import _ToListInt
from pyvista_validation._typing import _ToListStr
from pyvista_validation._typing import _ToTuple
from pyvista_validation._typing import _ToTupleBool
from pyvista_validation._typing import _ToTupleFloat
from pyvista_validation._typing import _ToTupleInt
from pyvista_validation._typing import _ToTupleStr

_ArrayOut = npt.NDArray[_Scalar] | _ToList | _ToTuple
_AnyArrayOut = npt.NDArray[_AnyScalar] | _ToAnyList | _ToAnyTuple


def flag() -> bool:
    """Return a bool no type checker can narrow to a literal."""
    return False


assert_types(validate_array(np.array([1.5], dtype=np.float32)), npt.NDArray[np.float32])
assert_types(validate_array([]), npt.NDArray[np.float64])
assert_types(validate_array(np.ones(2) > 0, must_be_real=False), npt.NDArray[np.bool_])
assert_types(validate_array([True, False], must_be_real=False), npt.NDArray[np.bool_])
assert_types(validate_array([1, 2]), npt.NDArray[np.int64])
assert_types(validate_array([1.5, 2.5]), npt.NDArray[np.float64])
assert_types(validate_array(['a', 'b'], must_be_real=False), npt.NDArray[np.str_])
assert_types(validate_array([1, 2], dtype_out=np.float32), npt.NDArray[np.float32])
assert_types(validate_array([1, 2], dtype_out=bool), npt.NDArray[np.bool_])
assert_types(validate_array([1, 2], dtype_out=int), npt.NDArray[np.int64])
assert_types(validate_array([1, 2], dtype_out=float), npt.NDArray[np.float64])
assert_types(validate_array([1, 2], dtype_out='float32'), npt.NDArray[_Scalar])
assert_types(
    validate_array(['a', 'b'], must_be_real=False, dtype_out='U1'),
    npt.NDArray[_AnyScalar],
)
assert_types(validate_array([], to_list=True), _ToListFloat)
assert_types(validate_array([True, False], must_be_real=False, to_list=True), _ToListBool)
assert_types(validate_array([1, 2], to_list=True), _ToListInt)
assert_types(validate_array([1.5, 2.5], to_list=True), _ToListFloat)
assert_types(validate_array(['a', 'b'], must_be_real=False, to_list=True), _ToListStr)
assert_types(validate_array([1, 2], dtype_out=bool, to_list=True), _ToListBool)
assert_types(validate_array([1, 2], dtype_out=int, to_list=True), _ToListInt)
assert_types(validate_array([1, 2], dtype_out=float, to_list=True), _ToListFloat)
assert_types(validate_array([1, 2], dtype_out='float32', to_list=True), _ToList)
assert_types(
    validate_array(['a', 'b'], must_be_real=False, dtype_out='U1', to_list=True),
    _ToAnyList,
)
assert_types(validate_array([], to_tuple=True), _ToTupleFloat)
assert_types(validate_array([True, False], must_be_real=False, to_tuple=True), _ToTupleBool)
assert_types(validate_array([1, 2], to_tuple=True), _ToTupleInt)
assert_types(validate_array([1.5, 2.5], to_tuple=True), _ToTupleFloat)
assert_types(validate_array(['a', 'b'], must_be_real=False, to_tuple=True), _ToTupleStr)
assert_types(validate_array([1, 2], dtype_out=bool, to_tuple=True), _ToTupleBool)
assert_types(validate_array([1, 2], dtype_out=int, to_tuple=True), _ToTupleInt)
assert_types(validate_array([1, 2], dtype_out=float, to_tuple=True), _ToTupleFloat)
assert_types(validate_array([1, 2], dtype_out='float32', to_tuple=True), _ToTuple)
assert_types(
    validate_array(['a', 'b'], must_be_real=False, dtype_out='U1', to_tuple=True),
    _ToAnyTuple,
)
assert_types(validate_array([1, 2], to_list=flag()), _ArrayOut)
assert_types(validate_array(['a', 'b'], must_be_real=False, to_list=flag()), _AnyArrayOut)


# An argument whose type spans an array and sequences of the same scalar keeps that scalar.
def float32_matrix() -> npt.NDArray[np.float32] | Sequence[Sequence[np.float32]]:
    """Return a value typed as either an array or a nested sequence."""
    return np.ones((2, 2), dtype=np.float32)


def floating_matrix() -> npt.NDArray[_Floating] | Sequence[Sequence[float]]:
    """Return a value typed as either a floating array or a nested sequence of floats."""
    return np.ones((2, 2))


assert_types(validate_array(float32_matrix()), npt.NDArray[np.float32])
assert_types(validate_array(floating_matrix(), to_list=True), _ToListFloat)
