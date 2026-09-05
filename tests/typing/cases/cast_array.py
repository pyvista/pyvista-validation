"""Typing cases for the array casting helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

from pyvista_validation._cast_array import _asarray
from pyvista_validation._cast_array import _cast_to_list
from pyvista_validation._cast_array import _cast_to_numpy
from pyvista_validation._cast_array import _cast_to_tuple
from pyvista_validation._cast_array import _to_tuple
from pyvista_validation._cast_array import _tolist
from pyvista_validation._typing import _AnyArrayLikeOrScalar
from pyvista_validation._typing import _AnyScalar
from pyvista_validation._typing import _ArrayLikeOrScalar
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


def array_like() -> _ArrayLikeOrScalar:
    """Return an array typed as broadly as the parameter that accepts it."""
    return [1, 2]


def any_array_like() -> _AnyArrayLikeOrScalar:
    """Return an array typed as broadly as the parameters that also accept text."""
    return ['a', 'b']


TEXT: npt.NDArray[np.str_] = np.array(['a', 'b'])


# Bound to a name first: passed inline, mypy would infer the constructor call against the
# first overload's parameter type and lose the dtype.
int8_array = np.ones(2, dtype=np.int8)

assert_types(_cast_to_numpy(np.array([1.5], dtype=np.float32)), npt.NDArray[np.float32])
assert_types(_cast_to_numpy(np.float32(1.5)), npt.NDArray[np.float32])
assert_types(_cast_to_numpy([]), npt.NDArray[np.float64])
assert_types(_cast_to_numpy([[]]), npt.NDArray[np.float64])
assert_types(_cast_to_numpy(True), npt.NDArray[np.bool_])
assert_types(_cast_to_numpy([True, False]), npt.NDArray[np.bool_])
assert_types(_cast_to_numpy(1), npt.NDArray[np.int64])
assert_types(_cast_to_numpy([[1, 2], [3, 4]]), npt.NDArray[np.int64])
assert_types(_cast_to_numpy(1.5), npt.NDArray[np.float64])
assert_types(_cast_to_numpy((1.5, 2.5)), npt.NDArray[np.float64])
assert_types(_cast_to_numpy([1, 2.5]), npt.NDArray[np.float64])
assert_types(_cast_to_numpy([1, 2], dtype=np.float32), npt.NDArray[np.float32])
assert_types(_cast_to_numpy([1, 2], dtype=bool), npt.NDArray[np.bool_])
assert_types(_cast_to_numpy([1.5, 2.5], dtype=int), npt.NDArray[np.int64])
assert_types(_cast_to_numpy([1, 2], dtype=float), npt.NDArray[np.float64])
assert_types(_cast_to_numpy([1, 2], dtype='f4'), npt.NDArray[_Scalar])
assert_types(_cast_to_numpy(array_like()), npt.NDArray[_Scalar])
assert_types(
    _cast_to_numpy(np.array([1, 2], dtype=np.int64), as_any=False, copy=True, must_be_real=True),
    npt.NDArray[np.int64],
)
assert_types(_cast_to_list([]), _ToListFloat)
assert_types(_cast_to_list([True]), _ToListBool)
assert_types(_cast_to_list(int8_array), _ToListInt)
assert_types(_cast_to_list([[1.5]]), _ToListFloat)
assert_types(_cast_to_list(np.float16(1.5)), _ToListFloat)
assert_types(_cast_to_list(array_like()), _ToList)
assert_types(_cast_to_tuple([]), _ToTupleFloat)
assert_types(_cast_to_tuple([True]), _ToTupleBool)
assert_types(_cast_to_tuple(int8_array), _ToTupleInt)
assert_types(_cast_to_tuple([[1.5]]), _ToTupleFloat)
assert_types(_cast_to_tuple(1), _ToTupleInt)
assert_types(_cast_to_tuple(array_like()), _ToTuple)
assert_types(_asarray([1, 2], dtype=None, as_any=True), npt.NDArray[_AnyScalar])
assert_types(_tolist(np.zeros(2)), _ToAnyList)
assert_types(_to_tuple([1, [2, 3]]), object)

# Text is admitted alongside numbers; NumPy sorts and reshapes it, so the same helpers apply.
assert_types(_cast_to_numpy('abc'), npt.NDArray[np.str_])
assert_types(_cast_to_numpy(['a', 'b']), npt.NDArray[np.str_])
assert_types(_cast_to_numpy(TEXT), npt.NDArray[np.str_])
assert_types(_cast_to_numpy(any_array_like()), npt.NDArray[_AnyScalar])
assert_types(_cast_to_list('abc'), _ToListStr)
assert_types(_cast_to_list([['a'], ['b']]), _ToListStr)
assert_types(_cast_to_list(any_array_like()), _ToAnyList)
assert_types(_cast_to_tuple(TEXT), _ToTupleStr)
assert_types(_cast_to_tuple(any_array_like()), _ToAnyTuple)
