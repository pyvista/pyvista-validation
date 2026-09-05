"""Typing cases for validate_number."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

from pyvista_validation import validate_number
from pyvista_validation._typing import _Scalar

_NumberOut = bool | int | float | npt.NDArray[_Scalar]


def flag() -> bool:
    """Return a bool no type checker can narrow to a literal."""
    return False


assert_types(validate_number(True, must_be_real=False), bool)
assert_types(validate_number(1), int)
assert_types(validate_number(1.5), float)
assert_types(validate_number(1, dtype_out=bool), bool)
assert_types(validate_number(1, dtype_out=int), int)
assert_types(validate_number(1, dtype_out=float), float)
assert_types(validate_number(1, dtype_out='float32'), bool | int | float)
assert_types(validate_number(np.float32(1.5), to_list=False), npt.NDArray[np.float32])
assert_types(
    validate_number(np.bool_(bool(1)), must_be_real=False, to_list=False),
    npt.NDArray[np.bool_],
)
assert_types(validate_number(True, must_be_real=False, to_list=False), npt.NDArray[np.bool_])
assert_types(validate_number(1, to_list=False), npt.NDArray[np.int64])
assert_types(validate_number(1.5, to_list=False), npt.NDArray[np.float64])
assert_types(validate_number(1, dtype_out=np.float32, to_list=False), npt.NDArray[np.float32])
assert_types(validate_number(1, dtype_out=bool, to_list=False), npt.NDArray[np.bool_])
assert_types(validate_number(1, dtype_out=int, to_list=False), npt.NDArray[np.int64])
assert_types(validate_number(1, dtype_out=float, to_list=False), npt.NDArray[np.float64])
assert_types(validate_number(1, dtype_out='float32', to_list=False), npt.NDArray[_Scalar])
assert_types(validate_number(1, to_list=flag()), _NumberOut)
assert_types(validate_number(np.int32(1)), int)
assert_types(validate_number(np.float32(1.5)), float)
assert_types(validate_number(np.True_, must_be_real=False), bool)
assert_types(validate_number(1, to_tuple=True), int)
assert_types(validate_number(1.5, reshape=False), float)
