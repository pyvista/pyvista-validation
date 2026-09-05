"""Typing cases for the check functions, which return their input typed as what they checked."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
import numbers

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

from pyvista_validation import check_contains
from pyvista_validation import check_finite
from pyvista_validation import check_greater_than
from pyvista_validation import check_instance
from pyvista_validation import check_integer
from pyvista_validation import check_iterable
from pyvista_validation import check_iterable_items
from pyvista_validation import check_length
from pyvista_validation import check_less_than
from pyvista_validation import check_ndim
from pyvista_validation import check_nonnegative
from pyvista_validation import check_number
from pyvista_validation import check_range
from pyvista_validation import check_real
from pyvista_validation import check_sequence
from pyvista_validation import check_shape
from pyvista_validation import check_sorted
from pyvista_validation import check_string
from pyvista_validation import check_subdtype
from pyvista_validation import check_type
from pyvista_validation._typing import _Scalar
from pyvista_validation.check import _dtype_of
from pyvista_validation.check import _issubdtype
from pyvista_validation.check import _Shape
from pyvista_validation.check import _shape_of
from pyvista_validation.check import _union_members
from pyvista_validation.check import _validate_real_value
from pyvista_validation.check import _validate_shape_value

# Declared rather than inferred, so a case reads the array type it names.
ONES: npt.NDArray[np.float64] = np.ones(2)
INTS: npt.NDArray[np.int64] = np.ones(2, dtype=np.int64)
MATRIX: npt.NDArray[np.int64] = np.array([[0, 1], [2, 3]])
SCALAR: npt.NDArray[np.float64] = np.array(1.0)
TEXT: npt.NDArray[np.str_] = np.array(['a', 'b'])


def unknown() -> object:
    """Return a list typed as nothing more than an object, for the checks that narrow it."""
    return [1, 2]


def unknown_str() -> object:
    """Return a string typed as an object."""
    return 'a'


def unknown_number() -> object:
    """Return a number typed as an object."""
    return 1


# The input comes back with its own type.
assert_types(check_subdtype(ONES, np.floating), npt.NDArray[np.float64])
assert_types(check_subdtype(float, (np.floating, np.integer)), type[float])
assert_types(check_subdtype('f8', [np.floating]), str)
assert_types(check_subdtype([1, 2], np.integer), list[int])
assert_types(check_real([1.0]), list[float])
assert_types(check_real(ONES), npt.NDArray[np.float64])
assert_types(check_real(1), int)
assert_types(check_real(TEXT), npt.NDArray[np.str_])
assert_types(check_sorted([1, 2]), list[int])
assert_types(check_sorted(MATRIX, axis=None), npt.NDArray[np.int64])
assert_types(check_sorted(MATRIX, axis=0), npt.NDArray[np.int64])
assert_types(check_sorted([2, 1], ascending=False, strict=True), list[int])
assert_types(check_sorted(TEXT), npt.NDArray[np.str_])
assert_types(check_sorted(['a', 'b']), list[str])
assert_types(check_finite(1.0), float)
assert_types(check_finite(ONES), npt.NDArray[np.float64])
assert_types(check_integer([1.0, 2.0]), list[float])
assert_types(check_integer(INTS, strict=True), npt.NDArray[np.int64])
assert_types(check_nonnegative([0, 1]), list[int])
assert_types(check_greater_than([1, 2], 0), list[int])
assert_types(check_greater_than([1, 2], np.float32(0.5)), list[int])
assert_types(check_greater_than([1, 2], 1, strict=False), list[int])
assert_types(check_less_than([1, 2], 3), list[int])
assert_types(check_less_than([1, 2], np.int8(2), strict=False), list[int])
assert_types(check_range([1], [0, 2]), list[int])
assert_types(check_range(ONES, np.array([0.0, 2.0]), strict_lower=True), npt.NDArray[np.float64])
assert_types(check_shape([1, 2], 2), list[int])
assert_types(check_shape([1, 2], [(2,), (3,)]), list[int])
assert_types(check_shape(1, ()), int)
assert_types(check_shape(TEXT, 2), npt.NDArray[np.str_])
assert_types(check_ndim([1], 1), list[int])
assert_types(check_ndim([[1]], [1, 2]), list[list[int]])
assert_types(check_ndim(MATRIX, range(3)), npt.NDArray[np.int64])
assert_types(check_ndim(['a'], 1), list[str])
assert_types(check_number(1), int)
assert_types(check_number(1.5), float)
assert_types(check_number(np.int32(1)), np.int32)
assert_types(check_string('a'), str)
assert_types(check_sequence([1]), list[int])
assert_types(check_iterable((1,)), tuple[int])
assert_types(check_instance(1, int), int)
assert_types(check_instance(1, (int, float)), int | float)
assert_types(check_instance(1, int | float), int)
assert_types(check_type(1, int), int)
assert_types(check_type(1, (int, float)), int | float)
assert_types(check_type(1, int | float), int)
assert_types(check_iterable_items([1, 2], int), list[int])
assert_types(check_iterable_items((1, 2.0), (int, float), allow_subclass=False), tuple[int, float])
assert_types(check_contains([1, 2], must_contain=1), int)
assert_types(check_contains('abc', must_contain='b'), str)
assert_types(check_length([1, 2], exact_length=2), list[int])
assert_types(check_length([1, 2], exact_length=[1, 2], min_length=1, max_length=2), list[int])
assert_types(check_length((1, 2), must_be_1d=True), tuple[int, int])
assert_types(check_length(1.0, allow_scalar=True), float)
assert_types(check_length(SCALAR, allow_scalar=True), npt.NDArray[np.float64])

# An input of unknown type comes back narrowed to what the check established.
assert_types(check_number(unknown_number()), numbers.Number)
assert_types(check_string(unknown_str()), str)
assert_types(check_sequence(unknown()), Sequence[object])
assert_types(check_iterable(unknown()), Iterable[object])
assert_types(check_instance(unknown_str(), str), str)
assert_types(check_instance(unknown_number(), (int, float)), int | float)
assert_types(check_instance(unknown_number(), (int, float, str)), int | float | str)
assert_types(check_instance(unknown(), int | list), object)
assert_types(check_type(unknown_number(), int), int)
assert_types(check_iterable_items(unknown(), int), Iterable[int])
assert_types(check_iterable_items(unknown(), (int, float)), Iterable[object])

# Helpers.
assert_types(_validate_real_value(1.0), npt.NDArray[_Scalar])
assert_types(_validate_real_value(np.int32(1)), npt.NDArray[_Scalar])
assert_types(_validate_shape_value((1, 2)), _Shape)
assert_types(_validate_shape_value(3), _Shape)
assert_types(_validate_shape_value(()), _Shape)
# np.generic cannot be subscripted at runtime, so the type is quoted: the checker holds the
# case to it and the runtime half is skipped.
assert_types(_dtype_of(np.zeros(2)), 'np.dtype[np.generic[object]]')
assert_types(_dtype_of('f8'), 'np.dtype[np.generic[object]]')
assert_types(_dtype_of(TEXT), 'np.dtype[np.generic[object]]')
assert_types(_shape_of([1, 2]), tuple[int, ...])
assert_types(_shape_of(['a', 'b']), tuple[int, ...])
assert_types(_issubdtype(np.dtype('f8'), np.floating), bool)
assert_types(_union_members(int | float), tuple[type[object], ...])

SKIP_RUNTIME = {
    'check_real(TEXT)': 'raises TypeError: text is not real, only the passthrough is typed',
}
