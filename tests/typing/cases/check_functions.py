"""Typing cases for the check functions."""

from __future__ import annotations

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

assert_types(check_subdtype(np.zeros(2), np.floating), None)
assert_types(check_subdtype(float, (np.floating, np.integer)), None)
assert_types(check_subdtype('f8', [np.floating]), None)
assert_types(check_subdtype([1, 2], np.integer), None)
assert_types(check_real([1.0]), None)
assert_types(check_real(np.zeros(2)), None)
assert_types(check_real(1), None)
assert_types(check_sorted([1, 2]), None)
assert_types(check_sorted(np.array([[0, 1], [2, 3]]), axis=None), None)
assert_types(check_sorted(np.array([[0, 1], [2, 3]]), axis=0), None)
assert_types(check_sorted([2, 1], ascending=False, strict=True), None)
assert_types(check_finite(1.0), None)
assert_types(check_finite(np.array([1.0, 2.0])), None)
assert_types(check_integer([1.0, 2.0]), None)
assert_types(check_integer(np.array([1]), strict=True), None)
assert_types(check_nonnegative([0, 1]), None)
assert_types(check_greater_than([1, 2], 0), None)
assert_types(check_greater_than([1, 2], np.float32(0.5)), None)
assert_types(check_greater_than([1, 2], 1, strict=False), None)
assert_types(check_less_than([1, 2], 3), None)
assert_types(check_less_than([1, 2], np.int8(2), strict=False), None)
assert_types(check_range([1], [0, 2]), None)
assert_types(check_range(np.array([1.0]), np.array([0.0, 2.0]), strict_lower=True), None)
assert_types(check_shape([1, 2], 2), None)
assert_types(check_shape([1, 2], [(2,), (3,)]), None)
assert_types(check_shape(1, ()), None)
assert_types(check_ndim([1], 1), None)
assert_types(check_ndim([[1]], [1, 2]), None)
assert_types(check_ndim(np.zeros((2, 2)), range(3)), None)
assert_types(check_number(1), None)
assert_types(check_number(1.5), None)
assert_types(check_number(np.int32(1)), None)
assert_types(check_string('a'), None)
assert_types(check_sequence([1]), None)
assert_types(check_iterable((1,)), None)
assert_types(check_instance(1, int), None)
assert_types(check_instance(1, (int, float)), None)
assert_types(check_instance(1, int | float), None)
assert_types(check_type(1, int), None)
assert_types(check_type(1, (int, float)), None)
assert_types(check_type(1, int | float), None)
assert_types(check_iterable_items([1, 2], int), None)
assert_types(check_iterable_items((1, 2.0), (int, float), allow_subclass=False), None)
assert_types(check_contains([1, 2], must_contain=1), None)
assert_types(check_contains('abc', must_contain='b'), None)
assert_types(check_length([1, 2], exact_length=2), None)
assert_types(check_length([1, 2], exact_length=[1, 2], min_length=1, max_length=2), None)
assert_types(check_length((1, 2), must_be_1d=True), None)
assert_types(check_length(1.0, allow_scalar=True), None)
assert_types(check_length(np.array(1.0), allow_scalar=True), None)
assert_types(_validate_real_value(1.0), npt.NDArray[_Scalar])
assert_types(_validate_real_value(np.int32(1)), npt.NDArray[_Scalar])
assert_types(_validate_shape_value((1, 2)), _Shape)
assert_types(_validate_shape_value(3), _Shape)
assert_types(_validate_shape_value(()), _Shape)
# np.generic cannot be subscripted at runtime, so the type is quoted: the checker holds the
# case to it and the runtime half is skipped.
assert_types(_dtype_of(np.zeros(2)), 'np.dtype[np.generic[object]]')
assert_types(_dtype_of('f8'), 'np.dtype[np.generic[object]]')
assert_types(_shape_of([1, 2]), tuple[int, ...])
assert_types(_issubdtype(np.dtype('f8'), np.floating), bool)
assert_types(_union_members(int | float), tuple[type[object], ...])
