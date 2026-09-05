"""Typing cases for the array-like aliases and their scalar type parameter."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

from pyvista_validation._typing import ArrayLike
from pyvista_validation._typing import MatrixLike
from pyvista_validation._typing import NumberType
from pyvista_validation._typing import NumpyArray
from pyvista_validation._typing import VectorLike
from pyvista_validation._typing import _ArrayLikeOrScalar
from pyvista_validation._typing import _Scalar


def vector(value: VectorLike) -> VectorLike:
    """Pass a vector-like through under the bare alias."""
    return value


def vector_of_floats(value: VectorLike[float]) -> VectorLike[float]:
    """Pass a vector-like through under the alias subscripted with its default."""
    return value


def vector_of_ints(value: VectorLike[int]) -> VectorLike[int]:
    """Pass a vector-like of ints through."""
    return value


def matrix_of_bools(value: MatrixLike[bool]) -> MatrixLike[bool]:
    """Pass a matrix-like of bools through."""
    return value


def array_of_ints(value: ArrayLike[int]) -> ArrayLike[int]:
    """Pass an array-like of ints through."""
    return value


def scalar_or_array(value: _ArrayLikeOrScalar[int]) -> _ArrayLikeOrScalar[int]:
    """Pass a scalar or array-like of ints through."""
    return value


def same(value: ArrayLike[NumberType]) -> ArrayLike[NumberType]:
    """Pass an array-like through, keeping its scalar type parameter."""
    return value


def any_array(value: NumpyArray) -> NumpyArray:
    """Pass an array of any numeric dtype through."""
    return value


# A bare alias keeps its type variable at runtime, so the cases compare against the
# subscripted spelling; the functions annotated bare prove the default statically.
assert_types(vector([1.0]), VectorLike[float])
assert_types(vector_of_floats([1.0]), VectorLike[float])
assert_types(vector_of_ints([1]), VectorLike[int])
assert_types(matrix_of_bools([[True]]), MatrixLike[bool])
assert_types(array_of_ints([1]), ArrayLike[int])
assert_types(scalar_or_array(1), _ArrayLikeOrScalar[int])
assert_types(same([1]), ArrayLike[int])
assert_types(same([1.5]), ArrayLike[float])
assert_types(same(np.zeros(2)), ArrayLike[float])
assert_types(any_array(np.zeros(2, dtype=np.int8)), NumpyArray[_Scalar])
