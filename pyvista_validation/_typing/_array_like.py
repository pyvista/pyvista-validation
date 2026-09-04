"""Array-like type definitions.

The aliases are concrete rather than generic so that code annotated with them
stays fully typed under mypy's coverage report, which scores any expression
involving a type variable as imprecise. Sequence elements are Python scalars;
``np.float64`` is accepted through its ``float`` subclass, and ``int`` and
``bool`` through numeric promotion.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt

# Every NumPy scalar type this package produces or preserves.
_Scalar = (
    np.float64
    | np.float32
    | np.float16
    | np.int64
    | np.int32
    | np.int16
    | np.int8
    | np.uint64
    | np.uint32
    | np.uint16
    | np.uint8
    | np.bool_
)
_Floating = np.float64 | np.float32 | np.float16
_Integer = np.int64 | np.int32 | np.int16 | np.int8 | np.uint64 | np.uint32 | np.uint16 | np.uint8

# For overload signatures that return the same dtype they are given.
_ScalarT = TypeVar('_ScalarT', bound=_Scalar)
NumpyArray = npt.NDArray[_ScalarT]

Number = float
NumberType = float

_NestedBool = (
    Sequence[bool]
    | Sequence[Sequence[bool]]
    | Sequence[Sequence[Sequence[bool]]]
    | Sequence[Sequence[Sequence[Sequence[bool]]]]
)
_NestedInt = (
    Sequence[int]
    | Sequence[Sequence[int]]
    | Sequence[Sequence[Sequence[int]]]
    | Sequence[Sequence[Sequence[Sequence[int]]]]
)
_NestedFloat = (
    Sequence[float]
    | Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
    | Sequence[Sequence[Sequence[Sequence[float]]]]
)

# What ndarray.tolist() returns for each dtype family, up to four dimensions.
_NestedListBool = (
    list[bool] | list[list[bool]] | list[list[list[bool]]] | list[list[list[list[bool]]]]
)
_NestedListInt = list[int] | list[list[int]] | list[list[list[int]]] | list[list[list[list[int]]]]
_NestedListFloat = (
    list[float] | list[list[float]] | list[list[list[float]]] | list[list[list[list[float]]]]
)
_NestedTupleBool = (
    tuple[bool, ...]
    | tuple[tuple[bool, ...], ...]
    | tuple[tuple[tuple[bool, ...], ...], ...]
    | tuple[tuple[tuple[tuple[bool, ...], ...], ...], ...]
)
_NestedTupleInt = (
    tuple[int, ...]
    | tuple[tuple[int, ...], ...]
    | tuple[tuple[tuple[int, ...], ...], ...]
    | tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]
)
_NestedTupleFloat = (
    tuple[float, ...]
    | tuple[tuple[float, ...], ...]
    | tuple[tuple[tuple[float, ...], ...], ...]
    | tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]
)
_FiniteNestedList = _NestedListFloat
_FiniteNestedTuple = _NestedTupleFloat

_ArrayLike1D = npt.NDArray[_Scalar] | Sequence[float] | Sequence[npt.NDArray[_Scalar]]
_ArrayLike2D = (
    npt.NDArray[_Scalar] | Sequence[Sequence[float]] | Sequence[Sequence[npt.NDArray[_Scalar]]]
)
_ArrayLike3D = (
    npt.NDArray[_Scalar]
    | Sequence[Sequence[Sequence[float]]]
    | Sequence[Sequence[Sequence[npt.NDArray[_Scalar]]]]
)
_ArrayLike4D = (
    npt.NDArray[_Scalar]
    | Sequence[Sequence[Sequence[Sequence[float]]]]
    | Sequence[Sequence[Sequence[Sequence[npt.NDArray[_Scalar]]]]]
)
_ArrayLike = _ArrayLike1D | _ArrayLike2D | _ArrayLike3D | _ArrayLike4D
