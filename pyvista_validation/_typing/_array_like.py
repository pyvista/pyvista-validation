"""Array-like type definitions."""

from __future__ import annotations

from collections.abc import Sequence
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeAlias
from typing import Union

import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    # Type variable defaults (PEP 696) reached the standard library in 3.13.
    from typing_extensions import TypeVar

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
# What ``must_be_real`` admits: to NumPy a boolean is not a number.
_Real = _Floating | _Integer
# Text, which NumPy holds as fixed-width unicode or bytes; accepted with ``must_be_real=False``.
_Text = np.str_ | np.bytes_
_AnyScalar = _Scalar | _Text

# Anything np.dtype() accepts for numeric data: a scalar type, a dtype, or a dtype name.
if TYPE_CHECKING:
    from typing import Never

    # Empty list literals, which NumPy turns into float64 arrays; only a type checker
    # can tell them apart from other lists, so the runtime value is a placeholder.
    _EmptyList: TypeAlias = (
        list[Never] | list[list[Never]] | list[list[list[Never]]] | list[list[list[list[Never]]]]
    )
    _DTypeLike: TypeAlias = (
        type[np.generic[object] | float | int | bool] | np.dtype[np.generic[object]] | str
    )
else:
    _DTypeLike = npt.DTypeLike
    _EmptyList = list

# Arrays of a known rank, for outputs whose rank the validation guarantees.
# Every scalar type an array can hold, abstract NumPy families included.
_AnyNumeric = Union['np.floating[Any]', 'np.integer[Any]', np.bool_]
_AnyDType = Union[_AnyNumeric, _Text]
_AnyScalarT = TypeVar('_AnyScalarT', bound=_AnyDType, default=_AnyScalar)
_Array0D = np.ndarray[tuple[()], np.dtype[_AnyScalarT]]
_Array1D = np.ndarray[tuple[int], np.dtype[_AnyScalarT]]
_Array2D = np.ndarray[tuple[int, int], np.dtype[_AnyScalarT]]
_Array3D = np.ndarray[tuple[int, int, int], np.dtype[_AnyScalarT]]
# Arrays of at least one dimension, so a known 0-D array is excluded.
_ArrayAtLeast1D = np.ndarray[tuple[int, *tuple[int, ...]], np.dtype[_AnyScalarT]]

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
_NestedStr = (
    Sequence[str]
    | Sequence[Sequence[str]]
    | Sequence[Sequence[Sequence[str]]]
    | Sequence[Sequence[Sequence[Sequence[str]]]]
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
_NestedListStr = list[str] | list[list[str]] | list[list[list[str]]] | list[list[list[list[str]]]]
_NestedTupleStr = (
    tuple[str, ...]
    | tuple[tuple[str, ...], ...]
    | tuple[tuple[tuple[str, ...], ...], ...]
    | tuple[tuple[tuple[tuple[str, ...], ...], ...], ...]
)
_FiniteNestedList = _NestedListFloat
_FiniteNestedTuple = _NestedTupleFloat

# What converting an array of each dtype family to lists or tuples produces; 0-D gives a scalar.
_ToListBool = bool | _NestedListBool
_ToListInt = int | _NestedListInt
_ToListFloat = float | _NestedListFloat
_ToList = _ToListBool | _ToListInt | _ToListFloat
_ToTupleBool = bool | _NestedTupleBool
_ToTupleInt = int | _NestedTupleInt
_ToTupleFloat = float | _NestedTupleFloat
_ToTuple = _ToTupleBool | _ToTupleInt | _ToTupleFloat
_ToListStr = str | _NestedListStr
_ToTupleStr = str | _NestedTupleStr
# The list and tuple results once text is admitted too.
_ToAnyList = _ToList | _ToListStr
_ToAnyTuple = _ToTuple | _ToTupleStr

# Array-likes of one kind, matching optype's ToFloat1D and friends; a float admits int and bool.
_PyT = TypeVar('_PyT', default=float)
_DTypeT = TypeVar('_DTypeT', bound=_AnyDType, default=_AnyDType)
# Each sequence holds Python values or NumPy values, never both, as NumPy's ArrayLike requires.
_VectorLikeOf = Union[Sequence[_PyT], Sequence[_DTypeT], _Array1D[_DTypeT]]
_MatrixLikeOf = Union[
    Sequence[Sequence[_PyT]],
    Sequence[Union[Sequence[_DTypeT], _Array1D[_DTypeT]]],
    _Array2D[_DTypeT],
]
# Sequences nested up to four deep, each level holding items or shallower sequences.
_PyNested1 = Sequence[_PyT]
_PyNested2 = Sequence[Union[_PyT, _PyNested1[_PyT]]]
_PyNested3 = Sequence[Union[_PyT, _PyNested1[_PyT], _PyNested2[_PyT]]]
_PyNested4 = Sequence[Union[_PyT, _PyNested1[_PyT], _PyNested2[_PyT], _PyNested3[_PyT]]]
_NpItem = Union[_DTypeT, npt.NDArray[_DTypeT]]
_NpNested1 = Sequence[_NpItem[_DTypeT]]
_NpNested2 = Sequence[Union[_NpItem[_DTypeT], _NpNested1[_DTypeT]]]
_NpNested3 = Sequence[Union[_NpItem[_DTypeT], _NpNested1[_DTypeT], _NpNested2[_DTypeT]]]
_NpNested4 = Sequence[
    Union[_NpItem[_DTypeT], _NpNested1[_DTypeT], _NpNested2[_DTypeT], _NpNested3[_DTypeT]]
]
_ArrayLikeOf = Union[_PyNested4[_PyT], _NpNested4[_DTypeT], _ArrayAtLeast1D[_DTypeT]]

_FloatDType = Union[np.floating, np.integer, np.bool_]
_IntDType = Union[np.integer, np.bool_]

VectorLikeFloat = _VectorLikeOf[float, _FloatDType]
VectorLikeInt = _VectorLikeOf[int, _IntDType]
VectorLikeBool = _VectorLikeOf[bool, np.bool_]
MatrixLikeFloat = _MatrixLikeOf[float, _FloatDType]
MatrixLikeInt = _MatrixLikeOf[int, _IntDType]
MatrixLikeBool = _MatrixLikeOf[bool, np.bool_]
ArrayLikeFloat = _ArrayLikeOf[float, _FloatDType]
ArrayLikeInt = _ArrayLikeOf[int, _IntDType]
ArrayLikeBool = _ArrayLikeOf[bool, np.bool_]

# The same shapes once text is admitted: any array the package handles, scalars included.
_AnyItem = Union[float, str, bytes, _AnyDType, npt.NDArray[_AnyDType]]
# Sequences nested up to four deep, each level holding items or shallower sequences.
_AnyNested1 = Sequence[_AnyItem]
_AnyNested2 = Sequence[Union[_AnyItem, _AnyNested1]]
_AnyNested3 = Sequence[Union[_AnyItem, _AnyNested1, _AnyNested2]]
_AnyNested = Sequence[Union[_AnyItem, _AnyNested1, _AnyNested2, _AnyNested3]]
_AnyArrayLike = Union[npt.NDArray[_AnyDType], _AnyNested]
_AnyArrayLikeOrScalar = Union[float, str, bytes, _AnyDType, _AnyArrayLike]
# The same without NumPy arrays: scalars and nested sequences.
_AnyNonArrayLikeOrScalar = Union[float, str, bytes, _AnyDType, _AnyNested]
