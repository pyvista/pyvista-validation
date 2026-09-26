"""Type aliases for annotating the inputs and outputs of the validation functions."""

from __future__ import annotations

from ._typing import ArrayLike
from ._typing import MatrixLike
from ._typing import RotationLike
from ._typing import TransformLike
from ._typing import VectorLike
from ._typing import _Array0D as Array0D
from ._typing import _Array1D as Array1D
from ._typing import _Array2D as Array2D
from ._typing import _Array3D as Array3D
from ._typing import _Floating as Floating
from ._typing import _Integer as Integer
from ._typing import _Real as Real
from ._typing import _Scalar as Scalar

__all__ = [
    'Array0D',
    'Array1D',
    'Array2D',
    'Array3D',
    'ArrayLike',
    'Floating',
    'Integer',
    'MatrixLike',
    'Real',
    'RotationLike',
    'Scalar',
    'TransformLike',
    'VectorLike',
]
