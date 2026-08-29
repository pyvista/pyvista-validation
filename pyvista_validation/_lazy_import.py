"""Lazily imported names from this package's optional dependencies.

VTK and SciPy are both optional and both slow to import, so their names are
resolved from their own submodules on first access and then cached in this
module's globals. Importing from ``vtkmodules`` rather than ``vtk`` keeps the
cost to the one submodule a name actually lives in.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation
    from vtkmodules.vtkCommonMath import vtkMatrix3x3
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkCommonTransforms import vtkTransform

__all__ = ['Rotation', 'vtkMatrix3x3', 'vtkMatrix4x4', 'vtkTransform']

_MODULES = {
    'Rotation': 'scipy.spatial.transform',
    'vtkMatrix3x3': 'vtkmodules.vtkCommonMath',
    'vtkMatrix4x4': 'vtkmodules.vtkCommonMath',
    'vtkTransform': 'vtkmodules.vtkCommonTransforms',
}


def __getattr__(name: str) -> Any:
    """Import a name from its optional dependency, caching it for next time."""
    if name not in _MODULES:
        msg = f'module {__name__!r} has no attribute {name!r}'
        raise AttributeError(msg)

    try:
        value: Any = getattr(importlib.import_module(_MODULES[name]), name)
    except ModuleNotFoundError:
        # No object of this type can exist without its package installed. A
        # placeholder class is never an instance and still composes into the
        # `Union` type aliases, unlike None or an empty tuple.
        value = type(name, (), {})

    globals()[name] = value  # __getattr__ only runs on a miss, so this caches
    return value
