"""Typing cases for the lazy import module."""

from __future__ import annotations

from type_assert import assert_types

from pyvista_validation import _lazy_import
from pyvista_validation._lazy_import import _import_scipy
from pyvista_validation._lazy_import import _import_vtk
from pyvista_validation._lazy_import import _placeholder
from pyvista_validation._lazy_import import _vtk_root

assert_types(_lazy_import.__getattr__('vtkMatrix3x3'), type[object])
assert_types(_placeholder('Missing'), type[object])
assert_types(_vtk_root(), str)
assert_types(_import_vtk('vtkTransform'), type[object])
assert_types(_import_scipy('Rotation'), type[object])
