"""Pytest plugin that points PyVista's validation imports at this package.

Substituting the modules lets PyVista's own validation test suite exercise
this package's implementation, which is the real downstream contract.
"""

from __future__ import annotations

import sys

from pyvista.core._validation.validate import _validate_color_sequence

import pyvista_validation
import pyvista_validation._cast_array
import pyvista_validation.check
import pyvista_validation.validate

# This helper stayed with PyVista, but its tests import it from the validation
# package, so put it back on the module that replaces it.
pyvista_validation.validate._validate_color_sequence = _validate_color_sequence

_REPLACEMENTS = {
    'pyvista.core._validation': pyvista_validation,
    'pyvista.core._validation._cast_array': pyvista_validation._cast_array,
    'pyvista.core._validation.check': pyvista_validation.check,
    'pyvista.core._validation.validate': pyvista_validation.validate,
}
sys.modules.update(_REPLACEMENTS)
