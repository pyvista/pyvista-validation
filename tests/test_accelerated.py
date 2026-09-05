"""The C fast paths present exactly like the Python functions they stand in for."""

from __future__ import annotations

import inspect

import pytest

import pyvista_validation
from pyvista_validation import _accelerate

ACCELERATED = sorted(
    name
    for name, function in _accelerate.reference.items()
    if getattr(pyvista_validation, name) is not function
)

pytestmark = pytest.mark.skipif(not _accelerate.enabled, reason='the C extension is not in use')


@pytest.mark.parametrize('name', ACCELERATED)
def test_builtin_presents_like_the_python_function(name):
    builtin = getattr(pyvista_validation, name)
    function = _accelerate.reference[name]
    assert inspect.isbuiltin(builtin)
    assert builtin.__name__ == function.__name__
    assert builtin.__module__ == function.__module__
    assert builtin.__doc__ == function.__doc__
    expected = [
        (p.name, p.kind, p.default) for p in inspect.signature(function).parameters.values()
    ]
    actual = [(p.name, p.kind, p.default) for p in inspect.signature(builtin).parameters.values()]
    assert actual == expected


def test_every_public_function_is_registered():
    public = {name for name in dir(pyvista_validation) if name.startswith(('check_', 'validate_'))}
    assert set(_accelerate.reference) == public
