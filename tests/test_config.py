"""The switches in ``pyvista_validation.config`` that turn the runtime checks off."""

from __future__ import annotations

import os
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest

import pyvista_validation as pvv
from pyvista_validation import Config
from pyvista_validation import _accelerate
from pyvista_validation import _config
from pyvista_validation import config

# One call per switch that its check rejects.
INVALID = {
    'axes': lambda: pvv.validate_axes([[1, 0, 0], [1, 1, 0], [0, 0, 1]]),
    'contains': lambda: pvv.check_contains(['a'], must_contain='b'),
    'finite': lambda: pvv.check_finite([np.nan]),
    'greater_than': lambda: pvv.check_greater_than([0], 1),
    'instance': lambda: pvv.check_instance(1, str),
    'integer': lambda: pvv.check_integer([0.5]),
    'iterable': lambda: pvv.check_iterable(1),
    'iterable_items': lambda: pvv.check_iterable_items([1, 'a'], int),
    'length': lambda: pvv.check_length([1, 2], exact_length=3),
    'less_than': lambda: pvv.check_less_than([2], 1),
    'ndim': lambda: pvv.check_ndim([1, 2], 2),
    'nonnegative': lambda: pvv.check_nonnegative([-1]),
    'number': lambda: pvv.check_number('a'),
    'range': lambda: pvv.check_range([5], [0, 1]),
    'real': lambda: pvv.check_real(['a']),
    'rotation': lambda: pvv.validate_rotation(np.diag([2, 1, 1])),
    'sequence': lambda: pvv.check_sequence(1),
    'shape': lambda: pvv.check_shape([1, 2], 3),
    'sorted': lambda: pvv.check_sorted([3, 1]),
    'string': lambda: pvv.check_string(1),
    'subdtype': lambda: pvv.check_subdtype(np.array([1.0]), np.integer),
    'type': lambda: pvv.check_type(True, int),
}

# One validate_array option per switch that it uses.
INVALID_OPTIONS = {
    'finite': ([np.inf], {'must_be_finite': True}),
    'integer': ([0.5], {'must_be_integer': True}),
    'length': ([1, 2], {'must_have_length': 3}),
    'ndim': ([1, 2], {'must_have_ndim': 2}),
    'nonnegative': ([-1], {'must_be_nonnegative': True}),
    'range': ([5], {'must_be_in_range': [0, 1]}),
    'real': ([True], {'must_be_real': True}),
    'shape': ([1, 2], {'must_have_shape': 3}),
    'sorted': ([3, 1], {'must_be_sorted': True}),
    'subdtype': ([1.0], {'must_have_dtype': np.integer}),
}


def test_every_switch_has_an_invalid_call():
    assert sorted(INVALID) == list(_config.CHECKS)


@pytest.mark.skipif(not _accelerate.enabled, reason='the C extension is not in use')
def test_c_extension_orders_the_checks_like_python():
    assert _accelerate._CHECKS == _config.CHECKS


@pytest.mark.parametrize('name', _config.CHECKS)
def test_switch_skips_only_its_own_check(name):
    call = INVALID[name]
    with pytest.raises((TypeError, ValueError)):
        call()
    with config.override(**{name: False}):
        call()
    others = dict.fromkeys(set(_config.CHECKS) - {name}, False)
    with config.override(**others), pytest.raises((TypeError, ValueError)):
        call()


@pytest.mark.parametrize('name', _config.CHECKS)
def test_enabled_skips_every_check(name):
    with config.override(enabled=False):
        INVALID[name]()


@pytest.mark.parametrize(('name', 'case'), INVALID_OPTIONS.items(), ids=list(INVALID_OPTIONS))
def test_switch_skips_the_validate_array_option(name, case):
    array, options = case
    with pytest.raises((TypeError, ValueError)):
        pvv.validate_array(array, **options)
    with config.override(**{name: False}):
        out = pvv.validate_array(array, **options)
    np.testing.assert_array_equal(out, array)


@pytest.mark.parametrize(
    ('call', 'expected'),
    [
        (lambda: pvv.check_sorted([3, 1]), [3, 1]),
        (lambda: pvv.check_contains(['a'], must_contain='b'), 'b'),
        (lambda: pvv.check_iterable_items([1, 'a'], int), [1, 'a']),
    ],
    ids=['input', 'must_contain', 'iterable'],
)
def test_skipped_check_returns_what_the_check_returns(call, expected):
    with config.override(enabled=False):
        assert call() == expected


def test_skipped_check_returns_the_same_object():
    array = np.array([3, 1])
    with config.override(enabled=False):
        assert pvv.check_sorted(array) is array


def test_casts_still_run_without_checks():
    with config.override(enabled=False):
        out = pvv.validate_array(
            [[1, 2, 3]], reshape_to=(3,), dtype_out=float, must_have_shape=(1, 3), to_tuple=True
        )
        assert out == (1.0, 2.0, 3.0)
        with pytest.raises(TypeError, match='Object arrays are not supported'):
            pvv.validate_array(object())


def test_transform_dispatch_does_not_depend_on_checks():
    vtk_math = pytest.importorskip('vtkmodules.vtkCommonMath')
    matrix = vtk_math.vtkMatrix4x4()
    matrix.SetElement(0, 3, 5.0)
    expected = pvv.validate_transform4x4(matrix)
    with config.override(enabled=False):
        np.testing.assert_array_equal(pvv.validate_transform4x4(matrix), expected)


def test_defaults_are_all_true():
    assert Config().to_dict() == dict.fromkeys(('enabled', *_config.CHECKS), True)


def test_setting_a_switch_publishes_it():
    try:
        config.sorted = False
        assert config.sorted is False
        pvv.check_sorted([3, 1])
    finally:
        config.sorted = True
    with pytest.raises(ValueError, match='sorted'):
        pvv.check_sorted([3, 1])


@pytest.mark.parametrize('name', ['enabled', *_config.CHECKS])
def test_every_setter_sets_its_switch(name):
    other = Config()
    setattr(other, name, False)
    assert getattr(other, name) is False
    assert other.to_dict() == {**Config().to_dict(), name: False}


@pytest.mark.parametrize('name', sorted(set(_config.CHECKS) - {'axes', 'rotation'}))
def test_python_implementation_skips_its_check(name):
    # The public function is the C builtin when the extension is in use
    function = _accelerate.reference.get(f'check_{name}', getattr(pvv, f'check_{name}'))
    call = INVALID[name]
    with config.override(**{name: False}), mock.patch.object(pvv, f'check_{name}', function):
        call()


def test_setter_rejects_a_value_that_is_not_a_bool():
    with pytest.raises(TypeError, match=r'`sorted` must be a bool, got int\.'):
        config.sorted = 0
    assert config.sorted is True


def test_override_restores_the_switches_after_an_error():
    before = config.to_dict()
    with pytest.raises(RuntimeError), config.override(enabled=False, sorted=False):
        raise RuntimeError
    assert config.to_dict() == before
    with pytest.raises(ValueError, match='sorted'):
        pvv.check_sorted([3, 1])


def test_override_yields_the_config():
    with config.override(sorted=False) as current:
        assert current is config


def test_override_rejects_an_unknown_switch_and_restores():
    before = config.to_dict()
    with (
        pytest.raises(AttributeError, match="'bogus' is not a validation switch"),
        config.override(sorted=False, bogus=False),
    ):
        pass  # pragma: no cover
    assert config.to_dict() == before


def test_override_rejects_a_value_that_is_not_a_bool():
    with pytest.raises(TypeError, match='`finite` must be a bool'), config.override(finite=1):
        pass  # pragma: no cover
    assert config.finite is True


def test_dict_round_trip():
    values = config.to_dict()
    values['sorted'] = False
    copy = Config.from_dict(values)
    assert copy.to_dict() == values
    assert copy != config
    assert Config.from_dict(config.to_dict()) == config


def test_separate_instance_does_not_change_the_checks():
    other = Config.from_dict({'enabled': False})
    assert other.enabled is False
    other.sorted = False
    with pytest.raises(ValueError, match='sorted'):
        pvv.check_sorted([3, 1])


def test_from_dict_rejects_an_unknown_switch():
    with pytest.raises(AttributeError, match="'bogus' is not a validation switch"):
        Config.from_dict({'bogus': True})


def test_equality_and_hashing():
    assert Config() == Config()
    assert Config() != Config.from_dict({'real': False})
    assert Config() != {'enabled': True}
    with pytest.raises(TypeError, match='unhashable'):
        hash(Config())


def test_repr_shows_every_switch():
    text = repr(Config.from_dict({'sorted': False}))
    assert text.startswith('Config(enabled=True, axes=True,')
    assert 'sorted=False' in text


@pytest.mark.parametrize(('value', 'enabled'), [('false', False), ('0', False), ('1', True)])
def test_environment_sets_enabled_at_import(value, enabled):
    code = (
        'from pyvista_validation import check_sorted, config\n'
        f'assert config.enabled is {enabled}\n'
        'assert config.sorted is True\n'
        f'if {not enabled}:\n'
        '    check_sorted([3, 1])\n'
    )
    env = {**os.environ, 'PYVISTA_VALIDATION_CHECKS': value}
    subprocess.run([sys.executable, '-c', code], check=True, env=env)
