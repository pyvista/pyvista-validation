"""The C fast paths present exactly like the Python functions they stand in for."""

from __future__ import annotations

import importlib.util
import inspect
import math
import numbers
import random
import sys
from types import SimpleNamespace
from typing import Optional
from typing import Union
from unittest import mock

import numpy as np
import pytest

import pyvista_validation
from pyvista_validation import _accelerate


class NdarraySubclass(np.ndarray):
    """Minimal ndarray subclass, standing in for PyVista's pyvista_ndarray."""

    def __new__(cls, array):
        """Create the subclass from any array-like input."""
        return np.asarray(array).view(cls)


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


# The fast paths must agree with the Python implementations on every input, including the
# ones they decline: the same result or the same error.


@pytest.fixture(scope='module')
def pure():
    """Load the Python implementations so that they call each other, not the C fast paths."""

    def keep(namespace, names):
        """Leave the module's functions as they are."""

    def load(name, **modules):
        spec = importlib.util.find_spec(f'pyvista_validation.{name}')
        module = importlib.util.module_from_spec(spec)
        no_op = mock.patch.object(_accelerate, 'accelerate', new=keep)
        with no_op, mock.patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        return module

    check = load('check')
    validate = load('validate', **{'pyvista_validation.check': check})
    names = {**vars(check), **vars(validate)}
    return SimpleNamespace(**{k: v for k, v in names.items() if not k.startswith('__')})


def outcome(function, *args, **kwargs):
    """Return what a call returns, or which error it raises."""
    try:
        return ('returned', function(*args, **kwargs))
    except Exception as error:  # noqa: BLE001
        return ('raised', type(error), str(error))


def equal(a, b):
    """Return whether two results are the same value of the same type, NaN included."""
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return (
            a.dtype == b.dtype
            and a.shape == b.shape
            and a.flags.writeable == b.flags.writeable
            and bool(np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc'))
        )
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    return bool(a == b)


def assert_same(fast, slow, *args, **kwargs):
    """Assert that the accelerated and the pure call agree."""
    expected = outcome(slow, *args, **kwargs)
    actual = outcome(fast, *args, **kwargs)
    compare(actual, expected)


def assert_same_fresh(fast, slow, make, *args, **kwargs):
    """Assert agreement on an input that each call must get fresh, like an iterator."""
    expected = outcome(slow, make(), *args, **kwargs)
    actual = outcome(fast, make(), *args, **kwargs)
    if expected[0] == 'returned':
        # Two fresh inputs only compare by type
        assert actual[0] == 'returned', actual
        assert type(actual[1]) is type(expected[1]), (actual[1], expected[1])
    else:
        compare(actual, expected)


def compare(actual, expected):
    """Assert that two outcomes are the same result or the same error."""
    if expected[0] == 'raised':
        assert actual == expected
    else:
        assert actual[0] == 'returned', actual
        assert equal(actual[1], expected[1]), (actual[1], expected[1])


def sample_arrays():
    """Arrays of every dtype in every layout the loops treat differently."""
    out = {}
    for dtype in (
        'bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
        'float16', 'float32', 'float64', 'longdouble', 'complex128', 'str_',
    ):  # fmt: skip
        if dtype == 'bool':
            values = [True, False, True, True]
        elif dtype.startswith('uint'):
            values = [3, 0, 1, 2**63 + 5 if dtype == 'uint64' else 250]
        elif dtype.startswith('int'):
            values = [1, -2, 0, 5]
        elif dtype == 'complex128':
            values = [1 + 1j, 0, -1, 2]
        elif dtype == 'str_':
            values = ['b', 'a', 'c', 'a']
        else:
            values = [2.5, -1.5, -0.0, 0.0, 1.0, np.nan, np.inf, -np.inf]
        base = np.array(values, dtype=dtype)
        out[f'{dtype}-1d'] = base
        out[f'{dtype}-1d-step2'] = np.repeat(base, 2)[::2]
        out[f'{dtype}-1d-reversed'] = base[::-1]
        out[f'{dtype}-sorted'] = np.sort(base[np.isfinite(base)] if dtype[0] == 'f' else base)
        two = np.stack([base, base[::-1]])
        out[f'{dtype}-2d'] = two
        out[f'{dtype}-2d-T'] = two.T
        out[f'{dtype}-2d-F'] = np.asfortranarray(two)
        out[f'{dtype}-3d-slice'] = np.stack([two, two])[:, ::-1, 1:]
        out[f'{dtype}-0d'] = np.array(values[0], dtype=dtype)
        out[f'{dtype}-empty'] = base[:0]
    return out


ARRAYS = sample_arrays()
VALUES = [
    0,
    -1,
    1,
    0.5,
    2**63,
    2**64,
    np.float32(1),
    np.int8(-2),
    np.uint64(2**63),
    True,
    [1],
    'a',
]
RANGES = [
    [0, 1], [1, 0], [-1, 2**63], [0.5, 1.5], [0, np.inf], [np.nan, 1], [1, 2, 3], [True, False],
    np.array([0, 3], dtype=np.uint64), (-1.5, 4),
]  # fmt: skip


@pytest.mark.parametrize('array', ARRAYS.values(), ids=ARRAYS)
def test_value_checks_agree(array, pure):
    assert_same(pyvista_validation.check_finite, pure.check_finite, array)
    assert_same(pyvista_validation.check_nonnegative, pure.check_nonnegative, array)
    for strict in (True, False):
        assert_same(pyvista_validation.check_integer, pure.check_integer, array, strict=strict)
        for value in VALUES:
            assert_same(
                pyvista_validation.check_greater_than, pure.check_greater_than, array, value,
                strict=strict,
            )  # fmt: skip
            assert_same(
                pyvista_validation.check_less_than, pure.check_less_than, array, value,
                strict=strict,
            )  # fmt: skip
        for rng in RANGES:
            assert_same(
                pyvista_validation.check_range, pure.check_range, array, rng,
                strict_lower=strict, strict_upper=not strict,
            )  # fmt: skip


@pytest.mark.parametrize('array', ARRAYS.values(), ids=ARRAYS)
@pytest.mark.parametrize('axis', [None, -1, 0, 1, -2, 2, 0.0, 1.5, 'a'])
def test_sorted_agrees(array, axis, pure):
    for ascending in (True, False):
        for strict in (True, False):
            assert_same(
                pyvista_validation.check_sorted, pure.check_sorted, array,
                ascending=ascending, strict=strict, axis=axis,
            )  # fmt: skip


def validate_array_kwargs(rng, array):
    """Return a random combination of validate_array's options for the array."""
    shape = array.shape
    pick = rng.choice
    kwargs = {
        'must_have_shape': pick([None, shape, (-1,), (-1, 3), [(), shape], 0]),
        'must_have_ndim': pick([None, array.ndim, [0, 2], 1.0, 3]),
        'must_have_dtype': pick([None, np.number, np.floating, [np.integer, np.bool_], 'float64']),
        'must_have_length': pick([None, len(shape) and shape[0], [1, 2], 2.5, range(5)]),
        'must_have_min_length': pick([None, 0, 1, 3]),
        'must_have_max_length': pick([None, 1, 2, 8]),
        'must_be_nonnegative': bool(pick([False, True])),
        'must_be_finite': bool(pick([False, True])),
        'must_be_real': bool(pick([True, True, False])),
        'must_be_integer': bool(pick([False, True])),
        'must_be_sorted': pick([False, True, {'ascending': False}, {'strict': True, 'axis': 0}]),
        'must_be_in_range': pick([None, [0, 1], [-2, 2**63], [1, 0]]),
        'strict_lower_bound': bool(pick([False, True])),
        'strict_upper_bound': bool(pick([False, True])),
        'reshape_to': pick([None, -1, shape, (2, -1)]),
        'broadcast_to': pick([None, shape, (2, *shape)]),
        'dtype_out': pick([None, float, int, 'float32', bool, np.uint8]),
        'as_any': bool(pick([True, False])),
        'copy': bool(pick([False, True])),
        'to_list': bool(pick([False, True])),
        'to_tuple': bool(pick([False, True])),
    }
    return {key: value for key, value in kwargs.items() if value is not None}


@pytest.mark.parametrize('array', ARRAYS.values(), ids=ARRAYS)
def test_validate_array_agrees(array, pure):
    rng = random.Random(array.dtype.name)
    inputs = [array, NdarraySubclass(array)]
    if array.ndim <= 2 and array.size:
        inputs.append(array.tolist())
    for _ in range(40):
        kwargs = validate_array_kwargs(rng, array)
        for value in inputs:
            assert_same(pyvista_validation.validate_array, pure.validate_array, value, **kwargs)


SHAPES = [(), (-1,), (4,), (2, 4), (-1, 4), [(), (4,)], 4, -1, 0, (1.5,), 'x', None, True, [(-2,)]]
NDIMS = [0, 1, 2, [0, 2], range(3), 1.0, 1.5, [1, 1.5], 'a', -1, np.array([1, 2])]
BASES = [
    np.floating, np.integer, np.number, np.generic, float, int, bool, 'float64', 'u1',
    (np.integer, np.bool_), [np.str_], [], None, 'not a dtype', 5, np.dtype('int8'),
]  # fmt: skip


@pytest.mark.parametrize('array', ARRAYS.values(), ids=ARRAYS)
def test_structural_checks_agree(array, pure):
    assert_same(pyvista_validation.check_real, pure.check_real, array)
    for shape in SHAPES:
        assert_same(pyvista_validation.check_shape, pure.check_shape, array, shape)
    for ndim in NDIMS:
        assert_same(pyvista_validation.check_ndim, pure.check_ndim, array, ndim)
    for base in BASES:
        assert_same(pyvista_validation.check_subdtype, pure.check_subdtype, array, base)
        assert_same(pyvista_validation.check_subdtype, pure.check_subdtype, array.dtype, base)


DTYPE_LIKES = ['uint8', np.dtype('f4'), float, int, np.int8, 1.5, [1, 2], 'abc', None, object]


@pytest.mark.parametrize('obj', DTYPE_LIKES, ids=repr)
def test_check_subdtype_agrees_on_dtype_likes(obj, pure):
    for base in BASES:
        assert_same(pyvista_validation.check_subdtype, pure.check_subdtype, obj, base)


SIZED = [
    lambda: [1, 2, 3], lambda: (1,), lambda: 'abc', lambda: 5, lambda: 5.0, lambda: True,
    lambda: np.int64(5), lambda: np.float32(1), lambda: np.array(5), lambda: np.zeros((2, 3)),
    lambda: np.zeros(0), lambda: {1: 2}, lambda: None, lambda: range(4), lambda: [[1], [2, 3]],
]  # fmt: skip
LENGTH_OPTIONS = [
    {}, {'exact_length': 3}, {'exact_length': [3, 4]}, {'exact_length': 2.5},
    {'exact_length': range(6)}, {'exact_length': 'a'}, {'min_length': 2}, {'max_length': 2},
    {'min_length': 1, 'max_length': 3}, {'min_length': 3, 'max_length': 1},
    {'min_length': np.nan}, {'max_length': -1}, {'min_length': 'a'}, {'must_be_1d': True},
    {'allow_scalar': True}, {'allow_scalar': True, 'exact_length': 1, 'must_be_1d': True},
]  # fmt: skip


@pytest.mark.parametrize('make', SIZED)
def test_check_length_agrees(make, pure):
    for options in LENGTH_OPTIONS:
        assert_same(pyvista_validation.check_length, pure.check_length, make(), **options)


class StrSubclass(str):
    """A str subclass, which exact-type checks reject."""


OBJECTS = [
    lambda: 0, lambda: 1.5, lambda: True, lambda: 1 + 1j, lambda: np.float64(1),
    lambda: np.int8(1),
    lambda: np.True_, lambda: np.array(1), lambda: np.array([1, 2]), lambda: 'a', lambda: b'a',
    lambda: StrSubclass('x'), lambda: [1], lambda: [1, 'a'], lambda: (1,), lambda: range(2),
    lambda: {1}, lambda: {1: 2}, lambda: None, lambda: object(), lambda: iter([1, 2]),
    lambda: (x for x in [1, 'a']),
]  # fmt: skip
CLASSINFOS = [
    int, str, (int, str), int | None, Optional[int], Union[int, str], numbers.Number,  # noqa: UP007, UP045
    np.generic, np.ndarray, object, [int], 5, (int, 5),
]  # fmt: skip
CONTAINERS = [lambda: [1, 2], lambda: (1, 2), lambda: {1, 2}, lambda: {1: 'a'}, lambda: 'abc',
              lambda: range(3), lambda: np.array([1, 2]), lambda: 5]  # fmt: skip


@pytest.mark.parametrize('make', OBJECTS)
def test_object_checks_agree(make, pure):
    same = assert_same_fresh
    for name in ('x', 5):
        same(pyvista_validation.check_number, pure.check_number, make, name=name)
        same(pyvista_validation.check_sequence, pure.check_sequence, make, name=name)
        same(pyvista_validation.check_iterable, pure.check_iterable, make, name=name)
        for allow_subclass in (True, False):
            same(
                pyvista_validation.check_string, pure.check_string, make,
                allow_subclass=allow_subclass, name=name,
            )  # fmt: skip
            for classinfo in CLASSINFOS:
                same(
                    pyvista_validation.check_instance, pure.check_instance, make, classinfo,
                    allow_subclass=allow_subclass, name=name,
                )  # fmt: skip
                same(pyvista_validation.check_type, pure.check_type, make, classinfo, name=name)
                same(
                    pyvista_validation.check_iterable_items, pure.check_iterable_items, make,
                    classinfo, allow_subclass=allow_subclass, name=name,
                )  # fmt: skip


@pytest.mark.parametrize('make', CONTAINERS)
def test_check_contains_agrees(make, pure):
    for item in (1, 3, 'a', 'bc', None, [1]):
        assert_same(pyvista_validation.check_contains, pure.check_contains, make(), item)


FAMILIES = [
    ('validate_number', {'reshape': [True, False]}),
    ('validate_data_range', {}),
    ('validate_arrayNx3', {'reshape': [True, False]}),
    ('validate_arrayN', {'reshape': [True, False]}),
    ('validate_arrayN_unsigned', {'reshape': [True, False]}),
    ('validate_array3', {'reshape': [True, False], 'broadcast': [True, False]}),
    ('validate_dimensionality', {'reshape': [True, False]}),
]
FAMILY_INPUTS = [
    1, 2.0, -1, 2.5, np.int8(2), np.float32(1.5), [2], [1, 2], [1, 2, 3], [[1, 2, 3]],
    [[1], [2], [3]], [[1, 2, 3], [4, 5, 6]], [0, 1], [1, 0], np.zeros((0, 3)), [], '2D', '4D',
    '1d', ['1D'], 'x', np.array([1e300]), [2**63], [300], True, [True, False],
]  # fmt: skip


@pytest.mark.parametrize(('name', 'flags'), FAMILIES, ids=[f[0] for f in FAMILIES])
def test_families_agree(name, flags, pure):
    rng = random.Random(name)
    fast, slow = getattr(pyvista_validation, name), getattr(pure, name)
    for value in FAMILY_INPUTS:
        assert_same(fast, slow, value)
        for _ in range(25):
            array = np.asarray(value) if not isinstance(value, str) else np.zeros(())
            kwargs = validate_array_kwargs(rng, array)
            kwargs = {k: v for k, v in kwargs.items() if rng.random() < 0.3}
            for flag, choices in flags.items():
                kwargs[flag] = rng.choice(choices)
            if name == 'validate_arrayN_unsigned' and rng.random() < 0.5:
                kwargs['dtype_out'] = rng.choice([int, np.uint8, 'int32', float])
            assert_same(fast, slow, value, **kwargs)


def transform_inputs():
    """Return the inputs the transform validators take, and some they reject."""
    inputs = [
        np.eye(3), np.eye(4), np.eye(3, dtype=int), np.eye(4, dtype=int), np.eye(3).tolist(),
        np.arange(16.0).reshape(4, 4), np.arange(9.0).reshape(3, 3).T, np.eye(2), np.ones(3),
        np.full((3, 3), np.nan), np.full((4, 4), np.inf), 1.0, [1, 2, 3], 'abc', None, object(),
        np.array(['a']), NdarraySubclass(np.eye(3)),
    ]  # fmt: skip
    if HAS_VTK:
        rotated = vtkMatrix3x3()
        rotated.SetElement(0, 1, 5.0)
        translated = vtkTransform()
        translated.Translate(1, 2, 3)
        translated.RotateZ(90)
        inputs += [vtkMatrix3x3(), rotated, vtkMatrix4x4(), vtkTransform(), translated]
    if HAS_SCIPY:
        inputs += [Rotation.identity(), Rotation.from_euler('xyz', [10, 20, 30], degrees=True)]
    return inputs


try:
    from vtkmodules.vtkCommonMath import vtkMatrix3x3
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkCommonTransforms import vtkTransform
except ModuleNotFoundError:
    HAS_VTK = False
else:
    HAS_VTK = True
try:
    from scipy.spatial.transform import Rotation
except ModuleNotFoundError:
    HAS_SCIPY = False
else:
    HAS_SCIPY = True


@pytest.mark.parametrize('transform', transform_inputs(), ids=lambda x: type(x).__name__)
def test_transforms_agree(transform, pure):
    for must_be_finite in (True, False):
        for name in ('T', 1):
            assert_same(
                pyvista_validation.validate_transform4x4, pure.validate_transform4x4, transform,
                must_be_finite=must_be_finite, name=name,
            )  # fmt: skip
            assert_same(
                pyvista_validation.validate_transform3x3, pure.validate_transform3x3, transform,
                must_be_finite=must_be_finite, name=name,
            )  # fmt: skip


def axes_inputs():
    """Return argument tuples for validate_axes, valid and not."""
    eye = np.eye(3)
    left = np.diag([1.0, 1.0, -1.0])
    bias = eye.copy()
    bias[0, 1] = 0.1
    inputs = [
        (eye,), (2 * eye,), (left,), (bias,), (np.eye(3, dtype=int),), (eye.tolist(),),
        ([[1, 0, 0], [1, 0, 0], [0, 1, 0]],), ([[2, 0, 0], [1, 0, 0], [0, 0, 1]],),
        ([[1, 0, 0], [0, 0, 0], [0, 0, 1]],), (np.full((3, 3), np.nan),), (np.eye(3)[:2],),
        ([1, 0, 0], [0, 1, 0]), ([1, 0, 0], [2, 0, 0]), ([0, 0, 0], [0, 1, 0]),
        ([1, 0, 0], [[0, 1, 0]], (0, 0, 1)), ([[1], [0], [0]], [0, 1, 0], [0, 0, -1]),
        ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]), ('abc',), (None,),
        (np.array([[0.6, 0.8, 0], [-0.8, 0.6, 0], [0, 0, 1]]),),
    ]  # fmt: skip
    if HAS_SCIPY:
        inputs.append((Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix(),))
    return inputs


AXES_OPTIONS = [
    {},
    {'normalize': False},
    {'must_be_orthogonal': False},
    {'must_have_orientation': 'left'},
    {'must_have_orientation': None, 'must_be_orthogonal': False},
    {'must_have_orientation': 'up'},
    {'normalize': False, 'must_be_orthogonal': False, 'must_have_orientation': None},
    {'name': 1},
]


@pytest.mark.parametrize('axes', axes_inputs(), ids=str)
def test_validate_axes_agrees(axes, pure):
    for options in AXES_OPTIONS:
        assert_same(pyvista_validation.validate_axes, pure.validate_axes, *axes, **options)


def rotation_inputs():
    """Return inputs for validate_rotation, valid and not."""
    nearly = np.eye(3)
    nearly[0, 0] += 1e-4
    inputs = [
        np.eye(3), -np.eye(3), 2 * np.eye(3), nearly, np.eye(3, dtype=int), np.eye(3).tolist(),
        np.eye(4), np.full((3, 3), np.nan), np.array([[0.6, 0.8, 0], [-0.8, 0.6, 0], [0, 0, 1]]),
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), 'abc', None,
    ]  # fmt: skip
    if HAS_VTK:
        inputs.append(vtkMatrix3x3())
    if HAS_SCIPY:
        inputs += [Rotation.identity(), Rotation.from_euler('xyz', [10, 20, 30], degrees=True)]
    return inputs


@pytest.mark.parametrize('rotation', rotation_inputs(), ids=lambda x: type(x).__name__)
def test_validate_rotation_agrees(rotation, pure):
    for hand in ('right', 'left', None, 'up'):
        for tolerance in (1e-6, 1e-3, 0, 'a'):
            for name in ('R', 1):
                assert_same(
                    pyvista_validation.validate_rotation, pure.validate_rotation, rotation, hand,
                    tolerance=tolerance, name=name,
                )  # fmt: skip


# The block size and the size from which the loops drop the GIL, from src/fast/values.c.
BLOCK = 4096
GIL_RELEASE = 65536
LARGE = GIL_RELEASE + BLOCK + 2


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int16', 'int64', 'uint8'])
@pytest.mark.parametrize('index', [0, BLOCK - 1, BLOCK, BLOCK + 1, GIL_RELEASE, LARGE - 2])
def test_large_arrays_agree(dtype, index, pure):
    """One element out of place at each block and GIL-release boundary must still be caught."""
    array = np.zeros(LARGE, dtype=dtype)
    assert_same(pyvista_validation.check_range, pure.check_range, array, [0, 1])
    assert_same(pyvista_validation.check_sorted, pure.check_sorted, array)
    array[index] = 5
    assert_same(pyvista_validation.check_range, pure.check_range, array, [0, 1])
    assert_same(pyvista_validation.check_sorted, pure.check_sorted, array)
    assert_same(
        pyvista_validation.validate_array, pure.validate_array, array,
        must_be_nonnegative=True, must_be_integer=True, must_be_in_range=[0, 5],
    )  # fmt: skip
    if dtype.startswith('float'):
        array[index] = np.nan
        assert_same(pyvista_validation.check_finite, pure.check_finite, array)
        assert_same(pyvista_validation.check_nonnegative, pure.check_nonnegative, array)
        assert_same(
            pyvista_validation.validate_array, pure.validate_array, array, must_be_finite=True
        )
