"""Tests for the input validation functions."""

from __future__ import annotations

import importlib.util
import itertools
import re
import sys
import types
from typing import NamedTuple
from typing import Optional
from typing import Union

import numpy as np
import pytest

from pyvista_validation import _lazy_import
from pyvista_validation import check_contains
from pyvista_validation import check_finite
from pyvista_validation import check_greater_than
from pyvista_validation import check_instance
from pyvista_validation import check_integer
from pyvista_validation import check_iterable
from pyvista_validation import check_iterable_items
from pyvista_validation import check_length
from pyvista_validation import check_less_than
from pyvista_validation import check_ndim
from pyvista_validation import check_nonnegative
from pyvista_validation import check_number
from pyvista_validation import check_range
from pyvista_validation import check_real
from pyvista_validation import check_sequence
from pyvista_validation import check_shape
from pyvista_validation import check_sorted
from pyvista_validation import check_string
from pyvista_validation import check_subdtype
from pyvista_validation import check_type
from pyvista_validation import validate_array
from pyvista_validation import validate_array3
from pyvista_validation import validate_arrayN
from pyvista_validation import validate_arrayN_unsigned
from pyvista_validation import validate_arrayNx3
from pyvista_validation import validate_axes
from pyvista_validation import validate_data_range
from pyvista_validation import validate_dimensionality
from pyvista_validation import validate_number
from pyvista_validation import validate_rotation
from pyvista_validation import validate_transform3x3
from pyvista_validation import validate_transform4x4
from pyvista_validation._cast_array import _cast_to_list
from pyvista_validation._cast_array import _cast_to_numpy
from pyvista_validation._cast_array import _cast_to_tuple
from pyvista_validation.check import _validate_shape_value
from pyvista_validation.validate import _array_from_vtkmatrix
from pyvista_validation.validate import _set_default_kwarg_mandatory

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

needs_vtk = pytest.mark.skipif(not HAS_VTK, reason='VTK is not installed.')
needs_scipy = pytest.mark.skipif(not HAS_SCIPY, reason='SciPy is not installed.')

NUMPY_VERSION_INFO = tuple(int(part) for part in np.__version__.split('.')[:2])


class NdarraySubclass(np.ndarray):
    """Minimal ndarray subclass, standing in for PyVista's pyvista_ndarray."""

    def __new__(cls, array):
        """Create the subclass from any array-like input."""
        return np.asarray(array).view(cls)


def vtkmatrix_from_array(array):
    """Convert a 3x3 or 4x4 array into the matching VTK matrix."""
    array = np.asarray(array)
    matrix = vtkMatrix3x3() if array.shape == (3, 3) else vtkMatrix4x4()
    for i, j in itertools.product(range(array.shape[0]), range(array.shape[1])):
        matrix.SetElement(i, j, array[i, j])
    return matrix


@pytest.mark.parametrize(
    'transform_like',
    [
        lambda: np.eye(3),
        lambda: np.eye(4),
        lambda: np.eye(3).tolist(),
        lambda: np.eye(4).tolist(),
        pytest.param(lambda: vtkmatrix_from_array(np.eye(3)), marks=needs_vtk),
        pytest.param(lambda: vtkmatrix_from_array(np.eye(4)), marks=needs_vtk),
        pytest.param(vtkTransform if HAS_VTK else None, marks=needs_vtk),
    ],
    ids=['array3x3', 'array4x4', 'list3x3', 'list4x4', 'vtk3x3', 'vtk4x4', 'vtktransform'],
)
def test_validate_transform4x4(transform_like):
    result = validate_transform4x4(transform_like())
    assert type(result) is np.ndarray
    assert np.array_equal(result, np.eye(4))


def test_validate_transform4x4_raises():
    with pytest.raises(ValueError, match=re.escape('Shape must be one of [(3, 3), (4, 4)].')):
        validate_transform4x4(np.array([1, 2, 3]))
    with pytest.raises(TypeError, match='Input transform must be one of'):
        validate_transform4x4('abc')


@pytest.mark.parametrize(
    'transform_like',
    [
        lambda: np.eye(3),
        lambda: np.eye(3).tolist(),
        pytest.param(lambda: vtkmatrix_from_array(np.eye(3)), marks=needs_vtk),
        pytest.param(lambda: Rotation.from_matrix(np.eye(3)), marks=needs_scipy),
    ],
    ids=['numpy', 'list', 'vtk', 'scipy'],
)
def test_validate_transform3x3(transform_like):
    result = validate_transform3x3(transform_like())
    assert type(result) is np.ndarray
    assert np.array_equal(result, np.eye(3))


def test_validate_transform3x3_raises():
    match = 'Transform has shape (3,) which is not allowed. Shape must be (3, 3).'
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_transform3x3(np.array([1, 2, 3]))
    match = (
        'Input transform must be one of:'
        '\n\tvtkMatrix3x3'
        '\n\t3x3 np.ndarray'
        '\n\tscipy.spatial.transform.Rotation'
        "\nGot 'abc' with type <class 'str'> instead."
    )
    with pytest.raises(TypeError, match=match):
        validate_transform3x3('abc')


def test_check_subdtype():
    check_subdtype(int, np.integer)
    check_subdtype(np.dtype(int), np.integer)
    check_subdtype(np.array([1, 2, 3]), np.integer)
    check_subdtype(np.array([1.0, 2, 3]), float)
    check_subdtype(np.array([1.0, 2, 3], dtype='uint8'), 'uint8')
    check_subdtype(np.array([1.0, 2, 3]), ('uint8', float))
    match = "Input has incorrect dtype of 'int32'. The dtype must be a subtype of <class 'float'>."
    with pytest.raises(TypeError, match=match):
        check_subdtype(np.array([1, 2, 3]).astype('int32'), float)
    match = (
        "Input has incorrect dtype of 'complex128'. The dtype must be a subtype of at least "
        "one of \n(<class 'numpy.integer'>, <class 'numpy.floating'>)."
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        check_subdtype(np.array([1 + 1j, 2, 3]), (np.integer, np.floating))


def test_validate_number():
    validate_number([2.0])
    num = validate_number(1)
    assert num == 1
    assert isinstance(num, int)

    num = validate_number(2.0, to_list=False, must_have_shape=(), reshape=False)
    assert num == 2.0
    assert type(num) is np.ndarray
    assert num.dtype.type is np.float64

    match = (
        "Parameter 'must_have_shape' cannot be set for function `validate_number`.\n"
        'Its value is automatically set to `()`.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_number(1, must_have_shape=2, reshape=False)


def test_validate_data_range():
    rng = validate_data_range([0, 1])
    assert rng == (0, 1)

    rng = validate_data_range((0, 2.5), to_list=True)
    assert rng == [0.0, 2.5]

    rng = validate_data_range((-10, -10), to_tuple=False, must_have_shape=2)
    assert type(rng) is np.ndarray

    match = 'Data Range with 2 elements must be sorted in ascending order. Got:\n    array([1, 0])'
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_data_range((1, 0))

    match = (
        "Parameter 'must_have_shape' cannot be set for function `validate_data_range`.\n"
        'Its value is automatically set to `2`.'
    )
    with pytest.raises(ValueError, match=match):
        validate_data_range((0, 1), must_have_shape=3)


def test_set_default_kwarg_mandatory():
    default_value = 1
    default_key = 'k'

    # Test parameter unset
    kwargs = {}
    _set_default_kwarg_mandatory(kwargs, default_key, default_value)
    assert kwargs[default_key] == default_value

    # Test parameter already set to default
    kwargs = {}
    kwargs[default_key] = default_value
    _set_default_kwarg_mandatory(kwargs, default_key, default_value)
    assert kwargs[default_key] == default_value

    # Test parameter set to non-default
    kwargs = {}
    kwargs[default_key] = default_value * 2
    match = (
        "Parameter 'k' cannot be set for function `test_set_default_kwarg_mandatory`.\n"
        'Its value is automatically set to `1`.'
    )
    with pytest.raises(ValueError, match=match):
        _set_default_kwarg_mandatory(kwargs, default_key, default_value)


def test_check_shape():
    check_shape(0, ())
    check_shape(0, [(), 2])
    check_shape((1, 2, 3), [(), 3])
    check_shape((1, 2, 3), [-1])
    check_shape((1, 2, 3), -1)

    match = 'Input has shape (3,) which is not allowed. Shape must be 0.'
    with pytest.raises(ValueError, match=re.escape(match)):
        check_shape((1, 2, 3), 0, name='Input')

    match = 'Array has shape (3,) which is not allowed. Shape must be one of [(), (4, 5)].'
    with pytest.raises(ValueError, match=re.escape(match)):
        check_shape((1, 2, 3), [(), (4, 5)])


def test_check_ndim():
    check_ndim(0, 0)
    check_ndim(np.array(0), 0)
    check_ndim((1, 2, 3), range(2))
    check_ndim([[1, 2, 3]], (0, 2))

    match = 'Input has the incorrect number of dimensions. Got 1, expected 0.'
    with pytest.raises(ValueError, match=re.escape(match)):
        check_ndim((1, 2, 3), 0, name='Input')

    match = 'Array has the incorrect number of dimensions. Got 1, expected one of [4, 5].'
    with pytest.raises(ValueError, match=re.escape(match)):
        check_ndim((1, 2, 3), [4, 5])


def test_validate_shape_value():
    match = '`None` is not a valid shape. Use `()` instead.'
    with pytest.raises(TypeError, match=re.escape(match)):
        _validate_shape_value(None)
    shape = _validate_shape_value(())
    assert shape == ()
    shape = _validate_shape_value(1)
    assert shape == (1,)
    shape = _validate_shape_value(-1)
    assert shape == (-1,)
    shape = _validate_shape_value((1, 2, 3))
    assert shape == (
        1,
        2,
        3,
    )
    shape = _validate_shape_value((-1, 2, -1))
    assert shape == (-1, 2, -1)

    match = (
        "Shape must be an instance of any type (<class 'int'>, <class 'tuple'>). "
        "Got <class 'float'> instead."
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        _validate_shape_value(1.0)

    match = 'Shape values must all be greater than or equal to -1.'
    with pytest.raises(ValueError, match=match):
        _validate_shape_value(-2)

    match = "All items of Shape must be an instance of <class 'int'>. Got <class 'tuple'> instead."
    with pytest.raises(TypeError, match=match):
        _validate_shape_value(((1, 2), (3, 4)))


@pytest.mark.parametrize('reshape', [True, False])
def test_validate_arrayNx3(reshape):  # noqa: N802
    arr = validate_arrayNx3((1, 2, 3))
    assert arr.shape == (1, 3)
    assert np.array_equal(arr, [[1, 2, 3]])

    if not reshape:
        match = 'Array has shape (3,) which is not allowed. Shape must be (-1, 3).'
        with pytest.raises(ValueError, match=re.escape(match)):
            validate_arrayNx3((1, 2, 3), reshape=False)

    arr = validate_arrayNx3([(1, 2, 3), (4, 5, 6)], reshape=reshape)
    assert arr.shape == (2, 3)

    match = (
        "Parameter 'must_have_shape' cannot be set for function `validate_arrayNx3`.\n"
        'Its value is automatically set to `[3, (-1, 3)]`.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_arrayNx3((1, 2, 3), must_have_shape=1)
    match = 'Array has shape () which is not allowed. Shape must be one of [3, (-1, 3)].'
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_arrayNx3(0)
    with pytest.raises(ValueError, match='_input'):
        validate_arrayNx3([1, 2, 3, 4], name='_input')


@pytest.mark.parametrize('reshape', [True, False])
def test_validate_arrayN(reshape):  # noqa: N802
    # test 0D input is reshaped to 1D by default
    arr = validate_arrayN(0)
    assert arr.shape == (1,)
    assert np.array_equal(arr, [0])

    # test 2D input is reshaped to 1D by default
    arr = validate_arrayN([[1, 2, 3]])
    assert arr.shape == (3,)
    assert np.array_equal(arr, [1, 2, 3])

    if not reshape:
        match = 'Array has shape () which is not allowed. Shape must be -1.'
        with pytest.raises(ValueError, match=re.escape(match)):
            validate_arrayN(0, reshape=False)

        match = 'Array has shape (1, 3) which is not allowed. Shape must be -1.'
        with pytest.raises(ValueError, match=re.escape(match)):
            validate_arrayN([[1, 2, 3]], reshape=False)

    arr = validate_arrayN((1, 2, 3, 4, 5, 6), reshape=reshape)
    assert arr.shape == (6,)

    match = (
        "Parameter 'must_have_shape' cannot be set for function `validate_arrayN`.\n"
        'Its value is automatically set to `[(), -1, (1, -1)]`.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_arrayN((1, 2, 3), must_have_shape=1)

    match = 'Array has shape (2, 2) which is not allowed. Shape must be one of [(), -1, (1, -1)].'
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_arrayN(((1, 2), (3, 4)))
    with pytest.raises(ValueError, match='_input'):
        validate_arrayN(((1, 2), (3, 4)), name='_input')


def test_validate_arrayN_unsigned():  # noqa: N802
    # test 0D input is reshaped to 1D by default
    arr = validate_arrayN_unsigned(0.0)
    assert arr.shape == (1,)
    assert np.array_equal(arr, [0])
    assert arr.dtype.type is np.int32 or arr.dtype.type is np.int64

    arr = validate_arrayN_unsigned(0.0, dtype_out='uint8')
    assert arr.dtype.type is np.uint8

    with pytest.raises(ValueError, match=r'Shape must be -1.'):
        validate_arrayN_unsigned(0.0, reshape=False)

    match = '_input values must all be greater than or equal to 0.'
    with pytest.raises(ValueError, match=match):
        validate_arrayN_unsigned([-1, 1], name='_input')


@pytest.mark.parametrize('reshape', [True, False])
def test_validate_array3(reshape):
    # test 0D input is reshaped to len-3 1D vector with broadcasting enabled
    arr = validate_array3(0, broadcast=True)
    assert arr.shape == (3,)
    assert np.array_equal(arr, [0, 0, 0])

    # test 2D input is reshaped to 1D by default
    arr = validate_array3([[1, 2, 3]])
    assert arr.shape == (3,)
    assert np.array_equal(arr, [1, 2, 3])

    arr = validate_array3([[1], [2], [3]])
    assert arr.shape == (3,)
    assert np.array_equal(arr, [1, 2, 3])

    if not reshape:
        # test check fails with 2D input and no reshape
        match = 'Array has shape (1, 3) which is not allowed. Shape must be (3,).'
        with pytest.raises(ValueError, match=re.escape(match)):
            validate_array3([[1, 2, 3]], reshape=reshape)

        # test correct shape with broadcast and no reshape
        match = 'Shape must be one of [(3,), (), (1,)].'
        with pytest.raises(ValueError, match=re.escape(match)):
            validate_array3((1, 2, 3, 4, 5, 6), reshape=reshape, broadcast=True)
    else:
        # test error match shows correct shape with broadcast and with reshape
        match = 'Shape must be one of [(3,), (1, 3), (3, 1), (), (1,)]'
        with pytest.raises(ValueError, match=re.escape(match)):
            validate_array3((1, 2, 3, 4, 5, 6), reshape=reshape, broadcast=True)

    # test shape cannot be overridden
    match = (
        "Parameter 'must_have_shape' cannot be set for function `validate_array3`.\n"
        'Its value is automatically set to `[(3,), (1, 3), (3, 1)]`.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_array3((1, 2, 3), must_have_shape=3)


def test_check_range_accepts_values_within_the_range():
    check_range((1, 2, 3), [1, 3])


def test_check_range_accepts_a_degenerate_range():
    check_range([2, 2], [2, 2])


def test_check_range_rejects_values_above_the_upper_bound():
    with pytest.raises(
        ValueError, match=re.escape('Array values must all be less than or equal to 2.')
    ):
        check_range((1, 2, 3), [1, 2])


def test_check_range_rejects_values_below_the_lower_bound():
    with pytest.raises(
        ValueError, match=re.escape('Input values must all be greater than or equal to 2.')
    ):
        check_range((1, 2, 3), [2, 3], name='Input')


def test_check_range_strict_upper_excludes_the_bound():
    with pytest.raises(ValueError, match=re.escape('Array values must all be less than 3.')):
        check_range((1, 2, 3), [1, 3], strict_upper=True)


def test_check_range_strict_lower_excludes_the_bound():
    with pytest.raises(ValueError, match=re.escape('Array values must all be greater than 1.')):
        check_range((1, 2, 3), [1, 3], strict_lower=True)


@pytest.mark.parametrize('array', [(1,), [1], np.ndarray((1,))])
def test_check_length_accepts_a_length_one_container(array):
    check_length(array)


def test_check_length_accepts_every_constraint_at_once():
    check_length((1,), exact_length=1, min_length=1, max_length=1, must_be_1d=True)


def test_check_length_accepts_any_of_several_exact_lengths():
    check_length((1,), exact_length=[1, 2.0])


def test_check_length_rejects_a_non_integer_exact_length():
    with pytest.raises(ValueError, match=r"'exact_length' must have integer-like values."):
        check_length((1,), exact_length=(1, 2.4), name='_input')


def test_check_length_rejects_a_wrong_exact_length():
    match = '_input must have a length equal to any of: 1. Got length 2 instead.'
    with pytest.raises(ValueError, match=match):
        check_length((1, 2), exact_length=1, name='_input')


def test_check_length_rejects_when_no_exact_length_matches():
    match = '_input must have a length equal to any of: [3, 4]. Got length 2 instead.'
    with pytest.raises(ValueError, match=re.escape(match)):
        check_length((1, 2), exact_length=[3, 4], name='_input')


def test_check_length_rejects_a_too_long_container():
    match = '_input must have a maximum length of 1. Got length 2 instead.'
    with pytest.raises(ValueError, match=match):
        check_length((1, 2), max_length=1, name='_input')


def test_check_length_rejects_a_too_short_container():
    match = '_input must have a minimum length of 2. Got length 1 instead.'
    with pytest.raises(ValueError, match=match):
        check_length((1,), min_length=2, name='_input')


def test_check_length_rejects_a_min_above_the_max():
    match = 'Range with 2 elements must be sorted in ascending order. Got:\n    array([4, 2])'
    with pytest.raises(ValueError, match=re.escape(match)):
        check_length((1, 2, 3), min_length=4, max_length=2)


def test_check_length_rejects_a_multidimensional_array_when_1d_required():
    with pytest.raises(ValueError, match=re.escape('Shape must be -1.')):
        check_length(((1, 2), (3, 4)), must_be_1d=True)


class Constraint(NamedTuple):
    """A validate_array constraint, with an array that satisfies it and one that does not."""

    kwargs: dict
    valid: object
    invalid: object
    error_type: type
    error_match: str


CONSTRAINTS = (
    Constraint(
        {'must_be_finite': True, 'must_be_real': False},
        0,
        np.inf,
        ValueError,
        'must have finite values',
    ),
    Constraint({'must_be_real': True}, 0, 1 + 1j, TypeError, 'must have real numbers'),
    Constraint({'must_be_integer': True}, 0.0, 0.1, ValueError, 'must have integer-like values'),
    Constraint({'must_be_sorted': True}, [0, 1], [1, 0], ValueError, 'must be sorted'),
    Constraint(
        {'must_be_sorted': {'ascending': True, 'strict': False, 'axis': -1}},
        [0, 1],
        [1, 0],
        ValueError,
        'must be sorted',
    ),
)
CONSTRAINT_IDS = ('finite', 'real', 'integer', 'sorted', 'sorted-kwargs')

REPRESENTATIONS = ('tuple', 'list', 'ndarray', 'subclass')


def as_representation(array, representation, *, stacked=False):
    """Return ``array`` as the named container type, optionally stacked to 2D."""
    array = np.array(array)
    if stacked:
        array = np.stack((array, array), axis=0)
        array = np.stack((array, array), axis=1)
    if representation == 'tuple':
        return _cast_to_tuple(array)
    if representation == 'list':
        return array.tolist()
    if representation == 'ndarray':
        return array
    return NdarraySubclass(array)


@pytest.fixture(params=REPRESENTATIONS)
def representation(request):
    """Each container type validate_array accepts as input."""
    return request.param


@pytest.fixture(params=[False, True], ids=['scalar-or-1d', 'stacked'])
def stacked(request):
    """Whether the input is stacked into extra dimensions."""
    return request.param


def constraint_kwargs(array, constraint):
    """Return the constraint's kwargs plus every shape/length constraint ``array`` satisfies."""
    as_array = np.array(array)
    shape = as_array.shape
    return dict(
        **constraint.kwargs,
        must_have_dtype=np.number,
        must_have_length=range(as_array.size + 1),
        must_have_min_length=1,
        must_have_max_length=as_array.size,
        must_have_shape=shape,
        must_have_ndim=len(shape),
        reshape_to=shape,
        broadcast_to=shape,
        must_be_in_range=(np.min(as_array), np.max(as_array)),
        must_be_nonnegative=bool(np.all(as_array > 0)),
    )


@pytest.mark.parametrize('constraint', CONSTRAINTS, ids=CONSTRAINT_IDS)
def test_validate_array_accepts_valid(constraint, representation, stacked):
    array = as_representation(constraint.valid, representation, stacked=stacked)
    assert np.array_equal(validate_array(array, **constraint_kwargs(array, constraint)), array)


@pytest.mark.parametrize('constraint', CONSTRAINTS, ids=CONSTRAINT_IDS)
def test_validate_array_rejects_invalid(constraint, representation, stacked):
    valid = as_representation(constraint.valid, representation, stacked=stacked)
    invalid = as_representation(constraint.invalid, representation, stacked=stacked)
    with pytest.raises(constraint.error_type, match=constraint.error_match):
        validate_array(invalid, **constraint_kwargs(valid, constraint))


@pytest.mark.parametrize('name', ['_array', '_input'])
@pytest.mark.parametrize('constraint', CONSTRAINTS, ids=CONSTRAINT_IDS)
def test_validate_array_error_reports_name(constraint, name):
    kwargs = constraint_kwargs(constraint.valid, constraint)
    with pytest.raises(constraint.error_type, match=name):
        validate_array(constraint.invalid, name=name, **kwargs)


@pytest.mark.parametrize('dtype_out', [np.float32, np.float64])
def test_validate_array_casts_to_dtype_out(representation, dtype_out):
    array = as_representation([0, 1], representation)
    assert validate_array(array, dtype_out=dtype_out).dtype.type is dtype_out


def test_validate_array_to_list(representation):
    array = as_representation([0, 1], representation)
    out = validate_array(array, to_list=True)
    assert isinstance(out, list)
    assert out == [0, 1]


def test_validate_array_to_tuple(representation):
    array = as_representation([0, 1], representation)
    out = validate_array(array, to_tuple=True)
    assert type(out) is tuple
    assert out == (0, 1)


def test_validate_array_to_tuple_wins_over_to_list():
    assert type(validate_array([0, 1], to_tuple=True, to_list=True)) is tuple


@pytest.mark.parametrize(('to_list', 'to_tuple'), [(True, False), (False, True), (True, True)])
def test_validate_array_scalar_converts_to_scalar(to_list, to_tuple):
    out = validate_array(1.0, to_list=to_list, to_tuple=to_tuple)
    assert isinstance(out, (float, int))
    assert out == 1.0


def test_validate_array_scalar_stays_an_array_without_conversion():
    assert isinstance(validate_array(1.0), np.ndarray)


def test_validate_array_copy_returns_a_new_array():
    array = np.array([0.0, 1.0])
    assert validate_array(array, copy=True) is not array


def test_validate_array_no_copy_reuses_the_input():
    array = np.array([0.0, 1.0])
    assert validate_array(array, copy=False, as_any=True, dtype_out=array.dtype) is array


def test_validate_array_no_copy_still_copies_to_change_dtype():
    array = np.array([0.0, 1.0])
    assert validate_array(array, copy=False, as_any=True, dtype_out=np.float32) is not array


@pytest.mark.parametrize(
    ('representation', 'expected_type'),
    [('ndarray', np.ndarray), ('subclass', NdarraySubclass)],
)
def test_validate_array_as_any_preserves_the_input_type(representation, expected_type):
    array = as_representation([0, 1], representation)
    assert type(validate_array(array, as_any=True)) is expected_type


def test_validate_array_without_as_any_returns_a_base_ndarray_view():
    array = as_representation([0, 1], 'subclass')
    out = validate_array(array, as_any=False)
    assert type(out) is np.ndarray
    assert np.shares_memory(out, array)


@pytest.mark.parametrize('array', [(True,), 'abc'])
def test_validate_array_non_numeric(array):
    match = 'Array must have real numbers.'
    with pytest.raises(TypeError, match=match):
        assert validate_array(array)
    assert validate_array(array, must_be_real=False)


def test_validate_array_text():
    array = validate_array(['b', 'a'], must_be_real=False)
    assert array.dtype.type is np.str_
    assert validate_array(['a', 'b'], must_be_real=False, to_list=True) == ['a', 'b']
    kwargs = {'must_be_real': False, 'must_be_sorted': True, 'must_have_shape': 2}
    assert validate_array(['a', 'b'], to_tuple=True, **kwargs) == ('a', 'b')
    with pytest.raises(ValueError, match='must be sorted'):
        validate_array(['b', 'a'], **kwargs)
    # The value checks are for numbers
    with pytest.raises(TypeError):
        validate_array(['a'], must_be_real=False, must_be_nonnegative=True)


def test_validate_families_accept_text():
    assert validate_data_range(['a', 'b'], must_be_real=False) == ('a', 'b')
    assert validate_arrayN(['a', 'b'], must_be_real=False, to_list=True) == ['a', 'b']
    assert validate_array3(['a', 'b', 'c'], must_be_real=False, to_tuple=True) == ('a', 'b', 'c')
    array = validate_arrayNx3([['a', 'b', 'c']], must_be_real=False, to_list=True)
    assert array == [['a', 'b', 'c']]
    with pytest.raises(TypeError, match='must have real numbers'):
        validate_arrayN(['a', 'b'])


def test_check_functions_accept_text():
    text = np.array(['a', 'b'])
    assert check_shape(text, 2) is text
    assert check_ndim(text, 1) is text
    assert check_sorted(text) is text
    with pytest.raises(ValueError, match='must be sorted'):
        check_sorted(['b', 'a'])
    with pytest.raises(TypeError, match='must have real numbers'):
        check_real(text)


def test_check_instance_accepts_a_matching_instance():
    check_instance(0, int)
    check_instance(0.0, (int, float))


def test_check_instance_accepts_a_subclass():
    check_instance(True, int)  # bool subclasses int


def test_check_instance_rejects_a_non_instance():
    with pytest.raises(TypeError, match='Object must be an instance of'):
        check_instance('0', int)


@pytest.mark.parametrize('name', ['_input', '_object'])
def test_check_instance_non_instance_error_reports_the_name(name):
    with pytest.raises(TypeError, match=f'{name} must be an instance of'):
        check_instance('0', int, name=name)


def test_check_instance_rejects_a_list_of_types():
    # classinfo must be a type or tuple of types, never a list.
    with pytest.raises(TypeError):
        check_instance(0, [int, float])


def test_check_instance_rejects_a_non_string_name():
    match = "Name must be a string, got <class 'int'> instead."
    with pytest.raises(TypeError, match=match):
        check_instance(0, int, name=0)


def test_check_type_accepts_an_exact_type():
    check_type(0, int)
    check_type(0, int, name='abc')


def test_check_type_rejects_a_subclass():
    # check_type is check_instance without subclass tolerance.
    with pytest.raises(TypeError, match='must have type'):
        check_type(True, int)


def test_check_type_rejects_a_wrong_type():
    with pytest.raises(TypeError, match='Object must have type'):
        check_type('0', int)


@pytest.mark.parametrize('name', ['_input', '_object'])
def test_check_type_wrong_type_error_reports_the_name(name):
    with pytest.raises(TypeError, match=f'{name} must have type'):
        check_type('0', int, name=name)


@pytest.mark.parametrize('classinfo', [(int, float), int | float])
def test_check_type_accepts_any_of_several_types(classinfo):
    check_type(0, classinfo)
    check_type(0.0, classinfo)


@pytest.mark.parametrize('classinfo', [(int, float), int | float])
@pytest.mark.parametrize('name', ['_input', '_object'])
def test_check_type_rejects_when_no_type_matches(classinfo, name):
    with pytest.raises(TypeError, match='Object must have one of the following types'):
        check_type('0', classinfo)
    with pytest.raises(TypeError, match=f'{name} must have one of the following types'):
        check_type('0', classinfo, name=name)


def test_check_type_rejects_a_non_string_name():
    with pytest.raises(TypeError, match='Name must be a string'):
        check_type(0, int, name=1)


def test_check_string():
    check_string('abc')
    check_string('abc', name='123')
    match = "Value must be an instance of <class 'str'>. Got <class 'int'> instead."
    with pytest.raises(TypeError, match=match):
        check_string(0, name='Value')
    match = "Object must be an instance of <class 'str'>. Got <class 'int'> instead."
    with pytest.raises(TypeError, match=match):
        check_string(0)
    match = "Name must be a string, got <class 'float'> instead."
    with pytest.raises(TypeError, match=match):
        check_string('abc', name=0.0)

    class StrSubclass(str):
        pass

    check_string(StrSubclass(), allow_subclass=True)
    with pytest.raises(TypeError, match=r"Object must have type <class 'str'>."):
        check_string(StrSubclass(), allow_subclass=False)


def test_check_less_than():
    check_less_than([0], 1)
    check_less_than(np.eye(3), 1, strict=False)
    match = 'Array values must all be less than 0.'
    with pytest.raises(ValueError, match=match):
        check_less_than(0, 0, strict=True)
    match = '_input values must all be less than or equal to 0.'
    with pytest.raises(ValueError, match=match):
        check_less_than(1, 0, strict=False, name='_input')


def test_check_greater_than():
    check_greater_than([1], 0)
    check_greater_than(np.eye(3), 0, strict=False)
    match = 'Array values must all be greater than 0.'
    with pytest.raises(ValueError, match=match):
        check_greater_than(0, 0, strict=True)
    match = '_input values must all be greater than or equal to 0.'
    with pytest.raises(ValueError, match=match):
        check_greater_than(-1, 0, strict=False, name='_input')


def test_check_real():
    check_real(1)
    check_real(-2.0)
    check_real(np.array(2.0, dtype='uint8'))
    match = 'Array must have real numbers.'
    with pytest.raises(TypeError, match=match):
        check_real(1 + 1j)
    with pytest.raises(TypeError, match=match):
        check_real(True)
    match = '_input must have real numbers.'
    with pytest.raises(TypeError, match=match):
        check_real(1 + 1j, name='_input')


def test_check_finite():
    check_finite(0)
    match = '_input must have finite values.'
    with pytest.raises(ValueError, match=match):
        check_finite(np.nan, name='_input')


def test_check_integer():
    check_integer(1)
    check_integer([2, 3.0])
    match = (
        "has incorrect dtype of 'float64'. The dtype must be a subtype of <class 'numpy.integer'>."
    )
    with pytest.raises(TypeError, match=match):
        check_integer([2, 3.0], strict=True)
    match = '_input must have integer-like values.'
    with pytest.raises(ValueError, match=match):
        check_integer([2, 3.4], strict=False, name='_input')


def test_check_sequence():
    check_sequence((1,), name='abc')
    check_sequence(range(3))
    check_sequence('abc')
    with pytest.raises(TypeError, match='_input'):
        check_sequence(np.array(1), name='_input')


def test_check_iterable():
    check_iterable((1,), name='abc')
    check_iterable(range(3))
    check_iterable('abc')
    check_iterable(np.array(1))
    with pytest.raises(TypeError, match='_input'):
        check_iterable(1, name='_input')


def test_check_nonnegative():
    check_nonnegative(0)
    check_nonnegative(np.eye(3))
    match = 'Array values must all be greater than or equal to 0.'
    with pytest.raises(ValueError, match=match):
        check_nonnegative(-1)


SORTED_SHAPES = ((8,), (4, 6), (2, 3, 4))


def sorted_axes(shape):
    """Return ``None`` plus every in-bounds axis for ``shape``."""
    ndim = len(shape)
    return [None, *range(-ndim, ndim)]


SHAPE_AXIS = [
    pytest.param(shape, axis, id=f'{"x".join(map(str, shape))}-axis{axis}')
    for shape in SORTED_SHAPES
    for axis in sorted_axes(shape)
]


def sorted_arrays(shape, axis):
    """Return ascending, strict-ascending, descending and strict-descending arrays."""
    strict_ascending = np.arange(int(np.prod(shape))).reshape(shape)
    ascending = np.repeat(strict_ascending, 2, axis=axis)
    return (
        ascending,
        strict_ascending,
        np.flip(ascending, axis=axis),
        np.flip(strict_ascending, axis=axis),
    )


@pytest.mark.parametrize('strict', [True, False])
@pytest.mark.parametrize('ascending', [True, False])
def test_check_sorted_accepts_a_scalar(ascending, strict):
    check_sorted(0, ascending=ascending, strict=strict)


@pytest.mark.parametrize(
    ('shape', 'axis'), [((8,), 1), ((8,), -2), ((4, 6), 2), ((4, 6), -3), ((2, 3, 4), 3)]
)
def test_check_sorted_axis_out_of_bounds_raises(shape, axis):
    array = np.arange(int(np.prod(shape))).reshape(shape)
    match = f'Axis {axis} is out of bounds for ndim {array.ndim}'
    with pytest.raises(ValueError, match=match):
        check_sorted(array, axis=axis)


def test_check_sorted_axis_none_flattens_the_array():
    # Every row ascends, but the flattened array does not.
    array = np.array([[0, 2, 4], [1, 3, 5]])
    check_sorted(array, axis=-1)
    with pytest.raises(ValueError, match='must be sorted'):
        check_sorted(array, axis=None)


@pytest.mark.parametrize(('shape', 'axis'), SHAPE_AXIS)
def test_check_sorted_accepts_ascending(shape, axis):
    ascending, strict_ascending, _, _ = sorted_arrays(shape, axis)
    check_sorted(ascending, axis=axis, ascending=True, strict=False)
    check_sorted(strict_ascending, axis=axis, ascending=True, strict=False)
    check_sorted(strict_ascending, axis=axis, ascending=True, strict=True)


@pytest.mark.parametrize(('shape', 'axis'), SHAPE_AXIS)
def test_check_sorted_accepts_descending(shape, axis):
    _, _, descending, strict_descending = sorted_arrays(shape, axis)
    check_sorted(descending, axis=axis, ascending=False, strict=False)
    check_sorted(strict_descending, axis=axis, ascending=False, strict=False)
    check_sorted(strict_descending, axis=axis, ascending=False, strict=True)


@pytest.mark.parametrize(('shape', 'axis'), SHAPE_AXIS)
def test_check_sorted_rejects_descending_when_ascending_expected(shape, axis):
    _, _, descending, strict_descending = sorted_arrays(shape, axis)
    for array in (descending, strict_descending):
        with pytest.raises(ValueError, match='must be sorted in ascending order'):
            check_sorted(array, axis=axis, ascending=True, strict=False)


@pytest.mark.parametrize(('shape', 'axis'), SHAPE_AXIS)
def test_check_sorted_rejects_ascending_when_descending_expected(shape, axis):
    ascending, strict_ascending, _, _ = sorted_arrays(shape, axis)
    for array in (ascending, strict_ascending):
        with pytest.raises(ValueError, match='must be sorted in descending order'):
            check_sorted(array, axis=axis, ascending=False, strict=False)


@pytest.mark.parametrize(('shape', 'axis'), SHAPE_AXIS)
def test_check_sorted_strict_rejects_duplicates(shape, axis):
    ascending, _, descending, _ = sorted_arrays(shape, axis)
    with pytest.raises(ValueError, match=re.escape('must be sorted in strict ascending order')):
        check_sorted(ascending, axis=axis, ascending=True, strict=True)
    with pytest.raises(ValueError, match='must be sorted in strict descending order'):
        check_sorted(descending, axis=axis, ascending=False, strict=True)


def test_check_iterable_items():
    check_iterable_items([1, 2, 3], int)
    check_iterable_items(('a', 'b', 'c'), str)
    check_iterable_items('abc', str)
    check_iterable_items(range(10), int)
    match = (
        "All items of Iterable must be an instance of <class 'str'>. Got <class 'int'> instead."
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        check_iterable_items(['abc', 1], str)
    with pytest.raises(TypeError, match='All items of _input'):
        check_iterable_items(['abc', 1], str, name='_input')


def test_check_number():
    check_number(1)
    check_number(1 + 1j)
    match = (
        "_input must be an instance of <class 'numbers.Number'>. "
        "Got <class 'numpy.ndarray'> instead."
    )
    with pytest.raises(TypeError, match=match):
        check_number(np.array(0), name='_input')
    match = 'Object must be'
    with pytest.raises(TypeError, match=match):
        check_number(np.array(0))


def test_check_contains():
    check_contains(['foo', 'bar'], must_contain='foo')
    match = "Input 'foo' is not valid. Input must be one of: \n\t['cat', 'bar']"
    with pytest.raises(ValueError, match=re.escape(match)):
        check_contains(['cat', 'bar'], must_contain='foo')
    match = "_input '5' is not valid. _input must be in: \n\trange(0, 4)"
    with pytest.raises(ValueError, match=re.escape(match)):
        check_contains(range(4), must_contain=5, name='_input')


RIGHT_HANDED_AXES = np.eye(3)
LEFT_HANDED_AXES = np.array([[1, 0.0, 0], [0, 1, 0], [0, 0, -1]])


@pytest.mark.parametrize(
    'axes',
    [
        (RIGHT_HANDED_AXES,),
        ([1, 0, 0], [[0, 1, 0]], (0, 0, 1)),
    ],
    ids=['matrix', 'three-vectors'],
)
def test_validate_axes_accepts_right_handed_axes(axes):
    assert np.array_equal(validate_axes(*axes), RIGHT_HANDED_AXES)


@pytest.mark.parametrize(
    ('orientation', 'expected_third'), [('right', [0, 0, 1]), ('left', [0, 0, -1])]
)
def test_validate_axes_computes_the_third_vector_from_the_orientation(orientation, expected_third):
    axes = validate_axes(
        [[1], [0], [0]], [[0, 1, 0]], must_have_orientation=orientation, must_be_orthogonal=True
    )
    assert np.array_equal(axes[:2], [[1, 0, 0], [0, 1, 0]])
    assert np.array_equal(axes[2], expected_third)


@pytest.mark.parametrize('name', ['_input', '_axes'])
def test_validate_axes_rejects_parallel_vectors(name):
    with pytest.raises(ValueError, match=f'{name} cannot be parallel.'):
        validate_axes([[1, 0, 0], [1, 0, 0], [0, 1, 0]], name=name)
    with pytest.raises(ValueError, match=re.escape('Axes cannot be parallel.')):
        validate_axes([[0, 1, 0], [1, 0, 0], [0, 1, 0]])


@pytest.mark.parametrize('zero_row', [0, 1, 2])
def test_validate_axes_rejects_a_zero_vector(zero_row):
    axes = np.eye(3)
    axes[zero_row] = 0
    with pytest.raises(ValueError, match=re.escape('Axes cannot be zeros.')):
        validate_axes(axes)


def test_validate_axes_zero_vector_error_reports_the_name():
    axes = np.eye(3)
    axes[2] = 0
    with pytest.raises(ValueError, match=re.escape('_input cannot be zeros.')):
        validate_axes(axes, name='_input')


def test_validate_axes_keeps_the_scale_without_normalize():
    scaled = RIGHT_HANDED_AXES * 2
    assert np.array_equal(validate_axes(scaled, normalize=False), scaled)


def test_validate_axes_normalize_returns_unit_vectors():
    assert np.array_equal(validate_axes(RIGHT_HANDED_AXES * 2, normalize=True), RIGHT_HANDED_AXES)


@pytest.mark.parametrize(
    ('axes', 'orientation'),
    [
        (LEFT_HANDED_AXES, None),
        (LEFT_HANDED_AXES, 'left'),
        (RIGHT_HANDED_AXES, None),
        (RIGHT_HANDED_AXES, 'right'),
    ],
    ids=['left-any', 'left-left', 'right-any', 'right-right'],
)
def test_validate_axes_accepts_a_matching_orientation(axes, orientation):
    validate_axes(axes, must_have_orientation=orientation)


@pytest.mark.parametrize(
    ('axes', 'orientation', 'expected'),
    [
        (LEFT_HANDED_AXES, 'right', 'right-handed'),
        (RIGHT_HANDED_AXES, 'left', 'left-handed'),
    ],
)
def test_validate_axes_rejects_a_mismatched_orientation(axes, orientation, expected):
    match = f'_input do not have a {expected} orientation.'
    with pytest.raises(ValueError, match=match):
        validate_axes(axes, must_have_orientation=orientation, name='_input')


def test_validate_axes_two_vectors_require_an_orientation():
    # The third vector cannot be computed without one.
    match = '_input orientation must be specified when only two vectors are given.'
    with pytest.raises(ValueError, match=match):
        validate_axes([1, 0, 0], [0, 1, 0], must_have_orientation=None, name='_input')


@pytest.mark.parametrize('bias_index', [(0, 1), (1, 0), (2, 0)])
def test_validate_axes_orthogonal(bias_index):
    axes_right = np.eye(3)
    axes_right[bias_index[0], bias_index[1]] = 0.1
    axes_left = np.array([[1, 0.0, 0], [0, 1, 0], [0, 0, -1]])
    axes_left[bias_index[0], bias_index[1]] = 0.1

    match = 'Axes are not orthogonal.'
    axes = validate_axes(
        axes_right,
        must_be_orthogonal=False,
        normalize=False,
        must_have_orientation='right',
    )
    assert np.array_equal(axes, axes_right)
    with pytest.raises(ValueError, match=match):
        validate_axes(axes_right, must_be_orthogonal=True)

    axes = validate_axes(
        axes_left,
        must_be_orthogonal=False,
        normalize=False,
        must_have_orientation='left',
    )
    assert np.array_equal(axes, axes_left)
    with pytest.raises(ValueError, match=match):
        validate_axes(axes_left, must_be_orthogonal=True)


def test_validate_rotation():
    identity3 = np.eye(3)
    validated = validate_rotation(identity3)
    assert np.array_equal(validated, identity3)
    validated = validate_rotation(identity3, must_have_handedness='right')
    assert np.array_equal(validated, identity3)
    match = (
        'Rotation has incorrect handedness. Expected a left-handed rotation, '
        'but got a right-handed rotation instead.'
    )
    with pytest.raises(ValueError, match=match):
        validate_rotation(identity3, must_have_handedness='left')

    validated = validate_rotation(-identity3)
    assert np.array_equal(validated, -identity3)
    validated = validate_rotation(-identity3, must_have_handedness='left')
    assert np.array_equal(validated, -identity3)
    match = (
        'Rotation has incorrect handedness. Expected a right-handed rotation, '
        'but got a left-handed rotation instead.'
    )
    with pytest.raises(ValueError, match=match):
        validate_rotation(-identity3, must_have_handedness='right')

    match = 'Rotation is not valid. Rotation must be orthogonal.'
    with pytest.raises(ValueError, match=match):
        validate_rotation(identity3 * 2)


def test_validate_rotation_tolerance():
    # Define valid rotation matrix which fails the check if the tolerance is too low
    # Matrix values come directly from a CI test failure
    # See https://github.com/pyvista/pyvista/pull/7053#issuecomment-2571663768
    rotation = np.array(
        [
            [6.1753786e-01, 4.8325321e-01, -6.2057501e-01],
            [-2.1952267e-04, 7.8909826e-01, 6.1426693e-01],
            [7.8654110e-01, -3.7919688e-01, 4.8740414e-01],
        ]
    )
    validate_rotation(rotation)


@pytest.mark.parametrize('as_any', [True, False])
@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize('dtype', [None, float])
def test_cast_to_numpy(as_any, copy, dtype):
    array_in = NdarraySubclass([1, 2])
    array_out = _cast_to_numpy(array_in, copy=copy, as_any=as_any, dtype=dtype)
    assert np.array_equal(array_out, array_in)
    if as_any:
        assert type(array_out) is NdarraySubclass
    else:
        assert type(array_out) is np.ndarray

    if copy:
        assert not np.shares_memory(array_out, array_in)

    if dtype is None:
        assert array_out.dtype.type is array_in.dtype.type
    else:
        assert array_out.dtype.type is np.dtype(dtype).type


@pytest.mark.filterwarnings('ignore:Creating an ndarray from ragged nested sequences:UserWarning')
def test_cast_to_numpy_raises():
    if NUMPY_VERSION_INFO < (1, 26) and sys.platform == 'linux':
        err = TypeError
        match = 'Object arrays are not supported.'
    else:
        err = ValueError
        match = "Input cannot be cast as <class 'numpy.ndarray'>."
    with pytest.raises(err, match=match):
        _cast_to_numpy([[1], [2, 3]])

    match = 'Object arrays are not supported.'
    with pytest.raises(TypeError, match=match):
        _cast_to_numpy(list)


def test_cast_to_numpy_must_be_real():
    _ = _cast_to_numpy([0, 1], must_be_real=True)
    _ = _cast_to_numpy('abc', must_be_real=False)

    match = "Array must have real numbers. Got dtype <class 'numpy.complex128'>"
    with pytest.raises(TypeError, match=match):
        _ = _cast_to_numpy([0, 1 + 1j], must_be_real=True)
    match = "Array must have real numbers. Got dtype <class 'numpy.str_'>"
    with pytest.raises(TypeError, match=match):
        _ = _cast_to_numpy('abc', must_be_real=True)


def test_cast_to_tuple():
    array_in = np.zeros(shape=(2, 2, 3))
    array_tuple = _cast_to_tuple(array_in)
    assert array_tuple == (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    array_list = array_in.tolist()
    assert np.array_equal(array_tuple, array_list)


def test_cast_to_list():
    array_in = np.zeros(shape=(3, 4, 5))
    array_list = _cast_to_list(array_in)
    assert np.array_equal(array_in, array_list)


@needs_vtk
@pytest.mark.parametrize('shape', [(3, 3), (4, 4)])
def test_array_from_vtkmatrix(shape):
    expected = np.random.default_rng().random(shape)
    mat = vtkMatrix3x3() if shape == (3, 3) else vtkMatrix4x4()
    for i, j in itertools.product(range(shape[0]), range(shape[1])):
        mat.SetElement(i, j, expected[i, j])
    actual = _array_from_vtkmatrix(mat, shape=shape)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    ('dimensionality', 'reshape', 'expected_dimensionality'),
    [
        (0, True, 0),
        (0, False, 0),
        (1, True, 1),
        (2, True, 2),
        (3, True, 3),
        ('0D', True, 0),
        ('1D', True, 1),
        ('2D', True, 2),
        ('3D', True, 3),
        (('1D',), True, 1),
    ],
)
def test_validate_dimensionality(dimensionality, reshape, expected_dimensionality):
    assert validate_dimensionality(dimensionality, reshape=reshape) == expected_dimensionality


@pytest.mark.parametrize(
    ('dimensionality', 'message'),
    [
        (-1, 'Dimensionality values must all be greater than or equal to 0.'),
        ('5D', 'Dimensionality values must all be less than or equal to 3.'),
        (
            [1, 1],
            'Dimensionality has shape (2,) which is not allowed. Shape must be one of [(), (1,)].',
        ),
        (
            'invalid',
            (
                '`invalid` is not a valid dimensionality. '
                'Use one of [0, 1, 2, 3, "0D", "1D", "2D", "3D"].'
            ),
        ),
    ],
)
def test_validate_dimensionality_errors(dimensionality, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_dimensionality(dimensionality)


def test_lazy_import_rejects_unknown_names():
    with pytest.raises(AttributeError, match="has no attribute 'not_a_real_name'"):
        _ = _lazy_import.not_a_real_name


@needs_vtk
def test_lazy_import_returns_the_real_vtk_classes():
    assert _lazy_import.vtkMatrix3x3 is vtkMatrix3x3
    assert _lazy_import.vtkMatrix4x4 is vtkMatrix4x4
    assert _lazy_import.vtkTransform is vtkTransform


@needs_scipy
def test_lazy_import_returns_the_real_rotation():
    assert _lazy_import.Rotation is Rotation


def test_lazy_import_caches_in_module_globals():
    """A resolved name is stored as a global so __getattr__ runs only once."""
    _lazy_import.__dict__.pop('vtkMatrix3x3', None)
    assert _lazy_import.vtkMatrix3x3 is not None
    assert 'vtkMatrix3x3' in vars(_lazy_import)


def test_lazy_import_follows_the_vtk_backend(monkeypatch):
    """PYVISTA_VTK_BACKEND selects a flat backend, matching PyVista."""
    backend = types.ModuleType('fake_flat_vtk')
    for name in ('vtkMatrix3x3', 'vtkMatrix4x4', 'vtkTransform'):
        setattr(backend, name, type(name, (), {}))
    monkeypatch.setitem(sys.modules, 'fake_flat_vtk', backend)
    monkeypatch.setenv('PYVISTA_VTK_BACKEND', 'fake_flat_vtk')
    for name in ('vtkMatrix3x3', 'vtkMatrix4x4', 'vtkTransform'):
        _lazy_import.__dict__.pop(name, None)
    try:
        assert _lazy_import.vtkMatrix4x4 is backend.vtkMatrix4x4
        assert _lazy_import.vtkTransform is backend.vtkTransform
        assert isinstance(backend.vtkMatrix4x4(), _lazy_import.vtkMatrix4x4)
    finally:
        for name in ('vtkMatrix3x3', 'vtkMatrix4x4', 'vtkTransform'):
            _lazy_import.__dict__.pop(name, None)


def test_lazy_import_falls_back_to_backend_submodules(monkeypatch):
    """A backend laid out like stock VTK resolves through its submodules."""
    root = types.ModuleType('fake_stock_vtk')
    root.__path__ = []  # mark as a package so submodules resolve
    submodule = types.ModuleType('fake_stock_vtk.vtkCommonMath')
    submodule.vtkMatrix4x4 = type('vtkMatrix4x4', (), {})
    monkeypatch.setitem(sys.modules, 'fake_stock_vtk', root)
    monkeypatch.setitem(sys.modules, 'fake_stock_vtk.vtkCommonMath', submodule)
    monkeypatch.setenv('PYVISTA_VTK_BACKEND', 'fake_stock_vtk')
    _lazy_import.__dict__.pop('vtkMatrix4x4', None)
    try:
        assert _lazy_import.vtkMatrix4x4 is submodule.vtkMatrix4x4
    finally:
        _lazy_import.__dict__.pop('vtkMatrix4x4', None)


@needs_vtk
def test_lazy_import_backend_vtk_means_vtkmodules(monkeypatch):
    monkeypatch.setenv('PYVISTA_VTK_BACKEND', 'vtk')
    _lazy_import.__dict__.pop('vtkMatrix3x3', None)
    try:
        assert _lazy_import.vtkMatrix3x3 is vtkMatrix3x3
    finally:
        _lazy_import.__dict__.pop('vtkMatrix3x3', None)


def test_lazy_import_placeholder_when_unavailable(monkeypatch):
    """An unimportable name resolves to a placeholder with no instances."""
    monkeypatch.setitem(_lazy_import._SCIPY_MODULES, 'Missing', 'a_package_that_is_not_installed')
    try:
        placeholder = _lazy_import.Missing
        assert placeholder.__name__ == 'Missing'
        assert not isinstance(np.eye(4), placeholder)
    finally:
        _lazy_import.__dict__.pop('Missing', None)


# Edge cases: empty and degenerate input, and boundary values.


@pytest.mark.parametrize(
    'function', [validate_array, validate_arrayN], ids=['validate_array', 'validate_arrayN']
)
def test_empty_array_is_valid(function):
    assert function([]).shape == (0,)


@pytest.mark.parametrize('kwargs', [{'must_be_sorted': True}, {'must_be_finite': True}])
def test_empty_array_satisfies_elementwise_constraints(kwargs):
    # There is no element to violate them.
    assert validate_array(np.array([]), **kwargs).size == 0


@pytest.mark.parametrize('array', [[], [5]], ids=['empty', 'single'])
@pytest.mark.parametrize('strict', [True, False])
@pytest.mark.parametrize('ascending', [True, False])
def test_check_sorted_accepts_fewer_than_two_elements(array, ascending, strict):
    check_sorted(array, ascending=ascending, strict=strict)


def test_check_sorted_accepts_equal_values_when_not_strict():
    check_sorted([1, 1, 1], strict=False)


def test_check_sorted_rejects_equal_values_when_strict():
    with pytest.raises(ValueError, match=re.escape('must be sorted in strict ascending order')):
        check_sorted([1, 1, 1], strict=True)


def test_check_range_rejects_a_value_equal_to_a_strict_bound():
    with pytest.raises(ValueError, match=re.escape('Array values must all be greater than 2.')):
        check_range([2], [2, 2], strict_lower=True)


def test_check_integer_accepts_infinity():
    # The check compares against np.floor, which leaves infinity unchanged.
    check_integer(np.inf)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_check_finite_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match='must have finite values'):
        check_finite(value)


def test_validate_array_negative_zero_is_nonnegative():
    assert validate_array(-0.0, must_be_nonnegative=True) == 0.0


def test_validate_array_broadcasts_a_scalar():
    assert np.array_equal(validate_array(1, broadcast_to=(3,)), [1, 1, 1])


def test_validate_array_reshapes_to_the_requested_shape():
    assert validate_array([1, 2, 3], reshape_to=(3, 1)).shape == (3, 1)


def test_check_shape_accepts_any_of_several_shapes():
    check_shape([1, 2], [(2,), (3,)])


def test_check_shape_rejects_when_no_shape_matches():
    with pytest.raises(ValueError, match='Shape must be one of'):
        check_shape([1, 2], [(1,), (3,)])


def test_check_ndim_accepts_any_of_several_dimensions():
    check_ndim([[1]], [1, 2])


@pytest.mark.parametrize('array', [1, [1], [[1]]], ids=['0d', '1d', '2d'])
def test_check_ndim_rejects_a_wrong_number_of_dimensions(array):
    ndim = np.array(array).ndim
    match = f'Array has the incorrect number of dimensions. Got {ndim}, expected {ndim + 1}.'
    with pytest.raises(ValueError, match=re.escape(match)):
        check_ndim(array, ndim + 1)


def test_check_iterable_items_accepts_an_empty_iterable():
    check_iterable_items([], int)


@pytest.mark.parametrize('scalar_type', [float, int, bool])
def test_typing_aliases_are_subscriptable(scalar_type):
    from pyvista_validation import _typing

    aliases = (
        _typing.ArrayLike,
        _typing.VectorLike,
        _typing.MatrixLike,
        _typing._ArrayLikeOrScalar,
    )
    for alias in aliases:
        assert alias[scalar_type] != alias
    assert _typing.NumberType.__default__ is float
    assert _typing.NumpyArray[np.float32] != _typing.NumpyArray


@pytest.mark.parametrize(
    ('check', 'args', 'kwargs'),
    [
        (check_subdtype, (np.floating,), {}),
        (check_real, (), {}),
        (check_sorted, (), {}),
        (check_finite, (), {}),
        (check_integer, (), {}),
        (check_nonnegative, (), {}),
        (check_greater_than, (-1,), {}),
        (check_less_than, (10,), {}),
        (check_range, ([0, 10],), {}),
        (check_shape, (2,), {}),
        (check_ndim, (1,), {}),
        (check_sequence, (), {}),
        (check_iterable, (), {}),
        (check_instance, (list,), {}),
        (check_type, (list,), {}),
        (check_iterable_items, (float,), {}),
        (check_length, (), {'exact_length': 2}),
    ],
)
def test_check_functions_return_their_input(check, args, kwargs):
    array = [1.0, 2.0]
    assert check(array, *args, **kwargs) is array


def test_scalar_check_functions_return_their_input():
    number = 1.5
    assert check_number(number) is number
    assert check_string('abc') == 'abc'
    assert check_contains([1, 2], must_contain=2) == 2


# Behaviour the API documents that the tests above do not pin down. Where the implementation
# falls short of its documentation, the test is a strict xfail that says how.


@pytest.mark.parametrize(
    ('kwargs', 'invalid', 'error_type'),
    [
        ({'must_have_shape': (2,)}, [1, 2, 3], ValueError),
        ({'must_have_ndim': 2}, [1, 2, 3], ValueError),
        ({'must_have_dtype': np.floating}, [1, 2, 3], TypeError),
        ({'must_have_length': 2}, [1, 2, 3], ValueError),
        ({'must_have_min_length': 4}, [1, 2, 3], ValueError),
        ({'must_have_max_length': 2}, [1, 2, 3], ValueError),
        ({'must_be_nonnegative': True}, [-1, 2, 3], ValueError),
        ({'must_be_in_range': [0, 2]}, [1, 2, 3], ValueError),
        ({'must_be_real': True}, ['a'], TypeError),
    ],
    ids=[
        'shape',
        'ndim',
        'dtype',
        'length',
        'min-length',
        'max-length',
        'nonneg',
        'range',
        'real',
    ],
)
def test_validate_array_every_constraint_error_starts_with_the_name(kwargs, invalid, error_type):
    with pytest.raises(error_type, match=r'^_input '):
        validate_array(invalid, name='_input', **kwargs)


def test_validate_array_shape_is_checked_before_reshaping():
    match = 'Array has shape (3,) which is not allowed. Shape must be (3, 1).'
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_array([1, 2, 3], must_have_shape=(3, 1), reshape_to=(3, 1))
    assert validate_array([1, 2, 3], must_have_shape=3, reshape_to=(3, 1)).shape == (3, 1)


def test_validate_array_length_is_checked_after_reshaping():
    assert validate_array([[1, 2, 3]], reshape_to=(3,), must_have_length=3).shape == (3,)


def test_validate_array_length_is_checked_after_broadcasting():
    assert validate_array(1, broadcast_to=(3,), must_have_length=3).shape == (3,)


def test_validate_array3_broadcast_satisfies_a_length_constraint():
    assert validate_array3(1, broadcast=True, must_have_length=3).shape == (3,)


def test_validate_array_reshape_accepts_an_integer():
    assert validate_array([[1, 2], [3, 4]], reshape_to=4).shape == (4,)


def test_validate_array_broadcasts_after_reshaping():
    out = validate_array([1, 2, 3], reshape_to=(3, 1), broadcast_to=(3, 2))
    assert out.tolist() == [[1, 1], [2, 2], [3, 3]]


def test_validate_array_broadcast_returns_a_read_only_view():
    out = validate_array([1, 2, 3], broadcast_to=(2, 3))
    assert out.shape == (2, 3)
    assert not out.flags.writeable


@pytest.mark.parametrize(
    'kwargs', [{'reshape_to': (2, 2)}, {'broadcast_to': (2,)}], ids=['reshape', 'broadcast']
)
def test_validate_array_rejects_an_incompatible_new_shape(kwargs):
    with pytest.raises(ValueError, match=r'reshape|broadcast'):
        validate_array([1, 2, 3], **kwargs)


def test_validate_array_accepts_any_of_several_dtypes():
    validate_array([1, 2], must_have_dtype=[np.floating, np.integer])
    validate_array([1.5], must_have_dtype='float64')
    with pytest.raises(TypeError, match='must be a subtype of at least one of'):
        validate_array([1, 2], must_have_dtype=[np.floating, np.bool_])


def test_validate_array_accepts_any_of_several_ndims():
    validate_array([1], must_have_ndim=[1, 2])
    with pytest.raises(ValueError, match=re.escape('Got 1, expected one of [0, 2].')):
        validate_array([1], must_have_ndim=[0, 2])


def test_validate_array_strict_bounds_exclude_the_range_endpoints():
    validate_array([1, 2], must_be_in_range=[1, 2])
    with pytest.raises(ValueError, match=re.escape('Array values must all be greater than 1.')):
        validate_array([1, 2], must_be_in_range=[1, 2], strict_lower_bound=True)
    with pytest.raises(ValueError, match=re.escape('Array values must all be less than 2.')):
        validate_array([1, 2], must_be_in_range=[1, 2], strict_upper_bound=True)


def test_validate_array_rejects_an_unsorted_range():
    with pytest.raises(ValueError, match='Range with 2 elements must be sorted'):
        validate_array([1], must_be_in_range=[1, 0])


def test_validate_array_forwards_the_sorting_options():
    validate_array([2, 1], must_be_sorted={'ascending': False})
    validate_array([[1, 3], [2, 4]], must_be_sorted={'axis': 0})
    with pytest.raises(ValueError, match='must be sorted in strict ascending order'):
        validate_array([1, 1], must_be_sorted={'strict': True})
    with pytest.raises(ValueError, match='must be sorted in descending order'):
        validate_array([1, 2], must_be_sorted={'ascending': False})


def test_validate_array_checks_finite_before_casting():
    # An infinite value would otherwise be cast to an arbitrary integer
    with pytest.raises(ValueError, match='Array must have finite values'):
        validate_array(np.inf, must_be_finite=True, dtype_out=int)


def test_validate_array_dtype_out_applies_to_list_and_tuple_output():
    assert validate_array([1, 2], dtype_out=float, to_list=True) == [1.0, 2.0]
    out = validate_array([1, 2], dtype_out=float, to_tuple=True)
    assert out == (1.0, 2.0)
    assert all(type(item) is float for item in out)


def test_validate_array_nested_output_keeps_the_shape():
    assert validate_array([[1, 2], [3, 4]], to_list=True) == [[1, 2], [3, 4]]
    assert validate_array([[1, 2], [3, 4]], to_tuple=True) == ((1, 2), (3, 4))


def test_validate_array_min_length_above_max_length_is_an_error():
    with pytest.raises(ValueError, match='Range with 2 elements must be sorted'):
        validate_array([1], must_have_min_length=2, must_have_max_length=1)


def test_validate_array_accepts_a_range():
    assert validate_array(range(3)).tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    'array', [None, {1, 2}, [None], object()], ids=['none', 'set', 'list-of-none', 'object']
)
def test_validate_array_rejects_input_that_is_not_array_like(array):
    with pytest.raises(TypeError, match='Object arrays are not supported'):
        validate_array(array)


def test_validate_array_copy_does_not_alias_the_input():
    array = np.array([0.0, 1.0])
    out = validate_array(array, copy=True)
    out[0] = 5
    assert array[0] == 0.0


def test_validate_array_copy_with_as_any_false_does_not_share_memory():
    array = NdarraySubclass([0.0, 1.0])
    assert not np.shares_memory(validate_array(array, as_any=False, copy=True), array)


def test_validate_array_length_of_a_numpy_scalar():
    assert validate_array(np.int64(1), must_have_length=1) == 1


# --- validate_number ---


def test_validate_number_rejects_non_finite_values_by_default():
    with pytest.raises(ValueError, match='Number must have finite values'):
        validate_number(np.inf)
    assert validate_number(np.inf, must_be_finite=False) == np.inf


@pytest.mark.parametrize('num', [[[1]], [1, 2], np.ones((1, 1))], ids=['nested', 'two', '1x1'])
def test_validate_number_rejects_more_than_a_single_reshapeable_element(num):
    with pytest.raises(ValueError, match=re.escape('Shape must be one of [(), (1,)].')):
        validate_number(num)


def test_validate_number_without_reshape_rejects_a_1d_array():
    match = 'Number has shape (1,) which is not allowed. Shape must be ().'
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_number([1], reshape=False)


@pytest.mark.parametrize(
    ('num', 'expected_type'),
    [
        (1, int),
        (1.5, float),
        (np.float32(1.5), float),
        (np.int8(1), int),
        (np.array(2.5), float),
        ([2], int),
    ],
    ids=['int', 'float', 'float32', 'int8', '0d-array', 'list'],
)
def test_validate_number_returns_a_python_scalar(num, expected_type):
    out = validate_number(num)
    assert type(out) is expected_type
    assert out == np.asarray(num).item()


def test_validate_number_to_tuple_still_returns_a_scalar():
    out = validate_number(1, to_tuple=True)
    assert type(out) is int
    assert out == 1


def test_validate_number_forwards_constraints():
    with pytest.raises(ValueError, match='Number must have integer-like values'):
        validate_number(1.5, must_be_integer=True)
    with pytest.raises(ValueError, match='Number values must all be less than or equal to 1'):
        validate_number(2, must_be_in_range=[0, 1])


def test_validate_number_treats_a_boolean_as_not_real():
    with pytest.raises(TypeError, match='Number must have real numbers'):
        validate_number(True)
    assert validate_number(True, must_be_real=False) is True


# --- validate_data_range ---


def test_validate_data_range_accepts_equal_bounds():
    assert validate_data_range((1, 1)) == (1, 1)


def test_validate_data_range_rejects_a_wrong_number_of_bounds():
    match = 'Data Range has shape (3,) which is not allowed. Shape must be 2.'
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_data_range([0, 1, 2])


def test_validate_data_range_sorting_cannot_be_disabled():
    with pytest.raises(ValueError, match="Parameter 'must_be_sorted' cannot be set"):
        validate_data_range([0, 1], must_be_sorted=False)


def test_validate_data_range_forwards_constraints():
    with pytest.raises(ValueError, match='Data Range values must all be greater than or equal'):
        validate_data_range([-1, 1], must_be_nonnegative=True)
    out = validate_data_range([0, 1], dtype_out=float)
    assert out == (0.0, 1.0)
    assert all(type(bound) is float for bound in out)


@pytest.mark.xfail(
    strict=True, reason='an explicit to_list=False switches the output from a tuple to an array'
)
def test_validate_data_range_returns_a_tuple_when_to_list_is_false():
    assert type(validate_data_range([0, 1], to_list=False)) is tuple


# --- validate_arrayNx3 / validate_arrayN / validate_array3 ---


def test_validate_arrayNx3_rejects_a_flat_array_of_six():  # noqa: N802
    match = 'Array has shape (6,) which is not allowed. Shape must be one of [3, (-1, 3)].'
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_arrayNx3([1, 2, 3, 4, 5, 6])


def test_validate_arrayNx3_accepts_an_empty_nx3():  # noqa: N802
    assert validate_arrayNx3(np.zeros((0, 3))).shape == (0, 3)
    with pytest.raises(ValueError, match=re.escape('Array has shape (0,)')):
        validate_arrayNx3([])


def test_validate_arrayNx3_nested_output():  # noqa: N802
    assert validate_arrayNx3([1, 2, 3], to_list=True) == [[1, 2, 3]]
    assert validate_arrayNx3([[1, 2, 3], [4, 5, 6]], to_tuple=True) == ((1, 2, 3), (4, 5, 6))


def test_validate_arrayNx3_forwards_constraints():  # noqa: N802
    with pytest.raises(
        ValueError, match=re.escape('Array values must all be less than or equal to 10.')
    ):
        validate_arrayNx3(((1, 2, 3), (4, 5, 60)), must_be_in_range=[0, 10])


def test_validate_arrayNx3_reshape_cannot_be_overridden():  # noqa: N802
    match = "Parameter 'reshape_to' cannot be set for function `validate_arrayNx3`."
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_arrayNx3([1, 2, 3], reshape_to=(3,))


def test_validate_arrayN_rejects_a_column_vector():  # noqa: N802
    match = 'Array has shape (2, 1) which is not allowed. Shape must be one of [(), -1, (1, -1)].'
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_arrayN([[1], [2]])


def test_validate_arrayN_reshape_cannot_be_overridden():  # noqa: N802
    validate_arrayN(1, reshape_to=-1)  # the value it is set to anyway
    match = "Parameter 'reshape_to' cannot be set for function `validate_arrayN`."
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_arrayN(1, reshape_to=(1,))


def test_validate_arrayN_forwards_constraints():  # noqa: N802
    validate_arrayN((1, 2, 3), must_have_length=3)
    with pytest.raises(ValueError, match='Array must have a length equal to any of: 4'):
        validate_arrayN((1, 2, 3), must_have_length=4)


def test_validate_arrayN_output_modes():  # noqa: N802
    assert validate_arrayN([[1, 2]], to_list=True) == [1, 2]
    assert validate_arrayN(1, to_tuple=True) == (1,)


def test_validate_arrayN_unsigned_casts_whole_floats_to_integers():  # noqa: N802
    out = validate_arrayN_unsigned((1.0, 2.0))
    assert np.issubdtype(out.dtype, np.integer)
    assert out.tolist() == [1, 2]
    assert validate_arrayN_unsigned([1.0, 2.0], to_list=True) == [1, 2]


def test_validate_arrayN_unsigned_rejects_fractions():  # noqa: N802
    with pytest.raises(ValueError, match=re.escape('Array must have integer-like values.')):
        validate_arrayN_unsigned([1.5])


def test_validate_arrayN_unsigned_dtype_out_must_be_integral():  # noqa: N802
    assert validate_arrayN_unsigned([1], dtype_out=np.uint8).dtype == np.uint8
    with pytest.raises(TypeError, match=re.escape("must be a subtype of <class 'numpy.integer'>")):
        validate_arrayN_unsigned([1], dtype_out=float)


@pytest.mark.parametrize('kwarg', ['must_be_integer', 'must_be_nonnegative'])
def test_validate_arrayN_unsigned_constraints_cannot_be_disabled(kwarg):  # noqa: N802
    with pytest.raises(ValueError, match=f"Parameter '{kwarg}' cannot be set"):
        validate_arrayN_unsigned([1], **{kwarg: False})


@pytest.mark.xfail(strict=True, reason='values beyond int64 wrap around to negative integers')
def test_validate_arrayN_unsigned_output_is_never_negative():  # noqa: N802
    assert np.all(validate_arrayN_unsigned([2**63]) >= 0)


def test_validate_array3_broadcasts_a_single_element_vector():
    assert validate_array3([5], broadcast=True).tolist() == [5, 5, 5]


def test_validate_array3_without_broadcast_rejects_a_scalar():
    match = 'Array has shape () which is not allowed. Shape must be one of [(3,), (1, 3), (3, 1)].'
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_array3(5)


def test_validate_array3_broadcast_rejects_a_1x1_array():
    with pytest.raises(ValueError, match=re.escape('Array has shape (1, 1)')):
        validate_array3([[5]], broadcast=True)


def test_validate_array3_output_modes():
    assert validate_array3([[1, 2, 3]], to_list=True) == [1, 2, 3]
    assert validate_array3(1, broadcast=True, to_tuple=True) == (1, 1, 1)


def test_validate_array3_forwards_constraints():
    with pytest.raises(
        ValueError, match=re.escape('Array values must all be greater than or equal to 0.')
    ):
        validate_array3((1, -2, 3), must_be_nonnegative=True)


# --- validate_axes ---


def test_validate_axes_rejects_a_wrong_number_of_vectors():
    match = 'Axes arguments must have a length equal to any of: [1, 2, 3]. Got length 4 instead.'
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_axes([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1])


def test_validate_axes_single_argument_must_be_3x3():
    match = 'Axes has shape (2, 3) which is not allowed. Shape must be (3, 3).'
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_axes(np.eye(3)[:2])


def test_validate_axes_rejects_an_unknown_orientation():
    with pytest.raises(ValueError, match="Axes orientation 'up' is not valid"):
        validate_axes(np.eye(3), must_have_orientation='up')


def test_validate_axes_rejects_non_finite_values():
    with pytest.raises(ValueError, match=re.escape('Axes must have finite values.')):
        validate_axes(np.full((3, 3), np.nan))


def test_validate_axes_normalize_returns_unit_float_vectors():
    axes = validate_axes(np.diag([2, 3, 4]))
    assert axes.dtype == np.float64
    assert np.array_equal(axes, np.eye(3))
    assert validate_axes(np.eye(3, dtype=int), normalize=False).dtype == np.int64


@pytest.mark.parametrize(
    'axes',
    [
        ([2, 0, 0], [1, 0, 0], [0, 0, 1]),
        ([1, 0, 0], [0, 1, 0], [0, 1, 0]),
        ([1, 0, 0], [-1, 0, 0], [0, 0, 1]),
        ([1, 0, 0], [2, 0, 0]),
    ],
    ids=['scaled', 'rows-1-and-2', 'antiparallel', 'two-vectors'],
)
def test_validate_axes_rejects_parallel_vectors_in_any_position(axes):
    with pytest.raises(ValueError, match='cannot be parallel'):
        validate_axes(*axes, must_be_orthogonal=False)


# --- validate_rotation ---


def test_validate_rotation_rejects_an_unknown_handedness():
    with pytest.raises(ValueError, match="must_have_handedness 'up' is not valid"):
        validate_rotation(np.eye(3), must_have_handedness='up')


def test_validate_rotation_tolerance_bounds_the_orthogonality_check():
    nearly = np.eye(3)
    nearly[0, 0] += 1e-4
    validate_rotation(nearly, tolerance=1e-3)
    with pytest.raises(ValueError, match='Rotation must be orthogonal'):
        validate_rotation(nearly, tolerance=1e-6)


def test_validate_rotation_keeps_the_input_dtype():
    out = validate_rotation(np.eye(3, dtype=int))
    assert np.issubdtype(out.dtype, np.integer)
    assert np.array_equal(out, np.eye(3))


def test_validate_rotation_rejects_a_4x4():
    match = 'Rotation has shape (4, 4) which is not allowed. Shape must be (3, 3).'
    with pytest.raises(ValueError, match=re.escape(match)):
        validate_rotation(np.eye(4))


@needs_scipy
def test_validate_rotation_accepts_a_scipy_rotation():
    rotation = Rotation.from_euler('z', 90, degrees=True)
    out = validate_rotation(rotation, must_have_handedness='right')
    assert np.allclose(out, rotation.as_matrix())
    with pytest.raises(ValueError, match='Expected a left-handed rotation'):
        validate_rotation(rotation, must_have_handedness='left')


@needs_vtk
def test_validate_rotation_accepts_a_vtk_matrix():
    assert np.array_equal(validate_rotation(vtkMatrix3x3()), np.eye(3))


# --- validate_transform4x4 / validate_transform3x3 ---


@needs_vtk
def test_validate_transform4x4_reads_the_vtk_transform_matrix():
    transform = vtkTransform()
    transform.Translate(1, 2, 3)
    transform.RotateZ(90)
    expected = [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    assert np.allclose(validate_transform4x4(transform), expected)


@needs_vtk
def test_validate_transform_reads_vtk_matrix_values():
    matrix = vtkMatrix3x3()
    matrix.SetElement(0, 1, 5.0)
    assert validate_transform3x3(matrix)[0].tolist() == [1, 5, 0]
    assert validate_transform4x4(matrix)[0].tolist() == [1, 5, 0, 0]


@needs_scipy
def test_validate_transform_accepts_a_scipy_rotation():
    rotation = Rotation.from_euler('z', 90, degrees=True)
    assert np.allclose(validate_transform3x3(rotation), rotation.as_matrix())
    padded = validate_transform4x4(rotation)
    assert np.allclose(padded[:3, :3], rotation.as_matrix())
    assert padded[3].tolist() == [0, 0, 0, 1]
    assert padded[:3, 3].tolist() == [0, 0, 0]


def test_validate_transform4x4_pads_a_3x3_with_the_identity():
    matrix = np.arange(1.0, 10.0).reshape(3, 3)
    padded = validate_transform4x4(matrix)
    assert np.array_equal(padded[:3, :3], matrix)
    assert padded[3].tolist() == [0, 0, 0, 1]
    assert padded[:3, 3].tolist() == [0, 0, 0]
    assert validate_transform4x4(np.eye(3, dtype=int)).dtype == np.float64


def test_validate_transform4x4_keeps_a_4x4_dtype():
    assert np.issubdtype(validate_transform4x4(np.eye(4, dtype=int)).dtype, np.integer)


def test_validate_transform4x4_rejects_non_finite_values():
    with pytest.raises(ValueError, match=re.escape('Transform must have finite values.')):
        validate_transform4x4(np.full((4, 4), np.nan))
    assert np.isnan(validate_transform4x4(np.full((4, 4), np.nan), must_be_finite=False)).all()


def test_validate_transform3x3_reports_non_finite_values():
    with pytest.raises(ValueError, match=re.escape('Transform must have finite values.')):
        validate_transform3x3(np.full((3, 3), np.nan))


def test_validate_transform3x3_accepts_non_finite_values_when_allowed():
    assert np.isnan(validate_transform3x3(np.full((3, 3), np.nan), must_be_finite=False)).all()


def test_validate_transform_error_reports_the_name():
    with pytest.raises(ValueError, match=re.escape('_input has shape (2, 2)')):
        validate_transform4x4(np.eye(2), name='_input')
    with pytest.raises(ValueError, match='_input must have finite values'):
        validate_transform4x4(np.full((4, 4), np.nan), name='_input')


@pytest.mark.parametrize('function', [validate_transform3x3, validate_transform4x4])
def test_validate_transform_rejects_a_non_string_name(function):
    with pytest.raises(TypeError, match="Name must be an instance of <class 'str'>"):
        function(np.eye(3), name=1)


# --- validate_dimensionality ---


@pytest.mark.parametrize(
    'dimensionality',
    [2, 2.0, np.int64(2), '2D', [2], np.array(2)],
    ids=['int', 'float', 'int64', 'alias', 'list', '0d-array'],
)
def test_validate_dimensionality_returns_a_python_int(dimensionality):
    out = validate_dimensionality(dimensionality)
    assert type(out) is int
    assert out == 2


def test_validate_dimensionality_without_reshape_rejects_a_1d_array():
    with pytest.raises(ValueError, match=re.escape('Shape must be ().')):
        validate_dimensionality([1], reshape=False)


def test_validate_dimensionality_rejects_a_lowercase_alias():
    with pytest.raises(ValueError, match='`1d` is not a valid dimensionality'):
        validate_dimensionality('1d')


def test_validate_dimensionality_rejects_an_out_of_range_alias():
    with pytest.raises(ValueError, match='must all be less than or equal to 3'):
        validate_dimensionality('4D')


@pytest.mark.parametrize(
    ('dimensionality', 'match'),
    [
        (2.5, 'Dimensionality must have integer-like values'),
        (-0.5, 'Dimensionality must have integer-like values'),
        (np.nan, 'Dimensionality must have finite values'),
    ],
)
def test_validate_dimensionality_rejects_non_integers(dimensionality, match):
    with pytest.raises(ValueError, match=match):
        validate_dimensionality(dimensionality)


def test_validate_dimensionality_rejects_a_boolean():
    with pytest.raises(TypeError, match='Dimensionality must have real numbers'):
        validate_dimensionality(True)


# --- check functions ---


def test_check_subdtype_accepts_dtype_names_and_objects():
    check_subdtype('uint8', np.integer)
    check_subdtype(np.dtype('float32'), 'float32')
    with pytest.raises(TypeError, match="_input has incorrect dtype of 'uint8'"):
        check_subdtype('uint8', np.floating, name='_input')


@pytest.mark.parametrize('dtype', [np.uint8, np.int16, np.float16, np.longlong])
def test_check_real_accepts_every_integer_and_floating_dtype(dtype):
    array = np.zeros(2, dtype=dtype)
    assert check_real(array) is array


def test_check_finite_rejects_a_single_non_finite_element():
    with pytest.raises(ValueError, match='_input must have finite values'):
        check_finite([1, np.nan, 2], name='_input')


def test_check_integer_strict_requires_an_integer_dtype():
    check_integer(np.array([1, 2]), strict=True)
    with pytest.raises(TypeError, match='must be a subtype of'):
        check_integer([1.0], strict=True)


@pytest.mark.xfail(strict=True, reason='check_integer does not pass the name on to check_subdtype')
def test_check_integer_strict_error_reports_the_name():
    with pytest.raises(TypeError, match='_input has incorrect dtype'):
        check_integer([1.0], strict=True, name='_input')


def test_check_nonnegative_error_reports_the_name():
    with pytest.raises(ValueError, match='_input values must all be greater than or equal to 0'):
        check_nonnegative(-1, name='_input')


@pytest.mark.parametrize('check', [check_greater_than, check_less_than])
def test_check_comparison_rejects_an_invalid_value(check):
    with pytest.raises(ValueError, match=re.escape('Value has shape (1,) which is not allowed')):
        check([1], [1])
    with pytest.raises(TypeError, match=re.escape('Value must have real numbers.')):
        check([1], 'a')


def test_check_range_rejects_an_unsorted_range():
    with pytest.raises(ValueError, match='Range with 2 elements must be sorted in ascending'):
        check_range([1], rng=[1, 0])


@pytest.mark.parametrize('rng', [[1], [1, 2, 3]])
def test_check_range_rejects_a_range_without_two_elements(rng):
    with pytest.raises(ValueError, match=r'Range has shape .* Shape must be 2\.'):
        check_range([1], rng=rng)


def test_check_range_accepts_open_intervals():
    check_range([1e300], [0, np.inf])
    check_range([-1e300], [-np.inf, 0])


def test_check_shape_wildcard_matches_any_size():
    check_shape(np.zeros((4, 3)), (-1, 3))
    check_shape(np.zeros((0, 3)), (-1, 3))
    match = 'Array has shape (4, 3) which is not allowed. Shape must be (-1, 2).'
    with pytest.raises(ValueError, match=re.escape(match)):
        check_shape(np.zeros((4, 3)), (-1, 2))


@pytest.mark.parametrize(
    ('shape', 'error_type', 'match'),
    [
        ((1.5,), TypeError, 'All items of Shape must be an instance of'),
        ((-2,), ValueError, 'Shape values must all be greater than or equal to -1'),
        ('abc', TypeError, 'Shape must be an instance of any type'),
        (None, TypeError, '`None` is not a valid shape'),
    ],
    ids=['float', 'negative', 'string', 'none'],
)
def test_check_shape_rejects_an_invalid_shape(shape, error_type, match):
    with pytest.raises(error_type, match=match):
        check_shape([1], shape)


def test_check_ndim_rejects_a_non_integer_ndim():
    with pytest.raises(TypeError, match='must be a subtype of'):
        check_ndim([[1]], 1.5)


def test_check_sorted_validates_the_axis():
    check_sorted([[1, 2]], axis=0.0)
    with pytest.raises(ValueError, match=re.escape('Axis must have integer-like values.')):
        check_sorted([[1, 2]], axis=1.5)
    with pytest.raises(TypeError, match='Axis must be an instance of'):
        check_sorted([[1, 2]], axis='a')


@pytest.mark.parametrize('num', [np.float64(1), np.int8(1), True, 1 + 1j])
def test_check_number_accepts_every_number(num):
    assert check_number(num) is num


def test_check_number_rejects_a_string():
    with pytest.raises(
        TypeError, match=re.escape("must be an instance of <class 'numbers.Number'>")
    ):
        check_number('1')


def test_check_sequence_rejects_a_generator():
    with pytest.raises(
        TypeError, match=re.escape("must be an instance of <class 'collections.abc.Sequence'>")
    ):
        check_sequence(x for x in [1])


@pytest.mark.parametrize(
    ('obj', 'classinfo'),
    [(None, int | None), (1, Optional[int]), ('a', Union[int, str])],  # noqa: UP007, UP045
    ids=['pep604-none', 'optional', 'union'],
)
def test_check_instance_accepts_union_types(obj, classinfo):
    assert check_instance(obj, classinfo) is obj


def test_check_instance_rejects_when_no_type_in_the_tuple_matches():
    with pytest.raises(TypeError, match='Object must be an instance of any type'):
        check_instance(0.0, (int, str))


def test_check_instance_exact_type_from_a_tuple():
    check_instance(0, (int, str), allow_subclass=False)
    with pytest.raises(TypeError, match='must have one of the following types'):
        check_instance(True, (int, str), allow_subclass=False)


def test_check_iterable_items_rejects_a_non_iterable():
    match = "_input must be an instance of <class 'collections.abc.Iterable'>"
    with pytest.raises(TypeError, match=match):
        check_iterable_items(1, int, name='_input')


def test_check_iterable_items_exact_type():
    match = "All items of Iterable must have type <class 'int'>. Got <class 'bool'> instead."
    with pytest.raises(TypeError, match=re.escape(match)):
        check_iterable_items([True], int, allow_subclass=False)


def test_check_contains_qualifier_depends_on_the_container():
    with pytest.raises(ValueError, match=re.escape('must be one of: \n\t(1, 2)')):
        check_contains((1, 2), must_contain=3)
    with pytest.raises(ValueError, match=re.escape('must be in: \n\t{1, 2}')):
        check_contains({1, 2}, must_contain=3)


@pytest.mark.parametrize(
    'scalar', [1, 1.5, np.float64(1), np.int64(1), np.float32(1.0), np.True_, np.array(5)]
)
def test_check_length_allow_scalar_counts_a_scalar_as_length_one(scalar):
    check_length(scalar, allow_scalar=True, exact_length=1)
    with pytest.raises(ValueError, match='must have a length equal to any of: 2'):
        check_length(scalar, allow_scalar=True, exact_length=2)


def test_check_length_rejects_a_scalar_without_allow_scalar():
    with pytest.raises(TypeError, match='has no len'):
        check_length(1)


def test_check_length_must_be_1d_accepts_any_1d_length():
    check_length([1, 2, 3], must_be_1d=True)


def test_check_length_of_a_multidimensional_array_is_its_first_dimension():
    check_length(np.zeros((2, 5)), exact_length=2)
    with pytest.raises(ValueError, match='Got length 2 instead'):
        check_length(np.zeros((2, 5)), exact_length=5)


# --- optional dependencies ---

LAZY_NAMES = ('vtkMatrix3x3', 'vtkMatrix4x4', 'vtkTransform', 'Rotation')


@pytest.fixture
def unresolved_lazy_names():
    """Drop the cached optional names so that the next access resolves them again."""
    for name in LAZY_NAMES:
        _lazy_import.__dict__.pop(name, None)
    yield
    for name in LAZY_NAMES:
        _lazy_import.__dict__.pop(name, None)


def test_lazy_import_prefers_cvista_when_installed(monkeypatch):
    monkeypatch.delenv('PYVISTA_VTK_BACKEND', raising=False)
    monkeypatch.setattr(
        importlib.util, 'find_spec', lambda name: object() if name == 'cvista' else None
    )
    assert _lazy_import._vtk_root() == 'cvista'
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None)
    assert _lazy_import._vtk_root() == 'vtkmodules'


def test_lazy_import_placeholder_when_the_vtk_backend_is_missing(
    monkeypatch, unresolved_lazy_names
):
    monkeypatch.setenv('PYVISTA_VTK_BACKEND', 'a_backend_that_is_not_installed')
    placeholder = _lazy_import.vtkMatrix4x4
    assert placeholder.__name__ == 'vtkMatrix4x4'
    assert not isinstance(np.eye(4), placeholder)
    # Arrays still validate without VTK, and anything else gets the usual error
    assert np.array_equal(validate_transform4x4(np.eye(4)), np.eye(4))
    with pytest.raises(TypeError, match='Input transform must be one of'):
        validate_transform4x4('abc')


@pytest.mark.parametrize(
    'function', [validate_transform4x4, validate_transform3x3, validate_rotation]
)
def test_array_input_does_not_resolve_optional_dependencies(function, unresolved_lazy_names):
    function(np.eye(3))
    assert not any(name in vars(_lazy_import) for name in LAZY_NAMES)
