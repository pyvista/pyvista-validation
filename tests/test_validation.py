"""Tests for the input validation functions."""

from __future__ import annotations

import itertools
import re
import sys
import types
from typing import NamedTuple

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
    match = (
        'Input transform must be one of:'
        '\n\tvtkMatrix3x3'
        '\n\t3x3 np.ndarray'
        '\n\tscipy.spatial.transform.Rotation'
        "\nGot array([1, 2, 3]) with type <class 'numpy.ndarray'> instead."
    )
    with pytest.raises(TypeError, match=re.escape(match)):
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


def test_validate_array_without_as_any_returns_a_base_ndarray():
    array = as_representation([0, 1], 'subclass')
    assert type(validate_array(array, as_any=False)) is np.ndarray


@pytest.mark.parametrize('array', [(True,), 'abc'])
def test_validate_array_non_numeric(array):
    match = 'Array must have real numbers.'
    with pytest.raises(TypeError, match=match):
        assert validate_array(array)
    assert validate_array(array, must_be_real=False)


def test_check_instance_accepts_a_matching_instance():
    check_instance(0, int)
    check_instance(0.0, (int, float))


def test_check_instance_accepts_a_subclass():
    check_instance(True, int)  # bool subclasses int


@pytest.mark.parametrize('name', ['_input', '_object'])
def test_check_instance_rejects_a_non_instance(name):
    with pytest.raises(TypeError, match='Object must be an instance of'):
        check_instance('0', int)
    with pytest.raises(TypeError, match=f'{name} must be an instance of'):
        check_instance('0', int, name=name)


def test_check_instance_rejects_a_list_of_types():
    with pytest.raises(TypeError, match='must be a type'):
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


@pytest.mark.parametrize('name', ['_input', '_object'])
def test_check_type_rejects_a_wrong_type(name):
    with pytest.raises(TypeError, match='Object must have type'):
        check_type('0', int)
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
        "Input has incorrect dtype of 'float64'. The dtype must be a "
        "subtype of <class 'numpy.integer'>."
    )
    with pytest.raises(TypeError, match=match):
        check_integer([2, 3.0], strict=True, name='_input')
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
    """Yield every axis that is in bounds for ``shape``, including ``None``."""
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
    with pytest.raises(ValueError, match='must be sorted in strict ascending order'):
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
    with pytest.raises(ValueError, match=r'Axes cannot be parallel.'):
        validate_axes([[0, 1, 0], [1, 0, 0], [0, 1, 0]])


@pytest.mark.parametrize('zero_row', [0, 1, 2])
def test_validate_axes_rejects_a_zero_vector(zero_row):
    axes = np.eye(3)
    axes[zero_row] = 0
    with pytest.raises(ValueError, match=r'Axes cannot be zeros.'):
        validate_axes(axes)


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
        assert array_out is not array_in

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


# Edge cases: empty and degenerate input, and boundary values, which the
# per-function tests above do not exercise.


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
    with pytest.raises(ValueError, match='must be sorted in strict ascending order'):
        check_sorted([1, 1, 1], strict=True)


def test_check_range_rejects_a_value_equal_to_a_strict_bound():
    with pytest.raises(ValueError, match=re.escape('Array values must all be greater than 2.')):
        check_range([2], [2, 2], strict_lower=True)


@pytest.mark.parametrize('value', [1, 1.0, -3.0, -0.0])
def test_check_integer_accepts_integer_valued_floats(value):
    check_integer(value)


def test_check_integer_accepts_infinity():
    # np.round(inf) == inf, so infinity counts as integer-like.
    check_integer(np.inf)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_check_finite_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match='must have finite values'):
        check_finite(value)


def test_check_greater_than_accepts_an_equal_value_when_not_strict():
    check_greater_than(1, 1, strict=False)


def test_check_greater_than_rejects_an_equal_value_when_strict():
    with pytest.raises(ValueError, match='must all be greater than 1'):
        check_greater_than(1, 1, strict=True)


def test_validate_array_rejects_a_bool_as_non_real():
    with pytest.raises(TypeError, match=re.escape('Array must have real numbers.')):
        validate_array(True, must_be_real=True)


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
