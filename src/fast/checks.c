/* checks.c: the check_* functions. */

static const char *const ARRAY_NAME_NAMES[] = {"array", "name"};
static params FINITE_PARAMS = {ARRAY_NAME_NAMES, NULL, 2, 1, 1};
static params NONNEGATIVE_PARAMS = {ARRAY_NAME_NAMES, NULL, 2, 1, 1};
static const char *const INTEGER_NAMES[] = {"array", "strict", "name"};
static params INTEGER_PARAMS = {INTEGER_NAMES, NULL, 3, 1, 1};
static const char *const COMPARE_NAMES[] = {"array", "value", "strict", "name"};
static params GREATER_PARAMS = {COMPARE_NAMES, NULL, 4, 2, 1};
static params LESS_PARAMS = {COMPARE_NAMES, NULL, 4, 2, 1};
static const char *const RANGE_NAMES[] = {"array", "rng", "strict_lower", "strict_upper", "name"};
static params RANGE_PARAMS = {RANGE_NAMES, NULL, 5, 2, 1};
static const char *const SORTED_NAMES[] = {"array", "ascending", "strict", "axis", "name"};
static params SORTED_PARAMS = {SORTED_NAMES, NULL, 5, 1, 1};

/* The input itself when every element passes the spec, else FALLBACK. */
static PyObject *checked(PyObject *input, const values_spec *spec)
{
    PyObject *array = any_array(input);
    if (array == FALLBACK) {
        return array;
    }
    int ok = values_ok((PyArrayObject *)array, spec);
    Py_DECREF(array);
    if (ok != 1) {
        RETURN_FALLBACK;
    }
    return Py_NewRef(input);
}

static PyObject *fast_check_finite(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[2];
    if (bind(&FINITE_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL) {
        RETURN_FALLBACK;
    }
    values_spec spec = {0, 1, 0, 0, 0};
    return checked(a[0], &spec);
}

static PyObject *fast_check_nonnegative(PyObject *const *args, Py_ssize_t nargs,
                                        PyObject *kwnames)
{
    PyObject *a[2];
    if (bind(&NONNEGATIVE_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL) {
        RETURN_FALLBACK;
    }
    values_spec spec = {1, 0, 0, 0, 0};
    return checked(a[0], &spec);
}

static PyObject *fast_check_integer(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[3];
    if (bind(&INTEGER_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL) {
        RETURN_FALLBACK;
    }
    int strict = truth(a[1], 0);
    if (strict < 0) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    if (strict) {
        /* The dtype itself must be integral */
        PyObject *array = any_array(a[0]);
        if (array == FALLBACK) {
            return array;
        }
        int integral = PyTypeNum_ISINTEGER(PyArray_TYPE((PyArrayObject *)array));
        Py_DECREF(array);
        if (!integral) {
            RETURN_FALLBACK;
        }
        return Py_NewRef(a[0]);
    }
    values_spec spec = {0, 0, 1, 0, 0};
    return checked(a[0], &spec);
}

/* check_greater_than and check_less_than: one bound, strict by default. */
static PyObject *compare(const params *spec_params, int is_low, PyObject *const *args,
                         Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[4];
    if (bind(spec_params, args, nargs, kwnames, a) < 0 || a[0] == NULL || a[1] == NULL) {
        RETURN_FALLBACK;
    }
    int strict = truth(a[2], 1);
    if (strict < 0) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    values_spec spec = {0, 0, 0, 0, 0};
    bound *b = is_low ? &spec.low : &spec.high;
    if (!scalar_bound(a[1], b)) {
        RETURN_FALLBACK;
    }
    if (is_low) {
        spec.low_set = 1;
        spec.strict_low = strict;
    }
    else {
        spec.high_set = 1;
        spec.strict_high = strict;
    }
    return checked(a[0], &spec);
}

static PyObject *fast_check_greater_than(PyObject *const *args, Py_ssize_t nargs,
                                         PyObject *kwnames)
{
    return compare(&GREATER_PARAMS, 1, args, nargs, kwnames);
}

static PyObject *fast_check_less_than(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    return compare(&LESS_PARAMS, 0, args, nargs, kwnames);
}

static PyObject *fast_check_range(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[5];
    if (bind(&RANGE_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL || a[1] == NULL) {
        RETURN_FALLBACK;
    }
    values_spec spec = {0, 0, 0, 1, 1};
    spec.strict_low = truth(a[2], 0);
    spec.strict_high = truth(a[3], 0);
    if (spec.strict_low < 0 || spec.strict_high < 0 || !range_bounds(a[1], &spec.low, &spec.high)) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    return checked(a[0], &spec);
}

static PyObject *fast_check_sorted(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[5];
    if (bind(&SORTED_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL) {
        RETURN_FALLBACK;
    }
    int ascending = truth(a[1], 1), strict = truth(a[2], 0);
    if (ascending < 0 || strict < 0) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    PyObject *array = any_array(a[0]);
    if (array == FALLBACK) {
        return array;
    }
    int ok = sorted_ok((PyArrayObject *)array, ascending, strict, a[3]);
    Py_DECREF(array);
    if (ok != 1) {
        RETURN_FALLBACK;
    }
    return Py_NewRef(a[0]);
}

/* ---- Structure ------------------------------------------------------------------------- */

static const char *const SUBDTYPE_NAMES[] = {"input_obj", "base_dtype", "name"};
static params SUBDTYPE_PARAMS = {SUBDTYPE_NAMES, NULL, 3, 2, 1};
static params REAL_PARAMS = {ARRAY_NAME_NAMES, NULL, 2, 1, 1};
static const char *const SHAPE_NAMES[] = {"array", "shape", "name"};
static params SHAPE_PARAMS = {SHAPE_NAMES, NULL, 3, 2, 1};
static const char *const NDIM_NAMES[] = {"array", "ndim", "name"};
static params NDIM_PARAMS = {NDIM_NAMES, NULL, 3, 2, 1};
static const char *const LENGTH_NAMES[] = {
    "sized_input", "exact_length", "min_length", "max_length", "must_be_1d", "allow_scalar", "name",
};
static params LENGTH_PARAMS = {LENGTH_NAMES, NULL, 7, 2, 1};

static PyObject *fast_check_subdtype(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[3];
    if (bind(&SUBDTYPE_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL || a[1] == NULL) {
        RETURN_FALLBACK;
    }
    PyArray_Descr *descr = dtype_of(a[0]);
    if (descr == NULL) {
        RETURN_FALLBACK;
    }
    int ok = subdtype_ok(descr, a[1]);
    Py_DECREF(descr);
    if (ok != 1) {
        RETURN_FALLBACK;
    }
    return Py_NewRef(a[0]);
}

static PyObject *fast_check_real(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[2];
    if (bind(&REAL_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL) {
        RETURN_FALLBACK;
    }
    PyObject *array = any_array(a[0]);
    if (array == FALLBACK) {
        return array;
    }
    int real = is_real_type(PyArray_TYPE((PyArrayObject *)array));
    Py_DECREF(array);
    if (!real) {
        RETURN_FALLBACK;
    }
    return Py_NewRef(a[0]);
}

static PyObject *fast_check_shape(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[3];
    if (bind(&SHAPE_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL || a[1] == NULL) {
        RETURN_FALLBACK;
    }
    PyObject *array = any_array(a[0]);
    if (array == FALLBACK) {
        return array;
    }
    int ok = shape_ok((PyArrayObject *)array, a[1]);
    Py_DECREF(array);
    if (ok != 1) {
        RETURN_FALLBACK;
    }
    return Py_NewRef(a[0]);
}

static PyObject *fast_check_ndim(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[3];
    if (bind(&NDIM_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL || a[1] == NULL) {
        RETURN_FALLBACK;
    }
    PyObject *array = any_array(a[0]);
    if (array == FALLBACK) {
        return array;
    }
    int ok = number_in(PyArray_NDIM((PyArrayObject *)array), a[1], 0);
    Py_DECREF(array);
    if (ok != 1) {
        RETURN_FALLBACK;
    }
    return Py_NewRef(a[0]);
}

static PyObject *fast_check_length(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[7];
    if (bind(&LENGTH_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL) {
        RETURN_FALLBACK;
    }
    int must_be_1d = truth(a[4], 0), allow_scalar = truth(a[5], 0);
    if (must_be_1d < 0 || allow_scalar < 0) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    PyObject *sized = a[0];
    npy_intp length;
    int ndim;
    int scalar = PyFloat_Check(sized) || PyLong_Check(sized) || PyArray_IsScalar(sized, Number) ||
                 PyArray_IsScalar(sized, Bool);
    if (allow_scalar && scalar) {
        length = 1;
        ndim = 1;
    }
    else if (PyArray_Check(sized)) {
        PyArrayObject *array = (PyArrayObject *)sized;
        if (PyArray_NDIM(array) == 0) {
            if (!allow_scalar) {
                RETURN_FALLBACK;
            }
            length = 1;
            ndim = 1;
        }
        else {
            length = PyArray_DIM(array, 0);
            ndim = PyArray_NDIM(array);
        }
    }
    else {
        length = PyObject_Size(sized);
        if (length < 0) {
            PyErr_Clear();
            RETURN_FALLBACK;
        }
        ndim = 1;
        if (must_be_1d) {
            /* np.shape of a sequence is that of the array it becomes */
            PyObject *array = any_array(sized);
            if (array == FALLBACK) {
                return array;
            }
            ndim = PyArray_NDIM((PyArrayObject *)array);
            Py_DECREF(array);
        }
    }
    if (must_be_1d && ndim != 1) {
        RETURN_FALLBACK;
    }
    PyObject *exact = GIVEN(a[1]) ? a[1] : NULL;
    PyObject *minimum = GIVEN(a[2]) ? a[2] : NULL;
    PyObject *maximum = GIVEN(a[3]) ? a[3] : NULL;
    if (length_checks(length, exact, minimum, maximum) != 1) {
        RETURN_FALLBACK;
    }
    return Py_NewRef(a[0]);
}
