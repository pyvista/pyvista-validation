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
