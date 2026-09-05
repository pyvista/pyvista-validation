/* validate.c: the validate_* functions. */

static const char *const ARRAY_NAMES[] = {
    "arr",
    "must_have_shape",
    "must_have_ndim",
    "must_have_dtype",
    "must_have_length",
    "must_have_min_length",
    "must_have_max_length",
    "must_be_nonnegative",
    "must_be_finite",
    "must_be_real",
    "must_be_integer",
    "must_be_sorted",
    "must_be_in_range",
    "strict_lower_bound",
    "strict_upper_bound",
    "reshape_to",
    "broadcast_to",
    "dtype_out",
    "as_any",
    "copy",
    "to_list",
    "to_tuple",
    "name",
};
enum {
    A_ARR,
    A_SHAPE,
    A_NDIM,
    A_DTYPE,
    A_LENGTH,
    A_MIN_LENGTH,
    A_MAX_LENGTH,
    A_NONNEGATIVE,
    A_FINITE,
    A_REAL,
    A_INTEGER,
    A_SORTED,
    A_RANGE,
    A_STRICT_LOWER,
    A_STRICT_UPPER,
    A_RESHAPE,
    A_BROADCAST,
    A_DTYPE_OUT,
    A_AS_ANY,
    A_COPY,
    A_TO_LIST,
    A_TO_TUPLE,
    A_NAME,
    A_COUNT,
};
static params ARRAY_PARAMS = {ARRAY_NAMES, NULL, A_COUNT, 1, 1};

/* validate_array on bound arguments: a new reference, FALLBACK, or NULL. */
static PyObject *array_core(PyObject *const *a)
{
    int as_any = truth(a[A_AS_ANY], 1);
    int copy = truth(a[A_COPY], 0);
    int to_list_ = truth(a[A_TO_LIST], 0);
    int to_tuple_ = truth(a[A_TO_TUPLE], 0);
    int must_be_real = truth(a[A_REAL], 1);
    int nonnegative = truth(a[A_NONNEGATIVE], 0);
    int finite = truth(a[A_FINITE], 0);
    int integer = truth(a[A_INTEGER], 0);
    int sorted_ = truth(a[A_SORTED], 0);
    if ((as_any | copy | to_list_ | to_tuple_ | must_be_real | nonnegative | finite | integer |
         sorted_) < 0) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }

    PyObject *out = as_array(a[A_ARR], as_any, copy);
    if (out == FALLBACK) {
        return out;
    }
    PyArrayObject *array = (PyArrayObject *)out;
#define DECLINE          \
    do {                 \
        Py_DECREF(out);  \
        RETURN_FALLBACK; \
    } while (0)

    if (must_be_real && !is_real_type(PyArray_TYPE(array))) {
        DECLINE;
    }
    if (GIVEN(a[A_DTYPE]) && subdtype_ok(PyArray_DESCR(array), a[A_DTYPE]) != 1) {
        DECLINE;
    }
    if (GIVEN(a[A_SHAPE]) && shape_ok(array, a[A_SHAPE]) != 1) {
        DECLINE;
    }
    if (GIVEN(a[A_NDIM]) && number_in(PyArray_NDIM(array), a[A_NDIM], 0) != 1) {
        DECLINE;
    }

    /* Reshape after checking the shape, then broadcast */
    if (GIVEN(a[A_RESHAPE])) {
        npy_intp dims[MAXDIMS];
        int n = read_dims(a[A_RESHAPE], dims, -1);
        if (n < 0) {
            DECLINE;
        }
        int same = PyTuple_Check(a[A_RESHAPE]) && n == PyArray_NDIM(array);
        for (int i = 0; same && i < n; i++) {
            same = dims[i] == PyArray_DIM(array, i);
        }
        if (!same) {
            PyArray_Dims shape = {dims, n};
            PyObject *reshaped = PyArray_Newshape(array, &shape, NPY_CORDER);
            Py_DECREF(out);
            if (reshaped == NULL) {
                PyErr_Clear();
                RETURN_FALLBACK;
            }
            out = reshaped;
            array = (PyArrayObject *)out;
        }
    }
    if (GIVEN(a[A_BROADCAST])) {
        npy_intp dims[MAXDIMS];
        int n = read_dims(a[A_BROADCAST], dims, 0);
        if (n < 0) {
            DECLINE;
        }
        int same = PyTuple_Check(a[A_BROADCAST]) && n == PyArray_NDIM(array);
        for (int i = 0; same && i < n; i++) {
            same = dims[i] == PyArray_DIM(array, i);
        }
        if (!same) {
            PyObject *broadcast = PyObject_CallFunctionObjArgs(cache.np_broadcast_to, out,
                                                               a[A_BROADCAST], Py_True, NULL);
            Py_DECREF(out);
            if (broadcast == NULL) {
                PyErr_Clear();
                RETURN_FALLBACK;
            }
            out = broadcast;
            array = (PyArrayObject *)out;
        }
    }

    /* The length is that of the reshaped and broadcast array */
    if ((GIVEN(a[A_LENGTH]) || GIVEN(a[A_MIN_LENGTH]) || GIVEN(a[A_MAX_LENGTH])) &&
        length_ok(array, GIVEN(a[A_LENGTH]) ? a[A_LENGTH] : NULL,
                  GIVEN(a[A_MIN_LENGTH]) ? a[A_MIN_LENGTH] : NULL,
                  GIVEN(a[A_MAX_LENGTH]) ? a[A_MAX_LENGTH] : NULL) != 1) {
        DECLINE;
    }

    /* The element-wise checks, one pass */
    values_spec spec = {nonnegative, finite, integer, 0, 0};
    if (GIVEN(a[A_RANGE])) {
        int strict_low = truth(a[A_STRICT_LOWER], 0), strict_high = truth(a[A_STRICT_UPPER], 0);
        if (strict_low < 0 || strict_high < 0 || !range_bounds(a[A_RANGE], &spec.low, &spec.high)) {
            PyErr_Clear();
            DECLINE;
        }
        spec.low_set = spec.high_set = 1;
        spec.strict_low = strict_low;
        spec.strict_high = strict_high;
    }
    if ((spec.nonnegative || spec.finite || spec.integer || spec.low_set) &&
        values_ok(array, &spec) != 1) {
        DECLINE;
    }
    if (sorted_) {
        int ascending, strict;
        PyObject *axis;
        if (!read_sorted_spec(a[A_SORTED], &ascending, &strict, &axis) ||
            sorted_ok(array, ascending, strict, axis) != 1) {
            DECLINE;
        }
    }

    if (GIVEN(a[A_DTYPE_OUT])) {
        PyObject *cast = cast_to(array, a[A_DTYPE_OUT]);
        Py_DECREF(out);
        if (cast == FALLBACK) {
            return cast;
        }
        out = cast;
        array = (PyArrayObject *)out;
    }
    if (to_tuple_ || to_list_) {
        PyObject *list = to_list(array);
        Py_DECREF(out);
        if (list == NULL || !to_tuple_) {
            return list;
        }
        PyObject *tuple = to_tuple(list);
        Py_DECREF(list);
        return tuple;
    }
    return out;
#undef DECLINE
}

static PyObject *fast_validate_array(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[A_COUNT];
    if (bind(&ARRAY_PARAMS, args, nargs, kwnames, a) < 0 || a[A_ARR] == NULL) {
        RETURN_FALLBACK;
    }
    return array_core(a);
}
