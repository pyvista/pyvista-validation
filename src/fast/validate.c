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

/* ---- The specialised validators ------------------------------------------------------- */

static const char *const FAMILY_NAMES[] = {
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
    "reshape",
    "broadcast",
};
enum { F_RESHAPE = A_COUNT, F_BROADCAST, F_COUNT };
static params FAMILY_PARAMS = {FAMILY_NAMES, NULL, F_COUNT, 1, 1};

/* The shape specs and defaults the families set, built once. */
static struct {
    PyObject *scalar_shapes;    /* [(), (1,)] */
    PyObject *scalar;           /* () */
    PyObject *two;              /* 2 */
    PyObject *nx3_shapes;       /* [3, (-1, 3)] */
    PyObject *nx3;              /* (-1, 3) */
    PyObject *n_shapes;         /* [(), -1, (1, -1)] */
    PyObject *flat;             /* -1 */
    PyObject *three_shapes[4];  /* by reshape + 2 * broadcast */
    PyObject *three;            /* (3,) */
    PyObject *dimensionalities; /* [0, 3] */
    PyObject *int_type;         /* int */
} spec;

static int build_specs(void)
{
    spec.scalar_shapes = Py_BuildValue("[()(i)]", 1);
    spec.scalar = PyTuple_New(0);
    spec.two = PyLong_FromLong(2);
    spec.nx3_shapes = Py_BuildValue("[i(ii)]", 3, -1, 3);
    spec.nx3 = Py_BuildValue("(ii)", -1, 3);
    spec.n_shapes = Py_BuildValue("[()i(ii)]", -1, 1, -1);
    spec.flat = PyLong_FromLong(-1);
    spec.three_shapes[0] = Py_BuildValue("[(i)]", 3);
    spec.three_shapes[1] = Py_BuildValue("[(i)(ii)(ii)]", 3, 1, 3, 3, 1);
    spec.three_shapes[2] = Py_BuildValue("[(i)()(i)]", 3, 1);
    spec.three_shapes[3] = Py_BuildValue("[(i)(ii)(ii)()(i)]", 3, 1, 3, 3, 1, 1);
    spec.three = Py_BuildValue("(i)", 3);
    spec.dimensionalities = Py_BuildValue("[ii]", 0, 3);
    spec.int_type = Py_NewRef((PyObject *)&PyLong_Type);
    for (int i = 0; i < 4; i++) {
        if (spec.three_shapes[i] == NULL) {
            return -1;
        }
    }
    return spec.scalar_shapes && spec.scalar && spec.two && spec.nx3_shapes && spec.nx3 &&
                   spec.n_shapes && spec.flat && spec.three && spec.dimensionalities
               ? 0
               : -1;
}

/* Bind a family call. The keys a family sets itself, and the flags it does not take, send the
 * call to Python, which accepts the default value or raises. Returns 0, or -1 to fall back. */
static int bind_family(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames, PyObject **a,
                       int takes_reshape, int takes_broadcast, const int *fixed, int nfixed)
{
    if (bind(&FAMILY_PARAMS, args, nargs, kwnames, a) < 0 || a[A_ARR] == NULL) {
        return -1;
    }
    if ((!takes_reshape && a[F_RESHAPE] != NULL) || (!takes_broadcast && a[F_BROADCAST] != NULL)) {
        return -1;
    }
    for (int i = 0; i < nfixed; i++) {
        if (a[fixed[i]] != NULL) {
            return -1;
        }
    }
    return 0;
}

static const int SHAPE_AND_RESHAPE[] = {A_SHAPE, A_RESHAPE};

/* validate_number and validate_dimensionality reshape a single element to a scalar. */
static int scalar_shape(PyObject **a)
{
    int reshape = truth(a[F_RESHAPE], 1);
    if (reshape < 0) {
        PyErr_Clear();
        return -1;
    }
    a[A_SHAPE] = reshape ? spec.scalar_shapes : spec.scalar;
    if (reshape) {
        a[A_RESHAPE] = spec.scalar;
    }
    return 0;
}

static PyObject *fast_validate_number(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[F_COUNT];
    if (bind_family(args, nargs, kwnames, a, 1, 0, SHAPE_AND_RESHAPE, 2) < 0 || scalar_shape(a) < 0) {
        RETURN_FALLBACK;
    }
    if (a[A_TO_LIST] == NULL) {
        a[A_TO_LIST] = Py_True;
    }
    if (a[A_FINITE] == NULL) {
        a[A_FINITE] = Py_True;
    }
    return array_core(a);
}

static const int SHAPE_AND_SORTED[] = {A_SHAPE, A_SORTED};

static PyObject *fast_validate_data_range(PyObject *const *args, Py_ssize_t nargs,
                                          PyObject *kwnames)
{
    PyObject *a[F_COUNT];
    if (bind_family(args, nargs, kwnames, a, 0, 0, SHAPE_AND_SORTED, 2) < 0) {
        RETURN_FALLBACK;
    }
    a[A_SHAPE] = spec.two;
    a[A_SORTED] = Py_True;
    int to_list_ = truth(a[A_TO_LIST], 0);
    if (to_list_ < 0) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    if (!to_list_ && a[A_TO_TUPLE] == NULL) {
        a[A_TO_TUPLE] = Py_True;
    }
    return array_core(a);
}

static PyObject *fast_validate_arrayNx3(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[F_COUNT];
    if (bind_family(args, nargs, kwnames, a, 1, 0, SHAPE_AND_RESHAPE, 2) < 0) {
        RETURN_FALLBACK;
    }
    int reshape = truth(a[F_RESHAPE], 1);
    if (reshape < 0) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    a[A_SHAPE] = reshape ? spec.nx3_shapes : spec.nx3;
    if (reshape) {
        a[A_RESHAPE] = spec.nx3;
    }
    return array_core(a);
}

/* The shape that validate_arrayN gives a call, shared with validate_arrayN_unsigned. */
static int flat_shape(PyObject **a)
{
    int reshape = truth(a[F_RESHAPE], 1);
    if (reshape < 0) {
        PyErr_Clear();
        return -1;
    }
    a[A_SHAPE] = reshape ? spec.n_shapes : spec.flat;
    if (reshape) {
        a[A_RESHAPE] = spec.flat;
    }
    return 0;
}

static PyObject *fast_validate_arrayN(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[F_COUNT];
    if (bind_family(args, nargs, kwnames, a, 1, 0, SHAPE_AND_RESHAPE, 2) < 0 || flat_shape(a) < 0) {
        RETURN_FALLBACK;
    }
    return array_core(a);
}

static const int UNSIGNED_FIXED[] = {A_SHAPE, A_RESHAPE, A_INTEGER, A_NONNEGATIVE, A_RANGE};

/* The largest value of an integer dtype as a Python int, or NULL for another dtype. */
static PyObject *integer_limit(PyObject *dtype)
{
    PyArray_Descr *descr = NULL;
    if (!PyArray_DescrConverter(dtype, &descr)) {
        PyErr_Clear();
        return NULL;
    }
    int typenum = descr->type_num;
    int bits = 8 * (int)PyDataType_ELSIZE(descr);
    Py_DECREF(descr);
    if (!PyTypeNum_ISINTEGER(typenum)) {
        return NULL;
    }
    npy_uint64 limit = PyTypeNum_ISUNSIGNED(typenum) ? (bits == 64 ? NPY_MAX_UINT64 : ((npy_uint64)1 << bits) - 1)
                                                     : ((npy_uint64)1 << (bits - 1)) - 1;
    return PyLong_FromUnsignedLongLong(limit);
}

static PyObject *fast_validate_arrayN_unsigned(PyObject *const *args, Py_ssize_t nargs,
                                               PyObject *kwnames)
{
    PyObject *a[F_COUNT];
    if (bind_family(args, nargs, kwnames, a, 1, 0, UNSIGNED_FIXED, 5) < 0 || flat_shape(a) < 0) {
        RETURN_FALLBACK;
    }
    if (a[A_DTYPE_OUT] == NULL) {
        a[A_DTYPE_OUT] = spec.int_type;
    }
    PyObject *limit = integer_limit(a[A_DTYPE_OUT]);
    if (limit == NULL) {
        RETURN_FALLBACK;
    }
    PyObject *range = Py_BuildValue("[iN]", 0, limit);
    if (range == NULL) {
        return NULL;
    }
    a[A_RANGE] = range;
    a[A_INTEGER] = Py_True;
    a[A_NONNEGATIVE] = Py_True;
    PyObject *result = array_core(a);
    Py_DECREF(range);
    return result;
}

static PyObject *fast_validate_array3(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[F_COUNT];
    if (bind_family(args, nargs, kwnames, a, 1, 1, NULL, 0) < 0) {
        RETURN_FALLBACK;
    }
    int reshape = truth(a[F_RESHAPE], 1), broadcast = truth(a[F_BROADCAST], 0);
    if (reshape < 0 || broadcast < 0) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    /* The keys the flags set: Python decides when the caller passes them too */
    if (a[A_SHAPE] != NULL || (reshape && a[A_RESHAPE] != NULL) || (broadcast && a[A_BROADCAST] != NULL)) {
        RETURN_FALLBACK;
    }
    a[A_SHAPE] = spec.three_shapes[reshape + 2 * broadcast];
    if (reshape) {
        a[A_RESHAPE] = spec.flat;
    }
    if (broadcast) {
        a[A_BROADCAST] = spec.three;
    }
    return array_core(a);
}

static const int DIMENSIONALITY_FIXED[] = {A_SHAPE, A_RESHAPE, A_INTEGER};

/* A dimensionality alias such as '2D' as its number; NULL for anything else. */
static PyObject *dimensionality_alias(PyObject *text)
{
    Py_ssize_t length = PyUnicode_GetLength(text);
    if (length < 1 || length > 2) {
        return NULL;
    }
    Py_UCS4 digit = PyUnicode_ReadChar(text, 0);
    if (digit < '0' || digit > '9' || (length == 2 && PyUnicode_ReadChar(text, 1) != 'D')) {
        return NULL;
    }
    return PyLong_FromLong(digit - '0');
}

static PyObject *fast_validate_dimensionality(PyObject *const *args, Py_ssize_t nargs,
                                              PyObject *kwnames)
{
    PyObject *a[F_COUNT];
    if (bind_family(args, nargs, kwnames, a, 1, 0, DIMENSIONALITY_FIXED, 3) < 0 || scalar_shape(a) < 0) {
        RETURN_FALLBACK;
    }
    PyObject *number = NULL;
    if (PyUnicode_Check(a[A_ARR])) {
        number = dimensionality_alias(a[A_ARR]);
        if (number == NULL) {
            RETURN_FALLBACK;
        }
        a[A_ARR] = number;
    }
    if (a[A_TO_LIST] == NULL) {
        a[A_TO_LIST] = Py_True;
    }
    if (a[A_FINITE] == NULL) {
        a[A_FINITE] = Py_True;
    }
    if (a[A_RANGE] == NULL) {
        a[A_RANGE] = spec.dimensionalities;
    }
    if (a[A_DTYPE_OUT] == NULL) {
        a[A_DTYPE_OUT] = spec.int_type;
    }
    a[A_INTEGER] = Py_True;
    PyObject *result = array_core(a);
    Py_XDECREF(number);
    return result;
}
