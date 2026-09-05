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
static PyObject *THREE_BY_THREE, *TRANSFORM_SHAPES;

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
            /* A subclass goes through np.broadcast_to itself, with subok=True */
            PyObject *broadcast =
                PyArray_CheckExact(array)
                    ? broadcast_view(array, dims, n)
                    : PyObject_CallFunctionObjArgs(cache.np_broadcast_to, out, a[A_BROADCAST],
                                                   Py_True, NULL);
            Py_DECREF(out);
            if (broadcast == NULL || broadcast == FALLBACK) {
                PyErr_Clear();
                Py_XDECREF(broadcast);
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
    THREE_BY_THREE = Py_BuildValue("(ii)", 3, 3);
    TRANSFORM_SHAPES = Py_BuildValue("[(ii)(ii)]", 3, 3, 4, 4);
    if (THREE_BY_THREE == NULL || TRANSFORM_SHAPES == NULL) {
        return -1;
    }
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

/* ---- Transforms ------------------------------------------------------------------------ */

static const char *const TRANSFORM_NAMES[] = {"transform", "must_be_finite", "name"};
static params TRANSFORM4X4_PARAMS = {TRANSFORM_NAMES, NULL, 3, 1, 1};
static params TRANSFORM3X3_PARAMS = {TRANSFORM_NAMES, NULL, 3, 1, 1};

static PyObject *lazy_module;

/* A class from pyvista_validation._lazy_import, but only once that module has resolved it.
 * Reading the name off the module would run its __getattr__, which imports VTK or SciPy on a
 * call that may need neither; the first object of one of those types goes to Python instead. */
static PyObject *lazy_class(const char *name)
{
    if (lazy_module == NULL) {
        lazy_module = PyImport_ImportModule("pyvista_validation._lazy_import");
        if (lazy_module == NULL) {
            PyErr_Clear();
            return NULL;
        }
    }
    PyObject *namespace = PyModule_GetDict(lazy_module);
    if (namespace == NULL) {
        return NULL;
    }
    PyObject *cls = PyDict_GetItemString(namespace, name);
    return cls == NULL ? NULL : Py_NewRef(cls);
}

static int is_lazy_instance(PyObject *obj, const char *name)
{
    PyObject *cls = lazy_class(name);
    if (cls == NULL) {
        return 0;
    }
    int result = PyObject_IsInstance(obj, cls);
    Py_DECREF(cls);
    if (result < 0) {
        PyErr_Clear();
        return 0;
    }
    return result;
}

/* What NumPy turns into a numeric array without asking the object: Python only tries the
 * VTK and SciPy types once validate_array has rejected the input as not array-like. */
static int array_like(PyObject *obj)
{
    return PyArray_Check(obj) || PyList_Check(obj) || PyTuple_Check(obj) || PyFloat_Check(obj) ||
           PyLong_Check(obj) || PyArray_IsScalar(obj, Generic);
}

/* A VTK matrix as a float64 array, read in one call through GetData(). NULL to fall back. */
static PyObject *vtk_matrix(PyObject *matrix, int size)
{
    PyObject *data = PyObject_CallMethod(matrix, "GetData", NULL);
    if (data == NULL) {
        PyErr_Clear();
        return NULL;
    }
    if (!PyTuple_Check(data) || PyTuple_Size(data) != size * size) {
        Py_DECREF(data);
        return NULL;
    }
    npy_intp dims[2] = {size, size};
    PyObject *array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (array == NULL) {
        Py_DECREF(data);
        return NULL;
    }
    double *values = (double *)PyArray_DATA((PyArrayObject *)array);
    for (int i = 0; i < size * size; i++) {
        values[i] = PyFloat_AsDouble(PyTuple_GetItem(data, i));
    }
    Py_DECREF(data);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        Py_DECREF(array);
        return NULL;
    }
    return array;
}

/* validate_array with a shape constraint and the finite check, as the transforms call it. */
static PyObject *shaped_array(PyObject *arr, PyObject *shape, int must_be_finite)
{
    PyObject *a[A_COUNT] = {NULL};
    a[A_ARR] = arr;
    a[A_SHAPE] = shape;
    a[A_FINITE] = must_be_finite ? Py_True : Py_False;
    return array_core(a);
}

/* validate_transform3x3 on its input: a new reference, or FALLBACK. */
static PyObject *transform3x3(PyObject *transform, int must_be_finite)
{
    if (array_like(transform)) {
        return shaped_array(transform, THREE_BY_THREE, must_be_finite);
    }
    if (is_lazy_instance(transform, "vtkMatrix3x3")) {
        PyObject *array = vtk_matrix(transform, 3);
        if (array == NULL) {
            RETURN_FALLBACK;
        }
        return array;
    }
    if (is_lazy_instance(transform, "Rotation")) {
        PyObject *matrix = PyObject_CallMethod(transform, "as_matrix", NULL);
        if (matrix == NULL) {
            PyErr_Clear();
            RETURN_FALLBACK;
        }
        PyObject *result = transform3x3(matrix, must_be_finite);
        Py_DECREF(matrix);
        return result;
    }
    RETURN_FALLBACK;
}

/* A 3x3 array embedded in the 4x4 identity, as float64. */
static PyObject *pad_to_4x4(PyObject *matrix)
{
    PyObject *doubles = PyArray_FromArray((PyArrayObject *)matrix, PyArray_DescrFromType(NPY_DOUBLE),
                                          NPY_ARRAY_CARRAY_RO | NPY_ARRAY_FORCECAST);
    if (doubles == NULL) {
        return NULL;
    }
    npy_intp dims[2] = {4, 4};
    PyObject *padded = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    if (padded == NULL) {
        Py_DECREF(doubles);
        return NULL;
    }
    const double *source = (const double *)PyArray_DATA((PyArrayObject *)doubles);
    double *target = (double *)PyArray_DATA((PyArrayObject *)padded);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            target[4 * i + j] = source[3 * i + j];
        }
    }
    target[15] = 1.0;
    Py_DECREF(doubles);
    return padded;
}

static PyObject *fast_validate_transform3x3(PyObject *const *args, Py_ssize_t nargs,
                                            PyObject *kwnames)
{
    PyObject *a[3];
    if (bind(&TRANSFORM3X3_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL ||
        (a[2] != NULL && !PyUnicode_Check(a[2]))) {
        RETURN_FALLBACK;
    }
    int must_be_finite = truth(a[1], 1);
    if (must_be_finite < 0) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    return transform3x3(a[0], must_be_finite);
}

static PyObject *fast_validate_transform4x4(PyObject *const *args, Py_ssize_t nargs,
                                            PyObject *kwnames)
{
    PyObject *a[3];
    if (bind(&TRANSFORM4X4_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL ||
        (a[2] != NULL && !PyUnicode_Check(a[2]))) {
        RETURN_FALLBACK;
    }
    int must_be_finite = truth(a[1], 1);
    if (must_be_finite < 0) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    PyObject *transform = a[0];
    PyObject *matrix;
    if (array_like(transform)) {
        matrix = shaped_array(transform, TRANSFORM_SHAPES, must_be_finite);
    }
    else if (is_lazy_instance(transform, "vtkMatrix4x4")) {
        matrix = vtk_matrix(transform, 4);
        if (matrix == NULL) {
            RETURN_FALLBACK;
        }
    }
    else if (is_lazy_instance(transform, "vtkTransform")) {
        PyObject *inner = PyObject_CallMethod(transform, "GetMatrix", NULL);
        matrix = inner == NULL ? NULL : vtk_matrix(inner, 4);
        Py_XDECREF(inner);
        if (matrix == NULL) {
            PyErr_Clear();
            RETURN_FALLBACK;
        }
    }
    else {
        matrix = transform3x3(transform, must_be_finite);
    }
    if (matrix == FALLBACK || matrix == NULL) {
        return matrix;
    }
    if (PyArray_NDIM((PyArrayObject *)matrix) == 2 && PyArray_DIM((PyArrayObject *)matrix, 0) == 3) {
        PyObject *padded = pad_to_4x4(matrix);
        Py_DECREF(matrix);
        return padded;
    }
    return matrix;
}

/* ---- Axes and rotations ---------------------------------------------------------------- */

static const char *const AXES_NAMES[] = {
    "normalize", "must_be_orthogonal", "must_have_orientation", "name",
};
static params AXES_PARAMS = {AXES_NAMES, NULL, 4, 0, 0};
static const char *const ROTATION_NAMES[] = {"rotation", "must_have_handedness", "tolerance", "name"};
static params ROTATION_PARAMS = {ROTATION_NAMES, NULL, 4, 2, 0};

/* np.isclose with its default tolerances. */
static inline int isclose(double a, double b)
{
    return fabs(a - b) <= 1e-8 + 1e-5 * fabs(b);
}

/* np.allclose of two 3-vectors. */
static int allclose3(const double *a, const double *b)
{
    return isclose(a[0], b[0]) && isclose(a[1], b[1]) && isclose(a[2], b[2]);
}

static void cross3(const double *a, const double *b, double *out)
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static double dot3(const double *a, const double *b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/* The values of a real array as doubles, in C order. 0 when there are not `count` of them. */
static int read_doubles(PyObject *array, double *out, int count)
{
    PyObject *doubles = PyArray_FromArray((PyArrayObject *)array, PyArray_DescrFromType(NPY_DOUBLE),
                                          NPY_ARRAY_CARRAY_RO | NPY_ARRAY_FORCECAST);
    if (doubles == NULL) {
        PyErr_Clear();
        return 0;
    }
    int ok = PyArray_SIZE((PyArrayObject *)doubles) == count;
    if (ok) {
        memcpy(out, PyArray_DATA((PyArrayObject *)doubles), count * sizeof(double));
    }
    Py_DECREF(doubles);
    return ok;
}

/* 'right' as 1, 'left' as -1, None as 0, and -2 for anything else. */
static int handedness(PyObject *obj, int missing)
{
    if (obj == NULL) {
        return missing;
    }
    if (obj == Py_None) {
        return 0;
    }
    if (PyUnicode_Check(obj)) {
        if (PyUnicode_CompareWithASCIIString(obj, "right") == 0) {
            return 1;
        }
        if (PyUnicode_CompareWithASCIIString(obj, "left") == 0) {
            return -1;
        }
    }
    return -2;
}

/* A fresh float64 3x3 array holding `values`. */
static PyObject *matrix_3x3(const double *values)
{
    npy_intp dims[2] = {3, 3};
    PyObject *array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (array != NULL) {
        memcpy(PyArray_DATA((PyArrayObject *)array), values, 9 * sizeof(double));
    }
    return array;
}

static PyObject *fast_validate_axes(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[4];
    if (nargs < 1 || nargs > 3 || bind(&AXES_PARAMS, args + nargs, 0, kwnames, a) < 0) {
        RETURN_FALLBACK;
    }
    int normalize = truth(a[0], 1), orthogonal = truth(a[1], 1);
    int orientation = handedness(a[2], 1);
    if (normalize < 0 || orthogonal < 0 || orientation == -2 || (orientation == 0 && nargs == 2)) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }

    PyObject *axes_array;
    if (nargs == 1) {
        axes_array = shaped_array(args[0], THREE_BY_THREE, 0);
        if (axes_array == FALLBACK) {
            return axes_array;
        }
    }
    else {
        /* Each vector through validate_array3, assigned into a float64 row */
        npy_intp dims[2] = {3, 3};
        axes_array = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
        if (axes_array == NULL) {
            return NULL;
        }
        double *vectors = (double *)PyArray_DATA((PyArrayObject *)axes_array);
        for (Py_ssize_t i = 0; i < nargs; i++) {
            PyObject *b[A_COUNT] = {NULL};
            b[A_ARR] = args[i];
            b[A_SHAPE] = spec.three_shapes[1];
            b[A_RESHAPE] = spec.flat;
            PyObject *row = array_core(b);
            int ok = row != FALLBACK && row != NULL && read_doubles(row, vectors + 3 * i, 3);
            Py_XDECREF(row);
            if (!ok) {
                Py_DECREF(axes_array);
                RETURN_FALLBACK;
            }
        }
        if (nargs == 2) {
            double third[3];
            cross3(vectors, vectors + 3, third);
            int zero_third = isclose(third[0], 0) && isclose(third[1], 0) && isclose(third[2], 0);
            int zero_first = isclose(vectors[0], 0) && isclose(vectors[1], 0) && isclose(vectors[2], 0);
            int zero_second = isclose(vectors[3], 0) && isclose(vectors[4], 0) && isclose(vectors[5], 0);
            if (zero_third && !zero_first && !zero_second) {
                Py_DECREF(axes_array);
                RETURN_FALLBACK;
            }
            if (orientation == 1) {
                memcpy(vectors + 6, third, sizeof third);
            }
            else {
                cross3(vectors + 3, vectors, vectors + 6);
            }
        }
    }

    values_spec finite = {0, 1, 0, 0, 0};
    double m[9];
    if (values_ok((PyArrayObject *)axes_array, &finite) != 1 || !read_doubles(axes_array, m, 9)) {
        Py_DECREF(axes_array);
        RETURN_FALLBACK;
    }
    double norms[3], n[9];
    for (int i = 0; i < 3; i++) {
        const double *row = m + 3 * i;
        if (isclose(row[0], 0) && isclose(row[1], 0) && isclose(row[2], 0)) {
            Py_DECREF(axes_array);
            RETURN_FALLBACK;
        }
        norms[i] = sqrt(row[0] * row[0] + row[1] * row[1] + row[2] * row[2]);
        for (int j = 0; j < 3; j++) {
            n[3 * i + j] = row[j] / norms[i];
        }
    }
    const double *n0 = n, *n1 = n + 3, *n2 = n + 6;
    if (isclose(fabs(dot3(n0, n1)), 1) || isclose(fabs(dot3(n0, n2)), 1) ||
        isclose(fabs(dot3(n1, n2)), 1)) {
        Py_DECREF(axes_array);
        RETURN_FALLBACK;
    }
    double cross01[3], cross12[3], minus[3];
    cross3(n0, n1, cross01);
    cross3(n1, n2, cross12);
    for (int j = 0; j < 3; j++) {
        minus[j] = -n2[j];
    }
    int is_orthogonal = (allclose3(cross01, n2) || allclose3(cross01, minus));
    for (int j = 0; j < 3; j++) {
        minus[j] = -n0[j];
    }
    is_orthogonal = is_orthogonal && (allclose3(cross12, n0) || allclose3(cross12, minus));
    if (orthogonal && !is_orthogonal) {
        Py_DECREF(axes_array);
        RETURN_FALLBACK;
    }
    if (orientation != 0) {
        double dot = dot3(cross01, n2);
        if ((orientation == 1 && dot < 0) || (orientation == -1 && dot > 0)) {
            Py_DECREF(axes_array);
            RETURN_FALLBACK;
        }
    }
    if (!normalize) {
        return axes_array;
    }
    Py_DECREF(axes_array);
    return matrix_3x3(n);
}

static PyObject *fast_validate_rotation(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[4];
    if (bind(&ROTATION_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL ||
        (a[3] != NULL && !PyUnicode_Check(a[3]))) {
        RETURN_FALLBACK;
    }
    int hand = handedness(a[1], 0);
    double tolerance = 1e-6;
    if (hand == -2 || (a[2] != NULL && !as_double(a[2], &tolerance))) {
        RETURN_FALLBACK;
    }
    PyObject *matrix = transform3x3(a[0], 1);
    if (matrix == FALLBACK || matrix == NULL) {
        return matrix;
    }
    double m[9];
    if (!read_doubles(matrix, m, 9)) {
        Py_DECREF(matrix);
        RETURN_FALLBACK;
    }
    /* The Frobenius norm of M Mᵀ - I */
    double sum = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double entry = dot3(m + 3 * i, m + 3 * j) - (i == j);
            sum += entry * entry;
        }
    }
    if (!(sqrt(sum) < tolerance)) {
        Py_DECREF(matrix);
        RETURN_FALLBACK;
    }
    if (hand != 0) {
        double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                     m[2] * (m[3] * m[7] - m[4] * m[6]);
        if ((hand == 1 && !(det > 0)) || (hand == -1 && !(det < 0))) {
            Py_DECREF(matrix);
            RETURN_FALLBACK;
        }
    }
    return matrix;
}
