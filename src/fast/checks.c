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

/* ---- Objects ---------------------------------------------------------------------------- */

static const char *const NUMBER_NAMES[] = {"num", "name"};
static params NUMBER_PARAMS = {NUMBER_NAMES, NULL, 2, 1, 1};
static const char *const STRING_NAMES[] = {"obj", "allow_subclass", "name"};
static params STRING_PARAMS = {STRING_NAMES, NULL, 3, 1, 1};
static const char *const OBJECT_NAMES[] = {"obj", "name"};
static params SEQUENCE_PARAMS = {OBJECT_NAMES, NULL, 2, 1, 1};
static params ITERABLE_PARAMS = {OBJECT_NAMES, NULL, 2, 1, 1};
static const char *const INSTANCE_NAMES[] = {"obj", "classinfo", "allow_subclass", "name"};
static params INSTANCE_PARAMS = {INSTANCE_NAMES, NULL, 4, 2, 1};
static const char *const TYPE_NAMES[] = {"obj", "classinfo", "name"};
static params TYPE_PARAMS = {TYPE_NAMES, NULL, 3, 2, 1};
static const char *const ITEMS_NAMES[] = {"iterable_obj", "item_type", "allow_subclass", "name"};
static params ITEMS_PARAMS = {ITEMS_NAMES, NULL, 4, 2, 1};
static const char *const CONTAINS_NAMES[] = {"container", "must_contain", "name"};
static params CONTAINS_PARAMS = {CONTAINS_NAMES, NULL, 3, 2, 1};

/* check_instance insists on a string name before anything else. */
static int name_ok(PyObject *name)
{
    return name == NULL || PyUnicode_Check(name);
}

/* isinstance, or the exact type when subclasses are not allowed. 1, 0, or -1 to fall back. */
static int instance_of(PyObject *obj, PyObject *classinfo, int allow_subclass)
{
    int result = PyObject_IsInstance(obj, classinfo);
    if (result < 0) {
        PyErr_Clear();
        return -1;
    }
    if (allow_subclass || !result) {
        return result;
    }
    PyObject *type = (PyObject *)Py_TYPE(obj);
    if (PyType_Check(classinfo)) {
        return type == classinfo;
    }
    /* A tuple, or a union whose members check_instance compares against */
    PyObject *members = PyTuple_Check(classinfo) ? Py_NewRef(classinfo)
                                                 : PyObject_GetAttrString(classinfo, "__args__");
    if (members == NULL || !PyTuple_Check(members)) {
        PyErr_Clear();
        Py_XDECREF(members);
        return -1;
    }
    int found = 0;
    Py_ssize_t n = PyTuple_Size(members);
    for (Py_ssize_t i = 0; i < n; i++) {
        found |= PyTuple_GetItem(members, i) == type;
    }
    Py_DECREF(members);
    return found;
}

static PyObject *fast_check_number(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[2];
    if (bind(&NUMBER_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL || !name_ok(a[1])) {
        RETURN_FALLBACK;
    }
    PyObject *num = a[0];
    if (!(PyLong_Check(num) || PyFloat_Check(num) || PyComplex_Check(num)) &&
        instance_of(num, cache.numbers_Number, 1) != 1) {
        RETURN_FALLBACK;
    }
    return Py_NewRef(num);
}

static PyObject *fast_check_string(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[3];
    if (bind(&STRING_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL || !name_ok(a[2])) {
        RETURN_FALLBACK;
    }
    int allow_subclass = truth(a[1], 1);
    if (allow_subclass < 0) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    if (!(allow_subclass ? PyUnicode_Check(a[0]) : PyUnicode_CheckExact(a[0]))) {
        RETURN_FALLBACK;
    }
    return Py_NewRef(a[0]);
}

static PyObject *fast_check_sequence(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[2];
    if (bind(&SEQUENCE_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL || !name_ok(a[1])) {
        RETURN_FALLBACK;
    }
    PyObject *obj = a[0];
    if (!(PyList_Check(obj) || PyTuple_Check(obj) || PyUnicode_Check(obj) || PyBytes_Check(obj) ||
          PyRange_Check(obj)) &&
        instance_of(obj, cache.abc_Sequence, 1) != 1) {
        RETURN_FALLBACK;
    }
    return Py_NewRef(obj);
}

/* The containers whose iteration neither consumes nor changes them. */
static int reiterable(PyObject *obj)
{
    return PyList_Check(obj) || PyTuple_Check(obj) || PyUnicode_Check(obj) || PyBytes_Check(obj) ||
           PyDict_Check(obj) || PyAnySet_Check(obj) || PyRange_Check(obj) || PyArray_Check(obj);
}

static PyObject *fast_check_iterable(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[2];
    if (bind(&ITERABLE_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL || !name_ok(a[1])) {
        RETURN_FALLBACK;
    }
    if (!reiterable(a[0]) && instance_of(a[0], cache.abc_Iterable, 1) != 1) {
        RETURN_FALLBACK;
    }
    return Py_NewRef(a[0]);
}

static PyObject *fast_check_instance(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[4];
    if (bind(&INSTANCE_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL || a[1] == NULL ||
        !name_ok(a[3])) {
        RETURN_FALLBACK;
    }
    int allow_subclass = truth(a[2], 1);
    if (allow_subclass < 0) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    if (instance_of(a[0], a[1], allow_subclass) != 1) {
        RETURN_FALLBACK;
    }
    return Py_NewRef(a[0]);
}

static PyObject *fast_check_type(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[3];
    if (bind(&TYPE_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL || a[1] == NULL ||
        !name_ok(a[2])) {
        RETURN_FALLBACK;
    }
    if (instance_of(a[0], a[1], 0) != 1) {
        RETURN_FALLBACK;
    }
    return Py_NewRef(a[0]);
}

static PyObject *fast_check_iterable_items(PyObject *const *args, Py_ssize_t nargs,
                                           PyObject *kwnames)
{
    PyObject *a[4];
    if (bind(&ITEMS_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL || a[1] == NULL ||
        !name_ok(a[3])) {
        RETURN_FALLBACK;
    }
    int allow_subclass = truth(a[2], 1);
    if (allow_subclass < 0) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    PyObject *items = a[0];
    /* An iterator would be consumed here and then again by Python; leave those to it */
    if (!reiterable(items)) {
        RETURN_FALLBACK;
    }
    int ok = 1;
    if (PyList_Check(items) || PyTuple_Check(items)) {
        Py_ssize_t n = PySequence_Size(items);
        for (Py_ssize_t i = 0; ok && i < n; i++) {
            PyObject *item = PySequence_GetItem(items, i);
            if (item == NULL) {
                PyErr_Clear();
                RETURN_FALLBACK;
            }
            ok = instance_of(item, a[1], allow_subclass) == 1;
            Py_DECREF(item);
        }
    }
    else {
        PyObject *iterator = PyObject_GetIter(items);
        if (iterator == NULL) {
            PyErr_Clear();
            RETURN_FALLBACK;
        }
        PyObject *item;
        while (ok && (item = PyIter_Next(iterator)) != NULL) {
            ok = instance_of(item, a[1], allow_subclass) == 1;
            Py_DECREF(item);
        }
        Py_DECREF(iterator);
        if (PyErr_Occurred()) {
            PyErr_Clear();
            RETURN_FALLBACK;
        }
    }
    if (!ok) {
        RETURN_FALLBACK;
    }
    return Py_NewRef(items);
}

static PyObject *fast_check_contains(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *a[3];
    if (bind(&CONTAINS_PARAMS, args, nargs, kwnames, a) < 0 || a[0] == NULL || a[1] == NULL) {
        RETURN_FALLBACK;
    }
    int found = PySequence_Contains(a[0], a[1]);
    if (found != 1) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    return Py_NewRef(a[1]);
}
