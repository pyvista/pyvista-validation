/* array.c: array conversion, the structural checks and the output modes. */

#define MAXDIMS 64

/* np.asanyarray or np.asarray, with the copy rule of _cast_to_numpy: a new reference to an
 * array of any dtype but object, or FALLBACK. */
static PyObject *as_array(PyObject *obj, int as_any, int copy)
{
    PyObject *array = PyArray_FromAny(obj, NULL, 0, 0, as_any ? 0 : NPY_ARRAY_ENSUREARRAY, NULL);
    if (array == NULL) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    if (PyArray_TYPE((PyArrayObject *)array) == NPY_OBJECT) {
        Py_DECREF(array);
        RETURN_FALLBACK;
    }
    if (copy && PyArray_Check(obj)) {
        /* ndarray.copy() in its default C order; a sequence already became a new array */
        PyObject *copied = PyArray_NewCopy((PyArrayObject *)array, NPY_CORDER);
        Py_DECREF(array);
        if (copied == NULL) {
            PyErr_Clear();
            RETURN_FALLBACK;
        }
        return copied;
    }
    return array;
}

/* np.asanyarray of anything but object arrays, for the checks that only read an array. */
static PyObject *any_array(PyObject *obj)
{
    if (PyArray_Check(obj)) {
        if (PyArray_TYPE((PyArrayObject *)obj) == NPY_OBJECT) {
            RETURN_FALLBACK;
        }
        return Py_NewRef(obj);
    }
    return as_array(obj, 1, 0);
}

static int is_real_type(int typenum)
{
    return PyTypeNum_ISINTEGER(typenum) || PyTypeNum_ISFLOAT(typenum);
}

/* A Python number as a double; 0 when it is not one Python would accept as a number. */
static int as_double(PyObject *obj, double *value)
{
    if (PyFloat_Check(obj)) {
        *value = PyFloat_AsDouble(obj);
        return 1;
    }
    if (PyLong_Check(obj) || PyArray_IsScalar(obj, Number) || PyArray_IsScalar(obj, Bool)) {
        *value = PyFloat_AsDouble(obj);
        if (*value == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return 0;
        }
        return 1;
    }
    return 0;
}

/* ---- Shapes ----------------------------------------------------------------------------- */

/* Read an int or a tuple of ints into dims. Returns the count, or -1 when the value is not
 * one the fast path takes (Python then validates or rejects it). */
static int read_dims(PyObject *spec, npy_intp *dims, npy_intp lowest)
{
    if (PyLong_Check(spec)) {
        long d = PyLong_AsLong(spec);
        if ((d == -1 && PyErr_Occurred()) || d < lowest) {
            PyErr_Clear();
            return -1;
        }
        dims[0] = d;
        return 1;
    }
    if (!PyTuple_Check(spec)) {
        return -1;
    }
    Py_ssize_t n = PyTuple_Size(spec);
    if (n > MAXDIMS) {
        return -1;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyTuple_GetItem(spec, i);
        if (!PyLong_Check(item)) {
            return -1;
        }
        long d = PyLong_AsLong(item);
        if ((d == -1 && PyErr_Occurred()) || d < lowest) {
            PyErr_Clear();
            return -1;
        }
        dims[i] = d;
    }
    return (int)n;
}

/* Whether the array has one shape spec, -1 standing for any size. 1, 0, or -1 to fall back. */
static int shape_matches(PyArrayObject *array, PyObject *spec)
{
    npy_intp dims[MAXDIMS];
    int n = read_dims(spec, dims, -1);
    if (n < 0) {
        return -1;
    }
    if (n != PyArray_NDIM(array)) {
        return 0;
    }
    for (int i = 0; i < n; i++) {
        if (dims[i] != -1 && dims[i] != PyArray_DIM(array, i)) {
            return 0;
        }
    }
    return 1;
}

/* check_shape: a shape spec, or a list of them of which one must match. */
static int shape_ok(PyArrayObject *array, PyObject *spec)
{
    if (!PyList_Check(spec)) {
        return shape_matches(array, spec);
    }
    Py_ssize_t n = PyList_Size(spec);
    for (Py_ssize_t i = 0; i < n; i++) {
        int match = shape_matches(array, PyList_GetItem(spec, i));
        if (match != 0) {
            return match;
        }
    }
    return 0;
}

/* Read the numbers an array-like holds as doubles, for the small specs that name sizes.
 * Returns the count, or -1 when they are not plain real numbers. */
static Py_ssize_t read_numbers(PyObject *spec, double **numbers, PyObject **holder)
{
    PyArrayObject *array = (PyArrayObject *)PyArray_FromAny(spec, NULL, 0, 0, 0, NULL);
    if (array == NULL) {
        PyErr_Clear();
        return -1;
    }
    int typenum = PyArray_TYPE(array);
    if (!(is_real_type(typenum) || typenum == NPY_BOOL)) {
        Py_DECREF(array);
        return -1;
    }
    PyObject *doubles = PyArray_FromArray(array, PyArray_DescrFromType(NPY_DOUBLE),
                                          NPY_ARRAY_CARRAY_RO | NPY_ARRAY_FORCECAST);
    Py_DECREF(array);
    if (doubles == NULL) {
        PyErr_Clear();
        return -1;
    }
    *holder = doubles;
    *numbers = (double *)PyArray_DATA((PyArrayObject *)doubles);
    return PyArray_SIZE((PyArrayObject *)doubles);
}

/* Whether `value` is among the numbers of `spec`, the way `value in np.atleast_1d(spec)`
 * decides it. 1, 0, or -1 to fall back. */
static int number_in(npy_intp value, PyObject *spec, int whole)
{
    if (PyLong_Check(spec)) {
        long v = PyLong_AsLong(spec);
        if (v == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return -1;
        }
        return v == value;
    }
    double *numbers;
    PyObject *holder;
    Py_ssize_t n = read_numbers(spec, &numbers, &holder);
    if (n < 0) {
        return -1;
    }
    int found = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        if (whole && numbers[i] != floor(numbers[i])) {
            found = -1;
            break;
        }
        if (numbers[i] == (double)value) {
            found = 1;
        }
    }
    Py_DECREF(holder);
    return found;
}

/* check_length on an array with allow_scalar=True. 1, 0, or -1 to fall back. */
static int length_ok(PyArrayObject *array, PyObject *exact, PyObject *minimum, PyObject *maximum)
{
    npy_intp length = PyArray_NDIM(array) == 0 ? 1 : PyArray_DIM(array, 0);
    if (exact != NULL) {
        int found = number_in(length, exact, 1);
        if (found != 1) {
            return found;
        }
    }
    double low = 0, high = 0;
    if (minimum != NULL && (!as_double(minimum, &low) || !isfinite(low))) {
        return -1;
    }
    if (maximum != NULL && (!as_double(maximum, &high) || !isfinite(high))) {
        return -1;
    }
    if (minimum != NULL && maximum != NULL && !(low <= high)) {
        return -1;
    }
    if (minimum != NULL && (double)length < low) {
        return 0;
    }
    if (maximum != NULL && (double)length > high) {
        return 0;
    }
    return 1;
}

/* ---- Data types ------------------------------------------------------------------------- */

/* np.issubdtype(descr, base) for a single base. 1, 0, or -1 to fall back. */
static int subdtype(PyArray_Descr *descr, PyObject *base)
{
    PyObject *cls;
    PyArray_Descr *base_descr = NULL;
    if (PyType_Check(base) && PyType_IsSubtype((PyTypeObject *)base, (PyTypeObject *)cache.np_generic)) {
        cls = base;
    }
    else {
        if (!PyArray_DescrConverter(base, &base_descr)) {
            PyErr_Clear();
            return -1;
        }
        cls = (PyObject *)base_descr->typeobj;
    }
    int result = PyObject_IsSubclass((PyObject *)descr->typeobj, cls);
    Py_XDECREF(base_descr);
    if (result < 0) {
        PyErr_Clear();
        return -1;
    }
    return result;
}

/* check_subdtype: one base, or a tuple or list of bases of which one must match. */
static int subdtype_ok(PyArray_Descr *descr, PyObject *base)
{
    if (!PyTuple_Check(base) && !PyList_Check(base)) {
        return subdtype(descr, base);
    }
    Py_ssize_t n = PySequence_Size(base);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(base, i);
        int match = subdtype(descr, item);
        Py_DECREF(item);
        if (match != 0) {
            return match;
        }
    }
    return 0;
}

/* ---- Output ------------------------------------------------------------------------------ */

/* ndarray.astype(dtype, copy=False): the array itself unless the dtype differs. */
static PyObject *cast_to(PyArrayObject *array, PyObject *dtype)
{
    PyArray_Descr *descr = NULL;
    if (!PyArray_DescrConverter2(dtype, &descr)) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    if (descr == NULL) {
        return Py_NewRef(array);
    }
    PyObject *out = PyArray_FromArray(array, descr, NPY_ARRAY_FORCECAST);
    if (out == NULL) {
        PyErr_Clear();
        RETURN_FALLBACK;
    }
    return out;
}

/* ndarray.tolist(), through the method for subclasses that override it. */
static PyObject *to_list(PyArrayObject *array)
{
    if (PyArray_CheckExact(array)) {
        return PyArray_ToList(array);
    }
    return PyObject_CallMethod((PyObject *)array, "tolist", NULL);
}

/* Nested lists as nested tuples. */
static PyObject *to_tuple(PyObject *obj)
{
    if (!PyList_Check(obj)) {
        return Py_NewRef(obj);
    }
    Py_ssize_t n = PyList_Size(obj);
    PyObject *tuple = PyTuple_New(n);
    if (tuple == NULL) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = to_tuple(PyList_GetItem(obj, i));
        if (item == NULL) {
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SetItem(tuple, i, item);
    }
    return tuple;
}
