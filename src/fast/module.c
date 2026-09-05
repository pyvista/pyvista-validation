/* pyvista_validation._fast: the C fast paths behind the public functions. */
#include "fast.h"

/* ---- Binding a call to a parameter table ---------------------------------------------- */

/* Leave a borrowed reference per parameter in `out`, NULL where it was not given.
 * Returns -1 when Python would raise, or when the call is one the fast path does not take. */
static int bind(const params *spec, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                PyObject **out)
{
    if (nargs > spec->positional) {
        return -1;
    }
    for (int i = 0; i < spec->count; i++) {
        out[i] = i < nargs ? args[i] : NULL;
    }
    if (kwnames == NULL) {
        return 0;
    }
    Py_ssize_t nkw = PyTuple_Size(kwnames);
    for (Py_ssize_t k = 0; k < nkw; k++) {
        PyObject *key = PyTuple_GetItem(kwnames, k);
        int index = -1;
        for (int j = spec->positional_only; j < spec->count; j++) {
            if (spec->interned[j] == key) {
                index = j;
                break;
            }
        }
        if (index < 0) {
            for (int j = spec->positional_only; j < spec->count; j++) {
                if (PyUnicode_CompareWithASCIIString(key, spec->names[j]) == 0) {
                    index = j;
                    break;
                }
            }
        }
        if (index < 0 || out[index] != NULL) {
            return -1;
        }
        out[index] = args[nargs + k];
    }
    return 0;
}

/* Intern the names of a parameter table. */
static int intern(params *spec)
{
    spec->interned = calloc((size_t)spec->count, sizeof(PyObject *));
    if (spec->interned == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    for (int i = 0; i < spec->count; i++) {
        spec->interned[i] = PyUnicode_InternFromString(spec->names[i]);
        if (spec->interned[i] == NULL) {
            return -1;
        }
    }
    return 0;
}

/* The truth value of an optional argument; -1 when Python would raise. */
static int truth(PyObject *obj, int missing)
{
    if (obj == NULL) {
        return missing;
    }
    if (obj == Py_True) {
        return 1;
    }
    if (obj == Py_False) {
        return 0;
    }
    return PyObject_IsTrue(obj);
}

/* Call a Python callable with the raw vectorcall arguments. */
static PyObject *call(PyObject *callable, PyObject *const *args, Py_ssize_t nargs,
                      PyObject *kwnames)
{
    PyObject *tuple = PyTuple_New(nargs);
    if (tuple == NULL) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < nargs; i++) {
        PyTuple_SetItem(tuple, i, Py_NewRef(args[i]));
    }
    PyObject *dict = NULL;
    if (kwnames != NULL) {
        dict = PyDict_New();
        if (dict == NULL) {
            Py_DECREF(tuple);
            return NULL;
        }
        Py_ssize_t nkw = PyTuple_Size(kwnames);
        for (Py_ssize_t k = 0; k < nkw; k++) {
            if (PyDict_SetItem(dict, PyTuple_GetItem(kwnames, k), args[nargs + k]) < 0) {
                Py_DECREF(tuple);
                Py_DECREF(dict);
                return NULL;
            }
        }
    }
    PyObject *result = PyObject_Call(callable, tuple, dict);
    Py_DECREF(tuple);
    Py_XDECREF(dict);
    return result;
}

#include "array.c"
#include "values.c"
#include "checks.c"
#include "validate.c"

/* ---- The public builtins ------------------------------------------------------------- */

/* What a builtin needs to know: its fast path and the Python function behind it. */
typedef struct {
    PyObject_HEAD
    PyObject *function;
    fastpath fast;
} Context;

static PyTypeObject *ContextType;

static void Context_dealloc(PyObject *self)
{
    Py_XDECREF(((Context *)self)->function);
    PyTypeObject *type = Py_TYPE(self);
    freefunc free_object = (freefunc)PyType_GetSlot(type, Py_tp_free);
    free_object(self);
    Py_DECREF(type);
}

static PyType_Slot Context_slots[] = {
    {Py_tp_dealloc, (void *)Context_dealloc},
    {0, NULL},
};

static PyType_Spec Context_spec = {
    "pyvista_validation._fast.Context", sizeof(Context), 0, Py_TPFLAGS_DEFAULT, Context_slots,
};

/* Run the fast path, and the Python function when the fast path declines. */
static PyObject *dispatch(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
                          PyObject *kwnames)
{
    Context *context = (Context *)self;
    PyObject *result = context->fast(args, nargs, kwnames);
    if (result != FALLBACK) {
        return result;
    }
    Py_DECREF(result);
    return call(context->function, args, nargs, kwnames);
}

typedef struct {
    const char *name;
    fastpath fast;
} entry;

static const entry TABLE[] = {
    {"check_subdtype", fast_check_subdtype},
    {"check_real", fast_check_real},
    {"check_shape", fast_check_shape},
    {"check_ndim", fast_check_ndim},
    {"check_length", fast_check_length},
    {"check_number", fast_check_number},
    {"check_string", fast_check_string},
    {"check_sequence", fast_check_sequence},
    {"check_iterable", fast_check_iterable},
    {"check_instance", fast_check_instance},
    {"check_type", fast_check_type},
    {"check_iterable_items", fast_check_iterable_items},
    {"check_contains", fast_check_contains},
    {"check_finite", fast_check_finite},
    {"check_nonnegative", fast_check_nonnegative},
    {"check_integer", fast_check_integer},
    {"check_greater_than", fast_check_greater_than},
    {"check_less_than", fast_check_less_than},
    {"check_range", fast_check_range},
    {"check_sorted", fast_check_sorted},
    {"validate_array", fast_validate_array},
    {NULL, NULL},
};

static char *duplicate(const char *text)
{
    size_t size = strlen(text) + 1;
    char *copy = malloc(size);
    if (copy != NULL) {
        memcpy(copy, text, size);
    }
    return copy;
}

/* wrap(function, name, text_signature, module): the builtin for a public function, or the
 * function itself when there is no fast path for it. */
static PyObject *fast_wrap(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError, "wrap() takes 4 positional arguments");
        return NULL;
    }
    PyObject *function = args[0];
    const char *name = PyUnicode_AsUTF8AndSize(args[1], NULL);
    const char *signature = PyUnicode_AsUTF8AndSize(args[2], NULL);
    if (name == NULL || signature == NULL || !PyUnicode_Check(args[3])) {
        PyErr_SetString(PyExc_TypeError, "wrap() expects strings after the function");
        return NULL;
    }
    fastpath fast = NULL;
    for (const entry *e = TABLE; e->name != NULL; e++) {
        if (strcmp(e->name, name) == 0) {
            fast = e->fast;
            break;
        }
    }
    if (fast == NULL) {
        return Py_NewRef(function);
    }

    PyObject *doc = PyObject_GetAttrString(function, "__doc__");
    if (doc == NULL) {
        return NULL;
    }
    const char *docstring = doc == Py_None ? "" : PyUnicode_AsUTF8AndSize(doc, NULL);
    if (docstring == NULL) {
        Py_DECREF(doc);
        return NULL;
    }
    /* CPython reads `name(...)\n--\n\n` off the front of the docstring as the signature. */
    size_t size = strlen(name) + strlen(signature) + 5 + strlen(docstring) + 1;
    char *text = malloc(size);
    PyMethodDef *def = calloc(1, sizeof(PyMethodDef));
    Context *context = (Context *)PyType_GenericAlloc(ContextType, 0);
    if (text == NULL || def == NULL || context == NULL) {
        Py_DECREF(doc);
        Py_XDECREF(context);
        free(text);
        free(def);
        return PyErr_NoMemory();
    }
    snprintf(text, size, "%s%s\n--\n\n%s", name, signature, docstring);
    Py_DECREF(doc);
    def->ml_name = duplicate(name);
    def->ml_meth = (PyCFunction)(void (*)(void))dispatch;
    def->ml_flags = METH_FASTCALL | METH_KEYWORDS;
    def->ml_doc = text;
    context->function = Py_NewRef(function);
    context->fast = fast;
    PyObject *builtin = PyCFunction_NewEx(def, (PyObject *)context, args[3]);
    Py_DECREF(context);
    return builtin;
}

static PyMethodDef methods[] = {
    {"wrap", (PyCFunction)(void (*)(void))fast_wrap, METH_FASTCALL,
     "wrap($module, function, name, text_signature, module, /)\n--\n\n"
     "Return the builtin that runs the C fast path for ``function``, or ``function`` itself."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "pyvista_validation._fast",
    "C fast paths for the validation functions.", -1, methods,
};

/* Import an attribute of a module, keeping a reference for the life of the process. */
static PyObject *import_attribute(const char *module, const char *attribute)
{
    PyObject *mod = PyImport_ImportModule(module);
    if (mod == NULL) {
        return NULL;
    }
    PyObject *value = PyObject_GetAttrString(mod, attribute);
    Py_DECREF(mod);
    return value;
}

static params *ALL_PARAMS[] = {
    &ARRAY_PARAMS,      &FINITE_PARAMS, &NONNEGATIVE_PARAMS, &INTEGER_PARAMS,
    &GREATER_PARAMS,    &LESS_PARAMS,   &RANGE_PARAMS,       &SORTED_PARAMS,
    &SUBDTYPE_PARAMS,   &REAL_PARAMS,   &SHAPE_PARAMS,       &NDIM_PARAMS,
    &LENGTH_PARAMS,     &NUMBER_PARAMS, &STRING_PARAMS,      &SEQUENCE_PARAMS,
    &ITERABLE_PARAMS,   &INSTANCE_PARAMS, &TYPE_PARAMS,      &ITEMS_PARAMS,
    &CONTAINS_PARAMS,
};

PyMODINIT_FUNC PyInit__fast(void)
{
    import_array();
    FALLBACK = PyObject_CallNoArgs((PyObject *)&PyBaseObject_Type);
    ContextType = (PyTypeObject *)PyType_FromSpec(&Context_spec);
    if (FALLBACK == NULL || ContextType == NULL) {
        return NULL;
    }
    cache.numbers_Number = import_attribute("numbers", "Number");
    cache.abc_Sequence = import_attribute("collections.abc", "Sequence");
    cache.abc_Iterable = import_attribute("collections.abc", "Iterable");
    cache.np_broadcast_to = import_attribute("numpy", "broadcast_to");
    cache.np_generic = import_attribute("numpy", "generic");
    if (cache.numbers_Number == NULL || cache.abc_Sequence == NULL || cache.abc_Iterable == NULL ||
        cache.np_broadcast_to == NULL || cache.np_generic == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < sizeof(ALL_PARAMS) / sizeof(ALL_PARAMS[0]); i++) {
        if (intern(ALL_PARAMS[i]) < 0) {
            return NULL;
        }
    }
    return PyModule_Create(&moduledef);
}
