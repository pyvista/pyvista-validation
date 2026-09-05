/* Shared declarations for the C fast paths of pyvista_validation.
 *
 * Every fast path mirrors the Python function of the same name. It returns the
 * result when it can establish it and FALLBACK otherwise, in which case the
 * Python function runs and returns or raises with the documented message.
 */
#ifndef PYVISTA_VALIDATION_FAST_H
#define PYVISTA_VALIDATION_FAST_H

#include <Python.h>

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PY_ARRAY_UNIQUE_SYMBOL pyvista_validation_fast_ARRAY_API
#include <numpy/arrayobject.h>

/* The sentinel a fast path returns to hand the call to Python. */
static PyObject *FALLBACK;
#define RETURN_FALLBACK return Py_NewRef(FALLBACK)

/* A fast path takes the raw call: a new reference, FALLBACK, or NULL comes back. */
typedef PyObject *(*fastpath)(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

/* The parameters of a function, positional ones first. */
typedef struct {
    const char *const *names;
    PyObject **interned;
    int count;
    int positional;      /* how many may be passed positionally */
    int positional_only; /* how many may not be passed by keyword */
} params;

static int bind(const params *spec, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                PyObject **out);
static int truth(PyObject *obj, int missing);
static PyObject *call(PyObject *callable, PyObject *const *args, Py_ssize_t nargs,
                      PyObject *kwnames);

/* Python objects the fast paths compare against, resolved once at import. */
typedef struct {
    PyObject *numbers_Number;
    PyObject *abc_Sequence;
    PyObject *abc_Iterable;
    PyObject *np_broadcast_to;
    PyObject *np_generic;
} cache_t;
static cache_t cache;

#endif
