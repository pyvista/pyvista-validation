/* values.c: the element-wise checks, each array walked once. */

/* A comparison bound, kept exact for integers the way NumPy compares them. */
typedef struct {
    enum { BOUND_INT, BOUND_UINT, BOUND_DOUBLE } kind;
    npy_int64 i;
    npy_uint64 u;
    double d;
} bound;

typedef struct {
    int nonnegative;
    int finite;
    int integer;
    int low_set, high_set;
    bound low, high;
    int strict_low, strict_high;
} values_spec;

static float half_to_float(npy_uint16 h)
{
    npy_uint32 sign = (npy_uint32)(h & 0x8000) << 16;
    npy_uint32 exponent = (h >> 10) & 0x1f;
    npy_uint32 mantissa = h & 0x3ff;
    npy_uint32 bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        }
        else {
            exponent = 113;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exponent--;
            }
            bits = sign | (exponent << 23) | ((mantissa & 0x3ff) << 13);
        }
    }
    else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    }
    else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static inline double bound_double(const bound *b)
{
    return b->kind == BOUND_INT ? (double)b->i : b->kind == BOUND_UINT ? (double)b->u : b->d;
}

/* x >= low (or > with strict) for each element category, promoting as NumPy does. */
static inline int dbl_ge(double x, const bound *b, int strict)
{
    double v = bound_double(b);
    return strict ? x > v : x >= v;
}
static inline int dbl_le(double x, const bound *b, int strict)
{
    double v = bound_double(b);
    return strict ? x < v : x <= v;
}
static inline int i64_ge(npy_int64 x, const bound *b, int strict)
{
    if (b->kind == BOUND_INT) {
        return strict ? x > b->i : x >= b->i;
    }
    if (b->kind == BOUND_UINT) {
        return x < 0 ? 0 : strict ? (npy_uint64)x > b->u : (npy_uint64)x >= b->u;
    }
    return dbl_ge((double)x, b, strict);
}
static inline int i64_le(npy_int64 x, const bound *b, int strict)
{
    if (b->kind == BOUND_INT) {
        return strict ? x < b->i : x <= b->i;
    }
    if (b->kind == BOUND_UINT) {
        return x < 0 ? 1 : strict ? (npy_uint64)x < b->u : (npy_uint64)x <= b->u;
    }
    return dbl_le((double)x, b, strict);
}
static inline int u64_ge(npy_uint64 x, const bound *b, int strict)
{
    if (b->kind == BOUND_INT) {
        return b->i < 0 ? 1 : strict ? x > (npy_uint64)b->i : x >= (npy_uint64)b->i;
    }
    if (b->kind == BOUND_UINT) {
        return strict ? x > b->u : x >= b->u;
    }
    return dbl_ge((double)x, b, strict);
}
static inline int u64_le(npy_uint64 x, const bound *b, int strict)
{
    if (b->kind == BOUND_INT) {
        return b->i < 0 ? 0 : strict ? x < (npy_uint64)b->i : x <= (npy_uint64)b->i;
    }
    if (b->kind == BOUND_UINT) {
        return strict ? x < b->u : x <= b->u;
    }
    return dbl_le((double)x, b, strict);
}

/* A bound as one element category sees it, so that the loops compare without branching. */
typedef struct {
    enum { L_NONE, L_ALWAYS, L_NEVER, L_INT, L_UINT, L_DBL } kind;
    npy_int64 i;
    npy_uint64 u;
    double d;
    int strict;
} limit;

static limit float_limit(const bound *b, int set, int strict)
{
    limit l = {L_NONE, 0, 0, 0, strict};
    if (set) {
        l.kind = L_DBL;
        l.d = bound_double(b);
    }
    return l;
}

/* The limits of a signed integer array: a uint64 beyond int64 is above every element. */
static limit signed_limit(const bound *b, int set, int strict, int is_low)
{
    limit l = {L_NONE, 0, 0, 0, strict};
    if (!set) {
        return l;
    }
    if (b->kind == BOUND_DOUBLE) {
        l.kind = L_DBL;
        l.d = b->d;
    }
    else if (b->kind == BOUND_INT) {
        l.kind = L_INT;
        l.i = b->i;
    }
    else {
        l.kind = is_low ? L_NEVER : L_ALWAYS;
    }
    return l;
}

/* The limits of an unsigned or boolean array: a negative bound is below every element. */
static limit unsigned_limit(const bound *b, int set, int strict, int is_low)
{
    limit l = {L_NONE, 0, 0, 0, strict};
    if (!set) {
        return l;
    }
    if (b->kind == BOUND_DOUBLE) {
        l.kind = L_DBL;
        l.d = b->d;
    }
    else if (b->kind == BOUND_INT && b->i < 0) {
        l.kind = is_low ? L_ALWAYS : L_NEVER;
    }
    else {
        l.kind = L_UINT;
        l.u = b->kind == BOUND_INT ? (npy_uint64)b->i : b->u;
    }
    return l;
}

/* The loops below accumulate a `bad` flag over a cache-sized block instead of returning at
 * the first failure, which lets the compiler vectorize them; they stop after the first bad
 * block. `x` is contiguous; `step` in elements is 1 unless the run is strided. */
#define BLOCK 4096

#define BLOCK_LOOP(cond)                                     \
    for (npy_intp k = start; k < end; k++) {                 \
        bad |= !(cond);                                      \
    }
#define LIMIT_LOOPS(low_cmp, high_cmp, CONVERT)                                                \
    if (low.kind == L_NEVER || high.kind == L_NEVER) {                                          \
        bad = 1;                                                                                \
    }                                                                                           \
    if (low.kind == L_DBL) {                                                                    \
        if (low.strict) BLOCK_LOOP((double)x[k * step] > low.d)                                 \
        else BLOCK_LOOP((double)x[k * step] >= low.d)                                           \
    }                                                                                           \
    else if (low.kind == L_INT || low.kind == L_UINT) {                                         \
        if (low.strict) BLOCK_LOOP(CONVERT(x[k * step]) > low_cmp)                              \
        else BLOCK_LOOP(CONVERT(x[k * step]) >= low_cmp)                                        \
    }                                                                                           \
    if (high.kind == L_DBL) {                                                                   \
        if (high.strict) BLOCK_LOOP((double)x[k * step] < high.d)                               \
        else BLOCK_LOOP((double)x[k * step] <= high.d)                                          \
    }                                                                                           \
    else if (high.kind == L_INT || high.kind == L_UINT) {                                       \
        if (high.strict) BLOCK_LOOP(CONVERT(x[k * step]) < high_cmp)                            \
        else BLOCK_LOOP(CONVERT(x[k * step]) <= high_cmp)                                       \
    }

#define DEFINE_FLOAT_CHECK(NAME, CTYPE, FABS, FLOOR, LARGEST)                                   \
    static int NAME(const CTYPE *x, npy_intp n, npy_intp step, const values_spec *s)            \
    {                                                                                           \
        limit low = float_limit(&s->low, s->low_set, s->strict_low);                            \
        limit high = float_limit(&s->high, s->high_set, s->strict_high);                        \
        for (npy_intp start = 0; start < n; start += BLOCK) {                                   \
            npy_intp end = start + BLOCK < n ? start + BLOCK : n;                               \
            int bad = 0;                                                                        \
            if (s->finite) BLOCK_LOOP(FABS(x[k * step]) <= LARGEST)                             \
            if (s->nonnegative) BLOCK_LOOP(x[k * step] >= 0)                                    \
            if (s->integer) BLOCK_LOOP(x[k * step] == FLOOR(x[k * step]))                       \
            LIMIT_LOOPS(low.d, high.d, (double))                                                \
            if (bad) {                                                                          \
                return 0;                                                                       \
            }                                                                                   \
        }                                                                                       \
        return 1;                                                                               \
    }
#define DEFINE_SIGNED_CHECK(NAME, CTYPE)                                                        \
    static int NAME(const CTYPE *x, npy_intp n, npy_intp step, const values_spec *s)            \
    {                                                                                           \
        limit low = signed_limit(&s->low, s->low_set, s->strict_low, 1);                        \
        limit high = signed_limit(&s->high, s->high_set, s->strict_high, 0);                    \
        for (npy_intp start = 0; start < n; start += BLOCK) {                                   \
            npy_intp end = start + BLOCK < n ? start + BLOCK : n;                               \
            int bad = 0;                                                                        \
            if (s->nonnegative) BLOCK_LOOP(x[k * step] >= 0)                                    \
            LIMIT_LOOPS(low.i, high.i, (npy_int64))                                             \
            if (bad) {                                                                          \
                return 0;                                                                       \
            }                                                                                   \
        }                                                                                       \
        return 1;                                                                               \
    }
#define DEFINE_UNSIGNED_CHECK(NAME, CTYPE)                                                      \
    static int NAME(const CTYPE *x, npy_intp n, npy_intp step, const values_spec *s)            \
    {                                                                                           \
        limit low = unsigned_limit(&s->low, s->low_set, s->strict_low, 1);                      \
        limit high = unsigned_limit(&s->high, s->high_set, s->strict_high, 0);                  \
        for (npy_intp start = 0; start < n; start += BLOCK) {                                   \
            npy_intp end = start + BLOCK < n ? start + BLOCK : n;                               \
            int bad = 0;                                                                        \
            LIMIT_LOOPS(low.u, high.u, (npy_uint64))                                            \
            if (bad) {                                                                          \
                return 0;                                                                       \
            }                                                                                   \
        }                                                                                       \
        return 1;                                                                               \
    }

DEFINE_FLOAT_CHECK(check_double, npy_double, fabs, floor, DBL_MAX)
DEFINE_FLOAT_CHECK(check_float, npy_float, fabsf, floorf, FLT_MAX)
DEFINE_SIGNED_CHECK(check_int8, npy_int8)
DEFINE_SIGNED_CHECK(check_int16, npy_int16)
DEFINE_SIGNED_CHECK(check_int32, npy_int32)
DEFINE_SIGNED_CHECK(check_int64, npy_int64)
DEFINE_UNSIGNED_CHECK(check_uint8, npy_uint8)
DEFINE_UNSIGNED_CHECK(check_uint16, npy_uint16)
DEFINE_UNSIGNED_CHECK(check_uint32, npy_uint32)
DEFINE_UNSIGNED_CHECK(check_uint64, npy_uint64)

/* Half floats are converted to a block of floats first. */
static int check_half(const npy_uint16 *x, npy_intp n, npy_intp step, const values_spec *s)
{
    float block[BLOCK];
    for (npy_intp start = 0; start < n; start += BLOCK) {
        npy_intp end = start + BLOCK < n ? start + BLOCK : n;
        for (npy_intp k = start; k < end; k++) {
            block[k - start] = half_to_float(x[k * step]);
        }
        if (!check_float(block, end - start, 1, s)) {
            return 0;
        }
    }
    return 1;
}

/* Whether every element of a run passes; `step` is in elements. */
static int values_run(int typenum, const char *p, npy_intp n, npy_intp step,
                      const values_spec *s)
{
    switch (typenum) {
        case NPY_BOOL: return check_uint8((const npy_uint8 *)p, n, step, s);
        case NPY_BYTE: return check_int8((const npy_int8 *)p, n, step, s);
        case NPY_UBYTE: return check_uint8((const npy_uint8 *)p, n, step, s);
        case NPY_SHORT: return check_int16((const npy_int16 *)p, n, step, s);
        case NPY_USHORT: return check_uint16((const npy_uint16 *)p, n, step, s);
        case NPY_INT: return check_int32((const npy_int32 *)p, n, step, s);
        case NPY_UINT: return check_uint32((const npy_uint32 *)p, n, step, s);
        case NPY_LONG:
        case NPY_LONGLONG:
            return sizeof(npy_long) == 8 || typenum == NPY_LONGLONG
                       ? check_int64((const npy_int64 *)p, n, step, s)
                       : check_int32((const npy_int32 *)p, n, step, s);
        case NPY_ULONG:
        case NPY_ULONGLONG:
            return sizeof(npy_ulong) == 8 || typenum == NPY_ULONGLONG
                       ? check_uint64((const npy_uint64 *)p, n, step, s)
                       : check_uint32((const npy_uint32 *)p, n, step, s);
        case NPY_HALF: return check_half((const npy_uint16 *)p, n, step, s);
        case NPY_FLOAT: return check_float((const npy_float *)p, n, step, s);
        case NPY_DOUBLE: return check_double((const npy_double *)p, n, step, s);
        default: return -1;
    }
}

/* The dtypes the element-wise checks handle. */
static int values_supported(int typenum)
{
    return typenum == NPY_BOOL || (is_real_type(typenum) && typenum != NPY_LONGDOUBLE);
}

#define RELEASE_GIL_FROM 65536

/* Whether every element of the array passes. 1, 0, or -1 to fall back. */
static int values_ok(PyArrayObject *array, const values_spec *s)
{
    int typenum = PyArray_TYPE(array);
    if (!values_supported(typenum)) {
        return -1;
    }
    npy_intp n = PyArray_SIZE(array);
    if (n == 0) {
        return 1;
    }
    PyArrayObject *walk = array;
    PyObject *owned = NULL;
    npy_intp itemsize = PyArray_ITEMSIZE(array), step = 1;
    if (PyArray_IS_C_CONTIGUOUS(array) || PyArray_IS_F_CONTIGUOUS(array)) {
        step = 1;
    }
    else if (PyArray_NDIM(array) == 1 && PyArray_STRIDE(array, 0) % itemsize == 0 &&
             PyArray_STRIDE(array, 0) > 0) {
        step = PyArray_STRIDE(array, 0) / itemsize;
    }
    else {
        owned = PyArray_NewCopy(array, NPY_KEEPORDER);
        if (owned == NULL) {
            PyErr_Clear();
            return -1;
        }
        walk = (PyArrayObject *)owned;
    }
    const char *p = PyArray_BYTES(walk);
    PyThreadState *state = n >= RELEASE_GIL_FROM ? PyEval_SaveThread() : NULL;
    int result = values_run(typenum, p, n, step, s);
    if (state != NULL) {
        PyEval_RestoreThread(state);
    }
    Py_XDECREF(owned);
    return result;
}

/* ---- Bounds ----------------------------------------------------------------------------- */

/* Element `index` of a real array as a bound, widened the way `.item()` and NumPy's
 * comparison promotion see it. 0 when the dtype is not one the fast path compares. */
static int read_bound(PyArrayObject *array, npy_intp index, bound *b)
{
    int typenum = PyArray_TYPE(array);
    const char *p = PyArray_BYTES(array) + (PyArray_NDIM(array) ? index * PyArray_STRIDE(array, 0) : 0);
#define READ(CTYPE, FIELD, KIND)          \
    do {                                  \
        CTYPE raw;                        \
        memcpy(&raw, p, sizeof raw);      \
        b->FIELD = raw;                   \
        b->kind = KIND;                   \
    } while (0)
    switch (typenum) {
        case NPY_BYTE: READ(npy_byte, i, BOUND_INT); break;
        case NPY_SHORT: READ(npy_short, i, BOUND_INT); break;
        case NPY_INT: READ(npy_int, i, BOUND_INT); break;
        case NPY_LONG: READ(npy_long, i, BOUND_INT); break;
        case NPY_LONGLONG: READ(npy_longlong, i, BOUND_INT); break;
        case NPY_UBYTE: READ(npy_ubyte, u, BOUND_UINT); break;
        case NPY_USHORT: READ(npy_ushort, u, BOUND_UINT); break;
        case NPY_UINT: READ(npy_uint, u, BOUND_UINT); break;
        case NPY_ULONG: READ(npy_ulong, u, BOUND_UINT); break;
        case NPY_ULONGLONG: READ(npy_ulonglong, u, BOUND_UINT); break;
        case NPY_HALF: {
            npy_uint16 raw;
            memcpy(&raw, p, sizeof raw);
            b->d = half_to_float(raw);
            b->kind = BOUND_DOUBLE;
            break;
        }
        case NPY_FLOAT: READ(npy_float, d, BOUND_DOUBLE); break;
        case NPY_DOUBLE: READ(npy_double, d, BOUND_DOUBLE); break;
        default: return 0;
    }
#undef READ
    /* A Python int that fits is an int64 again once it becomes an array */
    if (b->kind == BOUND_UINT && b->u <= (npy_uint64)NPY_MAX_INT64) {
        b->i = (npy_int64)b->u;
        b->kind = BOUND_INT;
    }
    return 1;
}

/* Whether low <= high, the way check_sorted decides it for the two elements. */
static int bounds_ordered(const bound *low, const bound *high)
{
    if (low->kind == BOUND_DOUBLE || high->kind == BOUND_DOUBLE) {
        return bound_double(low) <= bound_double(high);
    }
    if (low->kind == BOUND_INT) {
        return i64_le(low->i, high, 0);
    }
    return u64_le(low->u, high, 0);
}

/* _validate_real_value: a real scalar as a bound. 1, or 0 when Python would raise. */
static int scalar_bound(PyObject *value, bound *b)
{
    if (PyBool_Check(value)) {
        return 0;
    }
    if (PyFloat_Check(value)) {
        b->kind = BOUND_DOUBLE;
        b->d = PyFloat_AsDouble(value);
        return 1;
    }
    if (PyLong_Check(value)) {
        int overflow;
        long long v = PyLong_AsLongLongAndOverflow(value, &overflow);
        if (overflow == 0) {
            b->kind = BOUND_INT;
            b->i = v;
            return 1;
        }
        if (overflow > 0) {
            unsigned long long u = PyLong_AsUnsignedLongLong(value);
            if (u == (unsigned long long)-1 && PyErr_Occurred()) {
                PyErr_Clear();
                return 0;
            }
            b->kind = BOUND_UINT;
            b->u = u;
            return 1;
        }
        return 0;
    }
    PyArrayObject *array = (PyArrayObject *)PyArray_FromAny(value, NULL, 0, 0, 0, NULL);
    if (array == NULL) {
        PyErr_Clear();
        return 0;
    }
    int ok = PyArray_NDIM(array) == 0 && read_bound(array, 0, b);
    Py_DECREF(array);
    return ok;
}

/* check_range's rng: two ordered real numbers as bounds. 1, or 0 when Python would raise. */
static int range_bounds(PyObject *rng, bound *low, bound *high)
{
    PyArrayObject *array = (PyArrayObject *)PyArray_FromAny(rng, NULL, 0, 0, 0, NULL);
    if (array == NULL) {
        PyErr_Clear();
        return 0;
    }
    int ok = PyArray_NDIM(array) == 1 && PyArray_DIM(array, 0) == 2 &&
             read_bound(array, 0, low) && read_bound(array, 1, high) && bounds_ordered(low, high);
    Py_DECREF(array);
    return ok;
}

/* ---- Sorted ----------------------------------------------------------------------------- */

#define DEFINE_SORTED(NAME, CTYPE)                                                             \
    static int NAME(const CTYPE *x, npy_intp n, npy_intp step, int mode)                        \
    {                                                                                           \
        for (npy_intp start = 1; start < n; start += BLOCK) {                                   \
            npy_intp end = start + BLOCK < n ? start + BLOCK : n;                               \
            int bad = 0;                                                                        \
            switch (mode) {                                                                     \
                case 0: BLOCK_LOOP(x[(k - 1) * step] <= x[k * step]) break;                     \
                case 1: BLOCK_LOOP(x[(k - 1) * step] < x[k * step]) break;                      \
                case 2: BLOCK_LOOP(x[(k - 1) * step] >= x[k * step]) break;                     \
                default: BLOCK_LOOP(x[(k - 1) * step] > x[k * step]) break;                     \
            }                                                                                   \
            if (bad) {                                                                          \
                return 0;                                                                       \
            }                                                                                   \
        }                                                                                       \
        return 1;                                                                               \
    }

DEFINE_SORTED(sorted_int8, npy_int8)
DEFINE_SORTED(sorted_int16, npy_int16)
DEFINE_SORTED(sorted_int32, npy_int32)
DEFINE_SORTED(sorted_int64, npy_int64)
DEFINE_SORTED(sorted_uint8, npy_uint8)
DEFINE_SORTED(sorted_uint16, npy_uint16)
DEFINE_SORTED(sorted_uint32, npy_uint32)
DEFINE_SORTED(sorted_uint64, npy_uint64)
DEFINE_SORTED(sorted_float, npy_float)
DEFINE_SORTED(sorted_double, npy_double)
DEFINE_SORTED(sorted_longdouble, npy_longdouble)

static int sorted_half(const npy_uint16 *x, npy_intp n, npy_intp step, int mode)
{
    float block[BLOCK + 1];
    for (npy_intp start = 0; start + 1 < n; start += BLOCK) {
        npy_intp end = start + BLOCK + 1 < n ? start + BLOCK + 1 : n;
        for (npy_intp k = start; k < end; k++) {
            block[k - start] = half_to_float(x[k * step]);
        }
        if (!sorted_float(block, end - start, 1, mode)) {
            return 0;
        }
    }
    return 1;
}

/* Whether one run along the axis is sorted; mode: 0 <=, 1 <, 2 >=, 3 >. `step` in elements. */
static int sorted_run(int typenum, const char *p, npy_intp n, npy_intp step, int mode)
{
    switch (typenum) {
        case NPY_BOOL: return sorted_uint8((const npy_uint8 *)p, n, step, mode);
        case NPY_BYTE: return sorted_int8((const npy_int8 *)p, n, step, mode);
        case NPY_UBYTE: return sorted_uint8((const npy_uint8 *)p, n, step, mode);
        case NPY_SHORT: return sorted_int16((const npy_int16 *)p, n, step, mode);
        case NPY_USHORT: return sorted_uint16((const npy_uint16 *)p, n, step, mode);
        case NPY_INT: return sorted_int32((const npy_int32 *)p, n, step, mode);
        case NPY_UINT: return sorted_uint32((const npy_uint32 *)p, n, step, mode);
        case NPY_LONG:
        case NPY_LONGLONG:
            return sizeof(npy_long) == 8 || typenum == NPY_LONGLONG
                       ? sorted_int64((const npy_int64 *)p, n, step, mode)
                       : sorted_int32((const npy_int32 *)p, n, step, mode);
        case NPY_ULONG:
        case NPY_ULONGLONG:
            return sizeof(npy_ulong) == 8 || typenum == NPY_ULONGLONG
                       ? sorted_uint64((const npy_uint64 *)p, n, step, mode)
                       : sorted_uint32((const npy_uint32 *)p, n, step, mode);
        case NPY_HALF: return sorted_half((const npy_uint16 *)p, n, step, mode);
        case NPY_FLOAT: return sorted_float((const npy_float *)p, n, step, mode);
        case NPY_DOUBLE: return sorted_double((const npy_double *)p, n, step, mode);
        case NPY_LONGDOUBLE: return sorted_longdouble((const npy_longdouble *)p, n, step, mode);
        default: return -1;
    }
}

/* check_sorted's axis: -1 by default, None to flatten, else an in-range integer. Returns
 * the axis, NDIM for None, or -2 when Python would raise. */
static int read_axis(PyObject *axis_obj, int ndim)
{
    if (axis_obj == NULL) {
        return ndim - 1;
    }
    if (axis_obj == Py_None) {
        return ndim;
    }
    double value;
    if (!as_double(axis_obj, &value) || PyBool_Check(axis_obj) || value != floor(value) ||
        value < -ndim || value > ndim - 1) {
        return -2;
    }
    int axis = (int)value;
    return axis < 0 ? axis + ndim : axis;
}

/* check_sorted. 1, 0, or -1 to fall back. */
static int sorted_ok(PyArrayObject *array, int ascending, int strict, PyObject *axis_obj)
{
    int typenum = PyArray_TYPE(array);
    if (!(typenum == NPY_BOOL || is_real_type(typenum))) {
        return -1;
    }
    int ndim = PyArray_NDIM(array);
    if (ndim == 0) {
        return 1;
    }
    int axis = read_axis(axis_obj, ndim);
    if (axis == -2) {
        return -1;
    }
    int mode = ascending ? (strict ? 1 : 0) : (strict ? 3 : 2);
    npy_intp size = PyArray_SIZE(array);
    if (size < 2) {
        return 1;
    }
    PyObject *owned = NULL;
    PyArrayObject *walk = array;
    if (axis == ndim) {
        /* ravel(order='A'): a contiguous array is already flat in memory */
        if (!(PyArray_IS_C_CONTIGUOUS(array) || PyArray_IS_F_CONTIGUOUS(array))) {
            owned = PyArray_Ravel(array, NPY_ANYORDER);
            if (owned == NULL) {
                PyErr_Clear();
                return -1;
            }
            walk = (PyArrayObject *)owned;
        }
    }
    npy_intp itemsize = PyArray_ITEMSIZE(array);
    npy_intp stride = axis == ndim ? (PyArray_NDIM(walk) == 1 ? PyArray_STRIDE(walk, 0) : itemsize)
                                   : PyArray_STRIDE(array, axis);
    if (stride <= 0 || stride % itemsize != 0) {
        /* Reversed or unaligned views are rare; a C-ordered copy is walked instead */
        PyObject *copy = PyArray_NewCopy(array, NPY_CORDER);
        Py_XDECREF(owned);
        if (copy == NULL) {
            PyErr_Clear();
            return -1;
        }
        owned = copy;
        walk = array = (PyArrayObject *)copy;
        stride = axis == ndim ? itemsize : PyArray_STRIDE(array, axis);
    }
    npy_intp step = stride / itemsize;
    PyThreadState *state = size >= RELEASE_GIL_FROM ? PyEval_SaveThread() : NULL;
    int result;
    if (axis == ndim) {
        result = sorted_run(typenum, PyArray_BYTES(walk), size, step, mode);
    }
    else {
        /* Every run along the axis, stepping through the other dimensions */
        npy_intp n = PyArray_DIM(array, axis);
        npy_intp dims[MAXDIMS], strides[MAXDIMS], index[MAXDIMS];
        int m = 0;
        for (int i = 0; i < ndim; i++) {
            if (i != axis) {
                dims[m] = PyArray_DIM(array, i);
                strides[m] = PyArray_STRIDE(array, i);
                index[m] = 0;
                m++;
            }
        }
        const char *p = PyArray_BYTES(array);
        result = 1;
        while (result == 1) {
            result = sorted_run(typenum, p, n, step, mode);
            int j = m - 1;
            for (; j >= 0; j--) {
                p += strides[j];
                if (++index[j] < dims[j]) {
                    break;
                }
                p -= strides[j] * dims[j];
                index[j] = 0;
            }
            if (j < 0) {
                break;
            }
        }
    }
    if (state != NULL) {
        PyEval_RestoreThread(state);
    }
    Py_XDECREF(owned);
    return result;
}

/* must_be_sorted: True for the defaults or a dict of check_sorted's keywords. 1, or 0 when
 * Python would raise or the fast path does not read it. */
static int read_sorted_spec(PyObject *spec, int *ascending, int *strict, PyObject **axis)
{
    *ascending = 1;
    *strict = 0;
    *axis = NULL;
    if (PyBool_Check(spec)) {
        return 1;
    }
    if (!PyDict_Check(spec)) {
        return 0;
    }
    Py_ssize_t seen = 0;
    PyObject *item;
    if ((item = PyDict_GetItemString(spec, "ascending")) != NULL) {
        seen++;
        *ascending = truth(item, 1);
    }
    if ((item = PyDict_GetItemString(spec, "strict")) != NULL) {
        seen++;
        *strict = truth(item, 0);
    }
    if ((item = PyDict_GetItemString(spec, "axis")) != NULL) {
        seen++;
        *axis = item;
    }
    if (*ascending < 0 || *strict < 0) {
        PyErr_Clear();
        return 0;
    }
    return seen == PyDict_Size(spec);
}
