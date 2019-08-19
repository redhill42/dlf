// Enable loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#define TRANSFORM(name, op)                                             \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))               \
void X##name(const int n,                                               \
             const __global real* restrict xgm, const int x_offset,     \
             __global real* ygm, const int y_offset) {                  \
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {   \
    real x = xgm[id + x_offset];                                        \
    ygm[id + y_offset] = op(x);                                         \
  }                                                                     \
}                                                                       \
                                                                        \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))               \
void X##name##Strided(                                                  \
    const int n, const int rank, __constant int* shape,                 \
    const __global real* restrict xgm, const int x_offset,              \
    __global real* ygm, const int y_offset)                             \
{                                                                       \
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {   \
    int x_id = x_offset, y_id = y_offset;                               \
    unravel2(id, &x_id, &y_id, rank, shape, &shape[rank], &shape[rank*2]);\
    real x = xgm[x_id];                                                 \
    ygm[y_id] = op(x);                                                  \
  }                                                                     \
}

#if INTEGER_PRECISION
  TRANSFORM(abs, abs)
#else
  TRANSFORM(abs, fabs)
#endif

#define neg(x) (-(x))
TRANSFORM(neg, neg)

#if defined(CUDA) || INTEGER_PRECISION
  #define sign(x) ((ZERO<(x)) - ((x)<ZERO))
#endif
TRANSFORM(sign, sign)

#if PRECISION == 16 || PRECISION == 32 || PRECISION == 64

#if defined(CUDA) && PRECISION == 16
  #define XF(name) h##name
#elif defined(CUDA) && PRECISION == 32
  #define XF(name) name##f
#else
  #define XF(name) name
#endif
#define TRANSFORM_X(name) TRANSFORM(name, XF(name))

TRANSFORM_X(floor)
TRANSFORM_X(ceil)
TRANSFORM_X(round)
TRANSFORM_X(sqrt)
TRANSFORM_X(exp)
TRANSFORM_X(log)
TRANSFORM_X(sin)
TRANSFORM_X(cos)
TRANSFORM_X(tan)
TRANSFORM_X(asin)
TRANSFORM_X(acos)
TRANSFORM_X(atan)
TRANSFORM_X(sinh)
TRANSFORM_X(cosh)
TRANSFORM_X(tanh)
TRANSFORM_X(asinh)
TRANSFORM_X(acosh)
TRANSFORM_X(atanh)
TRANSFORM_X(erf)

#define reciprocal(x) (ONE/(x))
#define square(x) (x*x)
#define sigmoid(x) (ONE/(ONE+exp(-x)))

TRANSFORM(reciprocal, reciprocal)
TRANSFORM(square, square)
TRANSFORM(sigmoid, sigmoid)
#endif

)" // End of the C++11 raw string literal
