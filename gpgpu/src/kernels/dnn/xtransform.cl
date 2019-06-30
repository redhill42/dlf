// Enable loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#define TRANSFORM(name, op) \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1))) \
void X##name(const int n, \
    const __global real* restrict xgm, const int x_offset, const int x_inc, \
    __global real* ygm, const int y_offset, const int y_inc) \
{ \
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) { \
    real x_value = xgm[id*x_inc + x_offset]; \
    ygm[id*y_inc + y_offset] = op(x_value); \
  } \
}

#ifdef CUDA
  #if PRECISION == 16
    #define XOP(op) h##op
  #elif PRECISION == 32
    #define XOP(op) op##f
  #else
    #define XOP(op) op
  #endif
#else
  #define XOP(op) op
#endif

#define TRANSFORM_X(name) TRANSFORM(name, XOP(name))

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
#define sigmoid(x) (ONE/(ONE+exp(-x)))

TRANSFORM(reciprocal, reciprocal)
TRANSFORM(sigmoid, sigmoid)

)" // End of the C++11 raw string literal
