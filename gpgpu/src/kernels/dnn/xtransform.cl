// Enable loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#define TRANSFORM(name)                                                 \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))               \
void X##name(const int n,                                               \
             const __global real* restrict xgm, const int x_offset,     \
             __global real* ygm, const int y_offset) {                  \
  ygm = &ygm[y_offset];                                                 \
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {   \
    real x = xgm[id + x_offset];                                        \
    name##_op(ygm[id], x);                                              \
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
    unravel2(id, &x_id, &y_id, rank, shape);                            \
    real x = xgm[x_id];                                                 \
    name##_op(ygm[y_id], x);                                            \
  }                                                                     \
}

#define abs_op(a,b) SetReal(a, AbsoluteValue(b))

#if PRECISION == 3232 || PRECISION == 6464
  #define neg_op(a,b) a.x = -b.x; a.y = -b.y
#else
  #define neg_op(a,b) a = -b
#endif

#if defined(CUDA) || INTEGER_PRECISION
  #define sign(x) ((ZERO<(x)) - ((x)<ZERO))
#endif
#define sign_op(a,b) SetReal(a, sign(GetReal(b)))

#define square_op(c,a)  Multiply(c,a,a)

TRANSFORM(abs)
TRANSFORM(neg)
TRANSFORM(sign)
TRANSFORM(square)

#if !INTEGER_PRECISION

#if PRECISION == 3232 || PRECISION == 6464

#define floor_op(c,a)   c.x = floor(a.x); c.y = floor(a.y)
#define ceil_op(c,a)    c.x = ceil(a.x); c.y = ceil(a.y)
#define round_op(c,a)   c.x = round(a.x); c.y = round(a.y)
#define conj_op(c,a)    c.x = a.x; c.y = -a.y
#define sqrt_op(c,a)    c = zsqrt(a)
#define exp_op(c,a)     c = zexp(a)
#define log_op(c,a)     c = zlog(a)
#define sin_op(c,a)     c = zsin(a)
#define cos_op(c,a)     c = zcos(a)
#define tan_op(c,a)     c = ztan(a)
#define asin_op(c,a)    c = zasin(a)
#define acos_op(c,a)    c = zacos(a)
#define atan_op(c,a)    c = zatan(a)
#define sinh_op(c,a)    c = zsinh(a)
#define cosh_op(c,a)    c = zcosh(a)
#define tanh_op(c,a)    c = ztanh(a)
#define asinh_op(c,a)   c = zasinh(a)
#define acosh_op(c,a)   c = zacosh(a)
#define atanh_op(c,a)   c = zatanh(a)

#define reciprocal_op(c,a)              \
  do {                                  \
    singlereal d = a.x*a.x + a.y*a.y;   \
    c.x = a.x / d; c.y = -a.y / d;      \
  } while (0)

#define sigmoid_op(c,a)                 \
  do {                                  \
    real z;                             \
    z.x = -a.x;                         \
    z.y = -a.y;                         \
    z = zexp(z);                        \
    z.x += ONE;                         \
    reciprocal_op(c,z);                 \
  } while (0)

#define erf_op(c,a) SetReal(c, erf(GetReal(a))) /*FIXME*/

#else // !(PRECISION == 3232 || PRECISION == 6464)

#define floor_op(c,a)   c = floor(a)
#define ceil_op(c,a)    c = ceil(a)
#define round_op(c,a)   c = round(a)
#define conj_op(c,a)    c = a
#define sqrt_op(c,a)    c = sqrt(a)
#define exp_op(c,a)     c = exp(a)
#define log_op(c,a)     c = log(a)
#define sin_op(c,a)     c = sin(a)
#define cos_op(c,a)     c = cos(a)
#define tan_op(c,a)     c = tan(a)
#define asin_op(c,a)    c = asin(a)
#define acos_op(c,a)    c = acos(a)
#define atan_op(c,a)    c = atan(a)
#define sinh_op(c,a)    c = sinh(a)
#define cosh_op(c,a)    c = cosh(a)
#define tanh_op(c,a)    c = tanh(a)
#define asinh_op(c,a)   c = asinh(a)
#define acosh_op(c,a)   c = acosh(a)
#define atanh_op(c,a)   c = atanh(a)
#define erf_op(c,a)     c = erf(a)

#define reciprocal_op(c,a)  c = ONE / a
#define sigmoid_op(c,a)     c = (ONE/(ONE+exp(-a)))

#endif // !(PRECISION == 3232 || PRECISION == 6464)

TRANSFORM(floor)
TRANSFORM(ceil)
TRANSFORM(round)
TRANSFORM(conj)
TRANSFORM(sqrt)
TRANSFORM(exp)
TRANSFORM(log)
TRANSFORM(sin)
TRANSFORM(cos)
TRANSFORM(tan)
TRANSFORM(asin)
TRANSFORM(acos)
TRANSFORM(atan)
TRANSFORM(sinh)
TRANSFORM(cosh)
TRANSFORM(tanh)
TRANSFORM(asinh)
TRANSFORM(acosh)
TRANSFORM(atanh)
TRANSFORM(erf)
TRANSFORM(reciprocal)
TRANSFORM(sigmoid)

#endif // !INTEGER_PRECISION

)" // End of the C++11 raw string literal
