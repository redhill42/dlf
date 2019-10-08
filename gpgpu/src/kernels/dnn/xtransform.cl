// Enable loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
PROGRAM_STRING_DEBUG_INFO R"(

//---------------------------------------------------------------------------

#if defined(ROUTINE_abs)
#  define OP(c,a) SetReal(c, AbsoluteValue(a))
#elif defined(ROUTINE_neg)
  #if PRECISION == 3232 || PRECISION == 6464
  #  define OP(c,a) c.x = -a.x; c.y = -a.y
  #else
  #  define OP(c,a) c = -a
  #endif
#elif defined(ROUTINE_sign)
  #if defined(CUDA) || INTEGER_PRECISION
  #  define sign(x) ((ZERO<(x)) - ((x)<ZERO))
  #endif
  #define OP(c,a) SetReal(c, sign(GetReal(a)))
#elif defined(ROUTINE_square)
#  define OP(c,a) Multiply(c,a,a)
#endif

#if !INTEGER_PRECISION
#if PRECISION == 3232 || PRECISION == 6464

#if defined(ROUTINE_floor)
#  define OP(c,a)   c.x = floor(a.x); c.y = floor(a.y)
#elif defined(ROUTINE_ceil)
#  define OP(c,a)   c.x = ceil(a.x); c.y = ceil(a.y)
#elif defined(ROUTINE_round)
#  define OP(c,a)   c.x = round(a.x); c.y = round(a.y)
#elif defined(ROUTINE_conj)
#  define OP(c,a)   c.x = a.x; c.y = -a.y
#elif defined(ROUTINE_sqrt)
#  define OP(c,a)   c = zsqrt(a)
#elif defined(ROUTINE_exp)
#  define OP(c,a)   c = zexp(a)
#elif defined(ROUTINE_log)
#  define OP(c,a)   c = zlog(a)
#elif defined(ROUTINE_sin)
#  define OP(c,a)   c = zsin(a)
#elif defined(ROUTINE_cos)
#  define OP(c,a)   c = zcos(a)
#elif defined(ROUTINE_tan)
#  define OP(c,a)   c = ztan(a)
#elif defined(ROUTINE_asin)
#  define OP(c,a)   c = zasin(a)
#elif defined(ROUTINE_acos)
#  define OP(c,a)   c = zacos(a)
#elif defined(ROUTINE_atan)
#  define OP(c,a)   c = zatan(a)
#elif defined(ROUTINE_sinh)
#  define OP(c,a)   c = zsinh(a)
#elif defined(ROUTINE_cosh)
#  define OP(c,a)   c = zcosh(a)
#elif defined(ROUTINE_tanh)
#  define OP(c,a)   c = ztanh(a)
#elif defined(ROUTINE_asinh)
#  define OP(c,a)   c = zasinh(a)
#elif defined(ROUTINE_acosh)
#  define OP(c,a)   c = zacosh(a)
#elif defined(ROUTINE_atanh)
#  define OP(c,a)   c = zatanh(a)
#elif defined(ROUTINE_erf)
#  define OP(c,a) SetReal(c, erf(GetReal(a))) /*FIXME*/
#elif defined(ROUTINE_reciprocal)
  #define OP(c,a)                           \
     do {                                   \
       singlereal d = a.x*a.x + a.y*a.y;    \
       c.x = a.x/d; c.y = -a.y/d;           \
     } while(0)
#elif defined(ROUTINE_sigmoid)
  #define OP(c,a)                           \
     do {                                   \
       real z;                              \
       z.x = -a.x; z.y = -a.y;              \
       z = zexp(z);                         \
       z.x += ONE;                          \
       singlereal d = z.x*z.x + z.y*z.y;    \
       c.x = z.x/d; c.y = -z.y/d;           \
     } while(0)
#elif defined(ROUTINE_softsign)
  #define OP(c,a)                           \
     do {                                   \
       real t = hypot(a.x, a.y) + ONE;      \
       c.x = a.x/t; c.y = a.y/t;            \
     } while(0)
#elif defined(ROUTINE_softplus)
  #define OP(c,a)                           \
     do {                                   \
       real t = zexp(a);                    \
       t.x += ONE;                          \
       c = zlog(t);                         \
     } while(0)
#endif

#else // !(PRECISION == 3232 || PRECISION == 6464)

#if defined(ROUTINE_floor)
#  define OP(c,a)   c = floor(a)
#elif defined(ROUTINE_ceil)
#  define OP(c,a)   c = ceil(a)
#elif defined(ROUTINE_round)
#  define OP(c,a)   c = round(a)
#elif defined(ROUTINE_conj)
#  define OP(c,a)   c = a
#elif defined(ROUTINE_sqrt)
#  define OP(c,a)   c = sqrt(a)
#elif defined(ROUTINE_exp)
#  define OP(c,a)   c = exp(a)
#elif defined(ROUTINE_log)
#  define OP(c,a)   c = log(a)
#elif defined(ROUTINE_sin)
#  define OP(c,a)   c = sin(a)
#elif defined(ROUTINE_cos)
#  define OP(c,a)   c = cos(a)
#elif defined(ROUTINE_tan)
#  define OP(c,a)   c = tan(a)
#elif defined(ROUTINE_asin)
#  define OP(c,a)   c = asin(a)
#elif defined(ROUTINE_acos)
#  define OP(c,a)   c = acos(a)
#elif defined(ROUTINE_atan)
#  define OP(c,a)   c = atan(a)
#elif defined(ROUTINE_sinh)
#  define OP(c,a)   c = sinh(a)
#elif defined(ROUTINE_cosh)
#  define OP(c,a)   c = cosh(a)
#elif defined(ROUTINE_tanh)
#  define OP(c,a)   c = tanh(a)
#elif defined(ROUTINE_asinh)
#  define OP(c,a)   c = asinh(a)
#elif defined(ROUTINE_acosh)
#  define OP(c,a)   c = acosh(a)
#elif defined(ROUTINE_atanh)
#  define OP(c,a)   c = atanh(a)
#elif defined(ROUTINE_erf)
#  define OP(c,a)   c = erf(a)
#elif defined(ROUTINE_reciprocal)
#  define OP(c,a)   c = ONE / a
#elif defined(ROUTINE_sigmoid)
#  define OP(c,a)   c = (ONE/(ONE+exp(-a)))
#elif defined(ROUTINE_softsign)
#  define OP(c,a)   c = a/(ONE+fabs(x))
#elif defined(ROUTINE_softplus)
#  define OP(c,a)   c = log(exp(x)+ONE)
#endif

#endif // !(PRECISION == 3232 || PRECISION == 6464
#endif // !INTEGER_PRECISION

//---------------------------------------------------------------------------

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xtransform(
    const int n,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
    ygm = &ygm[y_offset];
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
        real x = xgm[id + x_offset];
        OP(ygm[id], x);
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XtransformStrided(
    const int n, const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
        int x_id = x_offset, y_id = y_offset;
        unravel2(id, &x_id, &y_id, rank, shape);
        real x = xgm[x_id];
        OP(ygm[y_id], x);
    }
}

)" // End of the C++11 raw string literal
