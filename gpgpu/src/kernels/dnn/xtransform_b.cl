// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
PROGRAM_STRING_DEBUG_INFO R"(

#if CUDA || PRECISION == 3232 || PRECISION == 6464
  #if VW == 1
    #define OPV(op,cvec,avec,bvec) \
      op(cvec, avec, bvec);
  #elif VW == 2
    #define OPV(op,cvec,avec,bvec) \
      op(cvec.x, avec.x, bvec.x); \
      op(cvec.y, avec.y, bvec.y);
  #elif VW == 4
    #define OPV(op,cvec,avec,bvec) \
      op(cvec.x, avec.x, bvec.x); \
      op(cvec.y, avec.y, bvec.y); \
      op(cvec.z, avec.z, bvec.z); \
      op(cvec.w, avec.w, bvec.w);
  #elif VW == 8
    #define OPV(op,cvec,avec,bvec) \
      op(cvec.s0, avec.s0, bvec.s0); \
      op(cvec.s1, avec.s1, bvec.s1); \
      op(cvec.s2, avec.s2, bvec.s2); \
      op(cvec.s3, avec.s3, bvec.s3); \
      op(cvec.s4, avec.s4, bvec.s4); \
      op(cvec.s5, avec.s5, bvec.s5); \
      op(cvec.s6, avec.s6, bvec.s6); \
      op(cvec.s7, avec.s7, bvec.s7);
  #elif VW == 16
    #define OPV(op,cvec,avec,bvec) \
      op(cvec.s0, avec.s0, bvec.s0); \
      op(cvec.s1, avec.s1, bvec.s1); \
      op(cvec.s2, avec.s2, bvec.s2); \
      op(cvec.s3, avec.s3, bvec.s3); \
      op(cvec.s4, avec.s4, bvec.s4); \
      op(cvec.s5, avec.s5, bvec.s5); \
      op(cvec.s6, avec.s6, bvec.s6); \
      op(cvec.s7, avec.s7, bvec.s7); \
      op(cvec.s8, avec.s8, bvec.s8); \
      op(cvec.s9, avec.s9, bvec.s9); \
      op(cvec.sA, avec.sA, bvec.sA); \
      op(cvec.sB, avec.sB, bvec.sB); \
      op(cvec.sC, avec.sC, bvec.sC); \
      op(cvec.sD, avec.sD, bvec.sD); \
      op(cvec.sE, avec.sE, bvec.sE); \
      op(cvec.sF, avec.sF, bvec.sF);
  #endif
#else
  #define OPV(op,cvec,avec,bvec) op(cvec,avec,bvec)
#endif

#define DEFINE_BINARY(name, op)                                             \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))                   \
void name(                                                                  \
  const int x_size, const __global real* restrict xgm, const int x_offset,  \
  const int y_size, const __global real* restrict ygm, const int y_offset,  \
  __global real *zgm, const int z_offset)                                   \
{                                                                           \
  for (int id = get_global_id(0); id<x_size; id += get_global_size(0)) {    \
    real x_value = xgm[id + x_offset];                                      \
    real y_value = ygm[id + y_offset];                                      \
    real z_value;                                                           \
    op(z_value, x_value, y_value);                                          \
    zgm[id + z_offset] = z_value;                                           \
  }                                                                         \
}                                                                           \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))                   \
void name##ExpandL(                                                         \
  const int x_size, const __global real* restrict xgm, const int x_offset,  \
  const int y_size, const __global real* restrict ygm, const int y_offset,  \
  __global real *zgm, const int z_offset)                                   \
{                                                                           \
  real x_value = xgm[x_offset];                                             \
  for (int id = get_global_id(0); id<y_size; id += get_global_size(0)) {    \
    real y_value = ygm[id + y_offset];                                      \
    real z_value;                                                           \
    op(z_value, x_value, y_value);                                          \
    zgm[id + z_offset] = z_value;                                           \
  }                                                                         \
}                                                                           \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))                   \
void name##ExpandR(                                                         \
  const int x_size, const __global real* restrict xgm, const int x_offset,  \
  const int y_size, const __global real* restrict ygm, const int y_offset,  \
  __global real *zgm, const int z_offset)                                   \
{                                                                           \
  real y_value = ygm[y_offset];                                             \
  for (int id = get_global_id(0); id<x_size; id += get_global_size(0)) {    \
    real x_value = xgm[id + x_offset];                                      \
    real z_value;                                                           \
    op(z_value, x_value, y_value);                                          \
    zgm[id + z_offset] = z_value;                                           \
  }                                                                         \
}                                                                           \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))                   \
void name##RepeatL(                                                         \
  const int x_size, const __global real* restrict xgm, const int x_offset,  \
  const int y_size, const __global real* restrict ygm, const int y_offset,  \
  __global real *zgm, const int z_offset)                                   \
{                                                                           \
  for (int id = get_global_id(0); id<y_size; id += get_global_size(0)) {    \
    real x_value = xgm[id % x_size + x_offset];                             \
    real y_value = ygm[id + y_offset];                                      \
    real z_value;                                                           \
    op(z_value, x_value, y_value);                                          \
    zgm[id + z_offset] = z_value;                                           \
  }                                                                         \
}                                                                           \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))                   \
void name##RepeatR(                                                         \
  const int x_size, const __global real* restrict xgm, const int x_offset,  \
  const int y_size, const __global real* restrict ygm, const int y_offset,  \
  __global real *zgm, const int z_offset)                                   \
{                                                                           \
  for (int id = get_global_id(0); id<x_size; id += get_global_size(0)) {    \
    real x_value = xgm[id + x_offset];                                      \
    real y_value = ygm[id % y_size + y_offset];                             \
    real z_value;                                                           \
    op(z_value, x_value, y_value);                                          \
    zgm[id + z_offset] = z_value;                                           \
  }                                                                         \
}                                                                           \
                                                                            \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))                   \
void name##Strided(const int n, const int rank, __constant int* shape,      \
                   const __global real* restrict xgm, const int x_offset,   \
                   const __global real* restrict ygm, const int y_offset,   \
                   __global real* zgm, const int z_offset)                  \
{                                                                           \
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {       \
    int x_id = x_offset, y_id = y_offset, z_id = z_offset;                  \
    unravel3(id, &x_id, &y_id, &z_id, rank, shape);                         \
    real x_value = xgm[x_id], y_value = ygm[y_id];                          \
    real z_value;                                                           \
    op(z_value, x_value, y_value);                                          \
    zgm[z_id] = z_value;                                                    \
  }                                                                         \
}                                                                           \
                                                                            \
__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))     \
void name##Channel(const int m, const int n, const int channels,            \
                   const __global real* restrict xgm, const int x_offset,   \
                   const __global real* restrict ygm, const int y_offset,   \
                   __global real* zgm, const int z_offset)                  \
{                                                                           \
  const int rid = get_global_id(0);                                         \
  if (rid < m) {                                                            \
    const real y = ygm[rid % channels + y_offset];                          \
    for (int id = get_global_id(1); id < n; id += get_global_size(1)) {     \
      const int offset = rid*n + id;                                        \
      real x = xgm[offset + x_offset];                                      \
      real z;                                                               \
      op(z, x, y);                                                          \
      zgm[offset + z_offset] = z;                                           \
    }                                                                       \
  }                                                                         \
}

#define DEFINE_BINARY_V(name, op)                                           \
DEFINE_BINARY(name, op)                                                     \
                                                                            \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))                   \
void name##Faster(const int n,                                              \
    const __global realV* restrict xgm, const __global realV* restrict ygm, \
    __global realV* zgm)                                                    \
{                                                                           \
  if (get_global_id(0) < n/VW) {                                            \
    _Pragma("unroll")                                                       \
    for (int _w = 0; _w < WPT; _w++) {                                      \
      const int id = _w*get_global_size(0) + get_global_id(0);              \
      realV xvec = xgm[id];                                                 \
      realV yvec = ygm[id];                                                 \
      realV zvec;                                                           \
      OPV(op, zvec, xvec, yvec);                                            \
      zgm[id] = zvec;                                                       \
    }                                                                       \
  }                                                                         \
}                                                                           \
                                                                            \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))                   \
void name##Fastest(const int n,                                             \
    const __global realV* restrict xgm, const __global realV* restrict ygm, \
    __global realV* zgm)                                                    \
{                                                                           \
  _Pragma("unroll")                                                         \
  for (int _w = 0; _w < WPT; _w++) {                                        \
    const int id = _w*get_global_size(0) + get_global_id(0);                \
    realV xvec = xgm[id];                                                   \
    realV yvec = ygm[id];                                                   \
    realV zvec;                                                             \
    OPV(op, zvec, xvec, yvec);                                              \
    zgm[id] = zvec;                                                         \
  }                                                                         \
}

#define DEFINE_RELATION(name, op)                                           \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))                   \
void name(                                                                  \
  const int x_size, const __global real* restrict xgm, const int x_offset,  \
  const int y_size, const __global real* restrict ygm, const int y_offset,  \
  __global char* zgm, const int z_offset)                                   \
{                                                                           \
  for (int id = get_global_id(0); id<x_size; id += get_global_size(0)) {    \
    zgm[id + z_offset] = xgm[id + x_offset] op ygm[id + y_offset];          \
  }                                                                         \
}                                                                           \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))                   \
void name##ExpandL(                                                         \
  const int x_size, const __global real* restrict xgm, const int x_offset,  \
  const int y_size, const __global real* restrict ygm, const int y_offset,  \
  __global char* zgm, const int z_offset)                                   \
{                                                                           \
  real x_value = xgm[x_offset];                                             \
  for (int id = get_global_id(0); id<y_size; id += get_global_size(0)) {    \
    zgm[id + z_offset] = x_value op ygm[id + y_offset];                     \
  }                                                                         \
}                                                                           \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))                   \
void name##ExpandR(                                                         \
  const int x_size, const __global real* restrict xgm, const int x_offset,  \
  const int y_size, const __global real* restrict ygm, const int y_offset,  \
  __global char* zgm, const int z_offset)                                   \
{                                                                           \
  real y_value = ygm[y_offset];                                             \
  for (int id = get_global_id(0); id<x_size; id += get_global_size(0)) {    \
    zgm[id + z_offset] = xgm[id + x_offset] op y_value;                     \
  }                                                                         \
}                                                                           \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))                   \
void name##RepeatL(                                                         \
  const int x_size, const __global real* restrict xgm, const int x_offset,  \
  const int y_size, const __global real* restrict ygm, const int y_offset,  \
  __global char* zgm, const int z_offset)                                   \
{                                                                           \
  for (int id = get_global_id(0); id<y_size; id += get_global_size(0)) {    \
    zgm[id + z_offset] = xgm[id % x_size + x_offset] op ygm[id + y_offset]; \
  }                                                                         \
}                                                                           \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))                   \
void name##RepeatR(                                                         \
  const int x_size, const __global real* restrict xgm, const int x_offset,  \
  const int y_size, const __global real* restrict ygm, const int y_offset,  \
  __global char* zgm, const int z_offset)                                   \
{                                                                           \
  for (int id = get_global_id(0); id<x_size; id += get_global_size(0)) {    \
    zgm[id + z_offset] = xgm[id + x_offset] op ygm[id % y_size + y_offset]; \
  }                                                                         \
}                                                                           \
                                                                            \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))                   \
void name##Strided(const int n, const int rank, __constant int* shape,      \
                   const __global real* restrict xgm, const int x_offset,   \
                   const __global real* restrict ygm, const int y_offset,   \
                   __global char* zgm, const int z_offset)                  \
{                                                                           \
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {       \
    int x_id = x_offset, y_id = y_offset, z_id = z_offset;                  \
    unravel3(id, &x_id, &y_id, &z_id, rank, shape);                         \
    zgm[z_id] = xgm[x_id] op ygm[y_id];                                     \
  }                                                                         \
}                                                                           \
                                                                            \
__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))     \
void name##Channel(const int m, const int n, const int channels,            \
                   const __global real* restrict xgm, const int x_offset,   \
                   const __global real* restrict ygm, const int y_offset,   \
                   __global char* zgm, const int z_offset)                  \
{                                                                           \
  const int rid = get_global_id(0);                                         \
  if (rid < m) {                                                            \
    const real y = ygm[rid % channels + y_offset];                          \
    for (int id = get_global_id(1); id < n; id += get_global_size(1)) {     \
      const int offset = rid*n + id;                                        \
      zgm[offset + z_offset] = xgm[offset + x_offset] op y;                 \
    }                                                                       \
  }                                                                         \
}

DEFINE_BINARY_V(Xadd_v, Add)
DEFINE_BINARY_V(Xsub_v, Subtract)
DEFINE_BINARY_V(Xmul_v, Multiply)
DEFINE_BINARY_V(Xdiv_v, DivideFull)

#if PRECISION == 3232 || PRECISION == 6464
  INLINE_FUNC real xpow(real a, real b) {
    // pow(a,b) = exp(b * log(a))
    real z, t;
    t.x = log(hypot(a.x, a.y));
    t.y = atan2(a.y, a.x);
    Multiply(z, b, t);

    singlereal e = exp(z.x);
    z.x = e * cos(z.y);
    z.y = e * sin(z.y);
    return z;
  }
#elif INTEGER_PRECISION
  #define xpow(x,y) pow((float)x,(float)y)
#else
  #define xpow pow
#endif

#define Pow(c,a,b) c = xpow(a,b)
DEFINE_BINARY(Xpow, Pow)

#if PRECISION != 3232 && PRECISION != 6464
  DEFINE_BINARY(Xmax, Max)
  DEFINE_BINARY(Xmin, Min)

  #if INTEGER_PRECISION
    #define Mod(c,a,b) c = a % b
  #else
    #define Mod(c,a,b) c = fmod(a, b)
  #endif
  DEFINE_BINARY(Xmod, Mod)

  #define PRelu(c,a,b) c = a<ZERO ? a*b : a
  DEFINE_BINARY(Xprelu, PRelu)

  DEFINE_RELATION(Xequal_to, ==)
  DEFINE_RELATION(Xnot_equal_to, !=)
  DEFINE_RELATION(Xless, <)
  DEFINE_RELATION(Xless_equal, <=)
  DEFINE_RELATION(Xgreater, >)
  DEFINE_RELATION(Xgreater_equal, >=)
#endif

#if INTEGER_PRECISION
  #define BitAnd(c,a,b) c = a & b;
  #define BitOr(c,a,b)  c = a | b;
  #define BitXor(c,a,b) c = a ^ b;

  DEFINE_BINARY(Xbit_and, BitAnd)
  DEFINE_BINARY(Xbit_or,  BitOr)
  DEFINE_BINARY(Xbit_xor, BitXor)
#endif

)" // End of the C++11 raw string literal
