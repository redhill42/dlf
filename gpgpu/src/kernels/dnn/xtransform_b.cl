// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#line 5

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
void name(const int x_size, const __global real* restrict xgm,              \
          const int y_size, const __global real* restrict ygm,              \
          __global real *zgm)                                               \
{                                                                           \
  if (x_size == 1) {                                                        \
    real x_value = xgm[0];                                                  \
    for (int id = get_global_id(0); id<y_size; id += get_global_size(0)) {  \
      real y_value = ygm[id];                                               \
      real z_value;                                                         \
      op(z_value, x_value, y_value);                                        \
      zgm[id] = z_value;                                                    \
    }                                                                       \
  } else if (y_size == 1) {                                                 \
    real y_value = ygm[0];                                                  \
    for (int id = get_global_id(0); id<x_size; id += get_global_size(0)) {  \
      real x_value = xgm[id];                                               \
      real z_value;                                                         \
      op(z_value, x_value, y_value);                                        \
      zgm[id] = z_value;                                                    \
    }                                                                       \
  } else if (x_size < y_size) {                                             \
    for (int id = get_global_id(0); id<y_size; id += get_global_size(0)) {  \
      real x_value = xgm[id % x_size];                                      \
      real y_value = ygm[id];                                               \
      real z_value;                                                         \
      op(z_value, x_value, y_value);                                        \
      zgm[id] = z_value;                                                    \
    }                                                                       \
  } else if (x_size > y_size) {                                             \
    for (int id = get_global_id(0); id<x_size; id += get_global_size(0)) {  \
      real x_value = xgm[id];                                               \
      real y_value = ygm[id % y_size];                                      \
      real z_value;                                                         \
      op(z_value, x_value, y_value);                                        \
      zgm[id] = z_value;                                                    \
    }                                                                       \
  } else {                                                                  \
    for (int id = get_global_id(0); id<x_size; id += get_global_size(0)) {  \
      real x_value = xgm[id];                                               \
      real y_value = ygm[id];                                               \
      real z_value;                                                         \
      op(z_value, x_value, y_value);                                        \
      zgm[id] = z_value;                                                    \
    }                                                                       \
  }                                                                         \
}                                                                           \
                                                                            \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))                   \
void name##Strided(const int n, const int rank, __constant int* shape,      \
                   const __global real* restrict xgm,                       \
                   const __global real* restrict ygm,                       \
                   __global real* zgm)                                      \
{                                                                           \
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {       \
    real x_value = xgm[unravel(id, rank, &shape[rank], shape)];             \
    real y_value = ygm[unravel(id, rank, &shape[rank*2], shape)];           \
    real z_value;                                                           \
    op(z_value, x_value, y_value);                                          \
    zgm[id] = z_value;                                                      \
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

// The scalar division function
#if PRECISION == 3232 || PRECISION == 6464
  #define Divide(c,a,b) \
    do { \
      singlereal num_x = (a.x * b.x) + (a.y * b.y); \
      singlereal num_y = (a.y * b.x) - (a.x * b.y); \
      singlereal denom = (b.x * b.x) + (b.y * b.y); \
      c.x = num_x / denom; \
      c.y = num_y / denom; \
    } while (0)
#else
  #define Divide(c,a,b) c = a / b
#endif

DEFINE_BINARY_V(Xadd_v, Add)
DEFINE_BINARY_V(Xsub_v, Subtract)
DEFINE_BINARY_V(Xmul_v, Multiply)
DEFINE_BINARY_V(Xdiv_v, Divide)

#if PRECISION !=3232 && PRECISION != 6464

#if INTEGER_PRECISION
  #define xpow(x,y) pow((float)x,(float)y)
  #define xmod(x,y) (x%y)
#elif defined(CUDA) && PRECISION == 32
  #define xpow powf
  #define xmod fmodf
#else
  #define xpow pow
  #define xmod fmod
#endif

#define Max(c,a,b) c = max(a,b)
#define Min(c,a,b) c = min(a,b)
DEFINE_BINARY(Xmax, Max)
DEFINE_BINARY(Xmin, Min)

#define Mod(c,a,b) c = xmod(a,b)
DEFINE_BINARY(Xmod, Mod)

#define Pow(c,a,b) c = xpow(a,b)
DEFINE_BINARY(Xpow, Pow)

#define PRelu(c,a,b) c = a<ZERO ? a*b : a
DEFINE_BINARY(Xprelu, PRelu)

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
