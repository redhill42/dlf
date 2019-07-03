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
void name(const int x_size, const __global real* restrict xgm, const int x_offset, const int x_inc, \
          const int y_size, const __global real* restrict ygm, const int y_offset, const int y_inc, \
          __global real *zgm, const int z_offset, const int z_inc)          \
{                                                                           \
  if (x_size == 1) {                                                        \
    real x_value = xgm[x_offset];                                           \
    for (int id = get_global_id(0); id < y_size; id += get_global_size(0)) {\
      real y_value = ygm[id*y_inc + y_offset];                              \
      real z_value;                                                         \
      op(z_value, x_value, y_value);                                        \
      zgm[id*z_inc + z_offset] = z_value;                                   \
    }                                                                       \
  } else if (y_size == 1) {                                                 \
    real y_value = ygm[y_offset];                                           \
    for (int id = get_global_id(0); id < x_size; id += get_global_size(0)) {\
      real x_value = xgm[id*x_inc + x_offset];                              \
      real z_value;                                                         \
      op(z_value, x_value, y_value);                                        \
      zgm[id*z_inc + z_offset] = z_value;                                   \
    }                                                                       \
  } else if (x_size < y_size) {                                             \
    for (int id = get_global_id(0); id < y_size; id += get_global_size(0)) {\
      real x_value = xgm[(id*x_inc + x_offset) % x_size];                   \
      real y_value = ygm[id*y_inc + y_offset];                              \
      real z_value;                                                         \
      op(z_value, x_value, y_value);                                        \
      zgm[id*z_inc + z_offset] = z_value;                                   \
    }                                                                       \
  } else if (x_size > y_size) {                                             \
    for (int id = get_global_id(0); id < x_size; id += get_global_size(0)) {\
      real x_value = xgm[id*x_inc + x_offset];                              \
      real y_value = ygm[(id*y_inc + y_offset) % y_size];                   \
      real z_value;                                                         \
      op(z_value, x_value, y_value);                                        \
      zgm[id*z_inc + z_offset] = z_value;                                   \
    }                                                                       \
  } else {                                                                  \
    for (int id = get_global_id(0); id < x_size; id += get_global_size(0)) {\
      real x_value = xgm[id*x_inc + x_offset];                              \
      real y_value = ygm[id*y_inc + y_offset];                              \
      real z_value;                                                         \
      op(z_value, x_value, y_value);                                        \
      zgm[id*z_inc + z_offset] = z_value;                                   \
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

#ifdef CUDA
  #if PRECISION == 16
    #define xpow hpow
  #elif PRECISION == 32
    #define xpow powf
  #elif PRECISION == 64
    #define xpow pow
  #else
    #define xpow(x,y) pow((float)x,(float)y)
  #endif
#else
  #if INTEGER_PRECISION
    #define xpow(x,y) pow((float)x,(float)y)
  #else
    #define xpow pow
  #endif
#endif
#define Pow(c,a,b) c = xpow(a,b)
DEFINE_BINARY(Xpow, Pow)

#define PRelu(c,a,b) c = a<ZERO ? a*b : a
DEFINE_BINARY(Xprelu, PRelu)

#endif

)" // End of the C++11 raw string literal
