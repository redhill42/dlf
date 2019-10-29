CL_PROGRAM R"(

typedef real realX;
typedef real realY;

#if defined(ROUTINE_equal_to) || defined(ROUTINE_not_equal_to) || \
    defined(ROUTINE_less)     || defined(ROUTINE_less_equal)   || \
    defined(ROUTINE_greater)  || defined(ROUTINE_greater_equal)
  typedef char realZ;
#else
  typedef real realZ;
#endif

#if defined(ROUTINE_add)
#  define OP Add
#elif defined(ROUTINE_sub)
#  define OP Subtract
#elif defined(ROUTINE_mul)
#  define OP Multiply
#elif defined(ROUTINE_div)
#  define OP DivideFull
#endif

#if defined(ROUTINE_pow)
  #if PRECISION == 3232 || PRECISION == 6464
     // pow(a,b) = exp(b * log(a))
    #define OP(c,a,b)                   \
      do {                              \
        real t;                         \
        t.x = log(hypot(a.x, a.y));     \
        t.y = atan2(a.y, a.x);          \
        Multiply(c, b, t);              \
                                        \
        singlereal e = exp(c.x);        \
        c.x = e * cos(c.y);             \
        c.y = e * sin(c.y);             \
      } while(0)
  #elif INTEGER_PRECISION
  #  define OP(c,a,b) c = (real)pow((float)x,(float)y)
  #else
  #  define OP(c,a,b) c = pow(a,b)
  #endif
#endif

#if PRECISION != 3232 && PRECISION != 6464
  #if defined(ROUTINE_max)
  #  define OP Max
  #elif defined(ROUTINE_min)
  #  define OP Min
  #elif defined(ROUTINE_mod)
    #if INTEGER_PRECISION
    #  define OP(c,a,b) c = a % b
    #else
    #  define OP(c,a,b) c = fmod(a, b)
    #endif
  #elif defined(ROUTINE_prelu)
  #  define OP(c,a,b) c = a<ZERO ? a*b : a
  #endif

  #if defined(ROUTINE_equal_to)
  #  define OP(c,a,b) c = a==b
  #elif defined(ROUTINE_not_equal_to)
  #  define OP(c,a,b) c = a!=b
  #elif defined(ROUTINE_less)
  #  define OP(c,a,b) c = a<b
  #elif defined(ROUTINE_less_equal)
  #  define OP(c,a,b) c = a<=b
  #elif defined(ROUTINE_greater)
  #  define OP(c,a,b) c = a>b
  #elif defined(ROUTINE_greater_equal)
  #  define OP(c,a,b) c = a>=b
  #endif
#endif

#if INTEGER_PRECISION
  #if defined(ROUTINE_bit_and)
  #  define OP(c,a,b) c = a & b
  #elif defined(ROUTINE_bit_or)
  #  define OP(c,a,b) c = a | b
  #elif defined(ROUTINE_bit_xor)
  #  define OP(c,a,b) c = a ^ b;
  #endif
#endif

//---------------------------------------------------------------------------

#if CUDA || PRECISION == 3232 || PRECISION == 6464
  #if VW == 1
    #define OPV(cvec,avec,bvec) \
      OP(cvec, avec, bvec);
  #elif VW == 2
    #define OPV(cvec,avec,bvec) \
      OP(cvec.x, avec.x, bvec.x); \
      OP(cvec.y, avec.y, bvec.y);
  #elif VW == 4
    #define OPV(cvec,avec,bvec) \
      OP(cvec.x, avec.x, bvec.x); \
      OP(cvec.y, avec.y, bvec.y); \
      OP(cvec.z, avec.z, bvec.z); \
      OP(cvec.w, avec.w, bvec.w);
  #elif VW == 8
    #define OPV(cvec,avec,bvec) \
      OP(cvec.s0, avec.s0, bvec.s0); \
      OP(cvec.s1, avec.s1, bvec.s1); \
      OP(cvec.s2, avec.s2, bvec.s2); \
      OP(cvec.s3, avec.s3, bvec.s3); \
      OP(cvec.s4, avec.s4, bvec.s4); \
      OP(cvec.s5, avec.s5, bvec.s5); \
      OP(cvec.s6, avec.s6, bvec.s6); \
      OP(cvec.s7, avec.s7, bvec.s7);
  #elif VW == 16
    #define OPV(cvec,avec,bvec) \
      OP(cvec.s0, avec.s0, bvec.s0); \
      OP(cvec.s1, avec.s1, bvec.s1); \
      OP(cvec.s2, avec.s2, bvec.s2); \
      OP(cvec.s3, avec.s3, bvec.s3); \
      OP(cvec.s4, avec.s4, bvec.s4); \
      OP(cvec.s5, avec.s5, bvec.s5); \
      OP(cvec.s6, avec.s6, bvec.s6); \
      OP(cvec.s7, avec.s7, bvec.s7); \
      OP(cvec.s8, avec.s8, bvec.s8); \
      OP(cvec.s9, avec.s9, bvec.s9); \
      OP(cvec.sA, avec.sA, bvec.sA); \
      OP(cvec.sB, avec.sB, bvec.sB); \
      OP(cvec.sC, avec.sC, bvec.sC); \
      OP(cvec.sD, avec.sD, bvec.sD); \
      OP(cvec.sE, avec.sE, bvec.sE); \
      OP(cvec.sF, avec.sF, bvec.sF);
  #endif
#else
  #define OPV(cvec,avec,bvec) OP(cvec,avec,bvec)
#endif

//---------------------------------------------------------------------------

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xtransform(
    const int x_size, const __global realX* restrict xgm, const int x_offset,
    const int y_size, const __global realY* restrict ygm, const int y_offset,
    __global realZ* zgm, const int z_offset)
{
    for (int id = get_global_id(0); id < x_size; id += get_global_size(0)) {
        realX x_value = xgm[id + x_offset];
        realY y_value = ygm[id + y_offset];
        realZ z_value;
        OP(z_value, x_value, y_value);
        zgm[id + z_offset] = z_value;
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XtransformExpandL(
    const int x_size, const __global realX* restrict xgm, const int x_offset,
    const int y_size, const __global realY* restrict ygm, const int y_offset,
    __global realZ* zgm, const int z_offset)
{
    realX x_value = xgm[x_offset];
    for (int id = get_global_id(0); id < y_size; id += get_global_size(0)) {
        realY y_value = ygm[id + y_offset];
        realZ z_value;
        OP(z_value, x_value, y_value);
        zgm[id + z_offset] = z_value;
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XtransformExpandR(
    const int x_size, const __global realX* restrict xgm, const int x_offset,
    const int y_size, const __global realY* restrict ygm, const int y_offset,
    __global realZ* zgm, const int z_offset)
{
    realY y_value = ygm[y_offset];
    for (int id = get_global_id(0); id < x_size; id += get_global_size(0)) {
        realX x_value = xgm[id + x_offset];
        realZ z_value;
        OP(z_value, x_value, y_value);
        zgm[id + z_offset] = z_value;
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XtransformRepeatL(
    const int x_size, const __global realX* restrict xgm, const int x_offset,
    const int y_size, const __global realY* restrict ygm, const int y_offset,
    __global realZ* zgm, const int z_offset)
{
    for (int id = get_global_id(0); id < y_size; id += get_global_size(0)) {
        realX x_value = xgm[id % x_size + x_offset];
        realY y_value = ygm[id + y_offset];
        realZ z_value;
        OP(z_value, x_value, y_value);
        zgm[id + z_offset] = z_value;
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XtransformRepeatR(
    const int x_size, const __global realX* restrict xgm, const int x_offset,
    const int y_size, const __global realY* restrict ygm, const int y_offset,
    __global realZ* zgm, const int z_offset)
{
    for (int id = get_global_id(0); id < x_size; id += get_global_size(0)) {
        realX x_value = xgm[id + x_offset];
        realY y_value = ygm[id % y_size + y_offset];
        realZ z_value;
        OP(z_value, x_value, y_value);
        zgm[id + z_offset] = z_value;
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XtransformStrided(
    const int n, const int rank, __constant int* shape,
    const __global realX* restrict xgm, const int x_offset,
    const __global realY* restrict ygm, const int y_offset,
    __global realZ* zgm, const int z_offset)
{
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
        int x_id = x_offset, y_id = y_offset, z_id = z_offset;
        unravel3(id, &x_id, &y_id, &z_id, rank, shape);
        realX x_value = xgm[x_id];
        realY y_value = ygm[y_id];
        realZ z_value;
        OP(z_value, x_value, y_value);
        zgm[z_id] = z_value;
    }
}

__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
void XtransformChannel(
    const int m, const int n, const int channels,
    const __global realX* restrict xgm, const int x_offset,
    const __global realY* restrict ygm, const int y_offset,
    __global realZ* zgm, const int z_offset)
{
    const int rid = get_global_id(0);
    if (rid < m) {
        const realY y = ygm[rid % channels + y_offset];
        for (int id = get_global_id(1); id < n; id += get_global_size(1)) {
            const int offset = rid*n + id;
            realX x = xgm[offset + x_offset];
            realZ z;
            OP(z, x, y);
            zgm[offset + z_offset] = z;
        }
    }
}

#if defined(ROUTINE_add) || defined(ROUTINE_sub) || defined(ROUTINE_mul) || defined(ROUTINE_div)
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XtransformFaster(const int n,
    const __global realV* restrict xgm, const __global realV* restrict ygm,
    __global realV* zgm)
{
    if (get_global_id(0) < n/VW) {
        #pragma("unroll")
        for (int _w = 0; _w < WPT; _w++) {
            const int id = _w*get_global_size(0) + get_global_id(0);
            realV xvec = xgm[id];
            realV yvec = ygm[id];
            realV zvec;
            OPV(zvec, xvec, yvec);
            zgm[id] = zvec;
        }
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XtransformFastest(const int n,
    const __global realV* restrict xgm, const __global realV* restrict ygm,
    __global realV* zgm)
{
    #pragma("unroll")
    for (int _w = 0; _w < WPT; _w++) {
        const int id = _w*get_global_size(0) + get_global_id(0);
        realV xvec = xgm[id];
        realV yvec = ygm[id];
        realV zvec;
        OPV(zvec, xvec, yvec);
        zgm[id] = zvec;
    }
}
#endif

)"
