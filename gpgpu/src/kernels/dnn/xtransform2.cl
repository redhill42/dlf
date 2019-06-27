// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// Full version of the kernel with offsets and strided accesses
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Kernel(const int x_size, const __global real* restrict xgm, const int x_offset, const int x_inc,
            const int y_size, const __global real* restrict ygm, const int y_offset, const int y_inc,
            __global real *zgm, const int z_offset, const int z_inc)
{
  if (x_size == 1) {
    real x_value = xgm[x_offset];
    for (int id = get_global_id(0); id < y_size; id += get_global_size(0)) {
      real y_value = ygm[id*y_inc + y_offset];
      real z_value;
      Xform(z_value, x_value, y_value);
      zgm[id*z_inc + z_offset] = z_value;
    }
  } else if (y_size == 1) {
    real y_value = ygm[y_offset];
    for (int id = get_global_id(0); id < x_size; id += get_global_size(0)) {
      real x_value = xgm[id*x_inc + x_offset];
      real z_value;
      Xform(z_value, x_value, y_value);
      zgm[id*z_inc + z_offset] = z_value;
    }
  } else if (x_size < y_size) {
    for (int id = get_global_id(0); id < y_size; id += get_global_size(0)) {
      real x_value = xgm[(id*x_inc + x_offset) % x_size];
      real y_value = ygm[id*y_inc + y_offset];
      real z_value;
      Xform(z_value, x_value, y_value);
      zgm[id*z_inc + z_offset] = z_value;
    }
  } else if (x_size > y_size) {
    for (int id = get_global_id(0); id < x_size; id += get_global_size(0)) {
      real x_value = xgm[id*x_inc + x_offset];
      real y_value = ygm[(id*y_inc + y_offset) % y_size];
      real z_value;
      Xform(z_value, x_value, y_value);
      zgm[id*z_inc + z_offset] = z_value;
    }
  } else {
    for (int id = get_global_id(0); id < x_size; id += get_global_size(0)) {
      real x_value = xgm[id*x_inc + x_offset];
      real y_value = ygm[id*y_inc + y_offset];
      real z_value;
      Xform(z_value, x_value, y_value);
      zgm[id*z_inc + z_offset] = z_value;
    }
  }
}

#if ALLOW_VECTOR

#if CUDA || PRECISION == 3232 || PRECISION == 6464
  #if VW == 1
    #define XformVector(cvec,avec,bvec) \
      Xform(cvec, avec, bvec);
  #elif VW == 2
    #define XformVector(cvec,avec,bvec) \
      Xform(cvec.x, avec.x, bvec.x); \
      Xform(cvec.y, avec.y, bvec.y);
  #elif VW == 4
    #define XformVector(cvec,avec,bvec) \
      Xform(cvec.x, avec.x, bvec.x); \
      Xform(cvec.y, avec.y, bvec.y); \
      Xform(cvec.z, avec.z, bvec.z); \
      Xform(cvec.w, avec.w, bvec.w);
  #elif VW == 8
    #define XformVector(cvec,avec,bvec) \
      Xform(cvec.s0, avec.s0, bvec.s0); \
      Xform(cvec.s1, avec.s1, bvec.s1); \
      Xform(cvec.s2, avec.s2, bvec.s2); \
      Xform(cvec.s3, avec.s3, bvec.s3); \
      Xform(cvec.s4, avec.s4, bvec.s4); \
      Xform(cvec.s5, avec.s5, bvec.s5); \
      Xform(cvec.s6, avec.s6, bvec.s6); \
      Xform(cvec.s7, avec.s7, bvec.s7);
  #elif VW == 16
    #define XformVector(cvec,avec,bvec) \
      Xform(cvec.s0, avec.s0, bvec.s0); \
      Xform(cvec.s1, avec.s1, bvec.s1); \
      Xform(cvec.s2, avec.s2, bvec.s2); \
      Xform(cvec.s3, avec.s3, bvec.s3); \
      Xform(cvec.s4, avec.s4, bvec.s4); \
      Xform(cvec.s5, avec.s5, bvec.s5); \
      Xform(cvec.s6, avec.s6, bvec.s6); \
      Xform(cvec.s7, avec.s7, bvec.s7); \
      Xform(cvec.s8, avec.s8, bvec.s8); \
      Xform(cvec.s9, avec.s9, bvec.s9); \
      Xform(cvec.sA, avec.sA, bvec.sA); \
      Xform(cvec.sB, avec.sB, bvec.sB); \
      Xform(cvec.sC, avec.sC, bvec.sC); \
      Xform(cvec.sD, avec.sD, bvec.sD); \
      Xform(cvec.sE, avec.sE, bvec.sE); \
      Xform(cvec.sF, avec.sF, bvec.sF);
  #endif
#else
  #define XformVector Xform
#endif

// Faster version of the kernel without offsets and strided access but with
// if-statement. Also assumes that 'n' is dividable by 'VW' and 'WPT'.
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void FasterKernel(const int n,
    const __global realV* restrict xgm, const __global realV* restrict ygm,
    __global realV* zgm)
{
  if (get_global_id(0) < n/VW) {
    #pragma unroll
    for (int _w = 0; _w < WPT; _w++) {
      const int id = _w*get_global_size(0) + get_global_id(0);
      realV xvec = xgm[id];
      realV yvec = ygm[id];
      realV zvec;
      XformVector(zvec, xvec, yvec);
      zgm[id] = zvec;
    }
  }
}

// Faster version of the kernel without offsets and strided accesses.
// Also assumes that 'n' is dividable by 'VW', 'WGS' and 'WPT'.
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void FastestKernel(const int n,
    const __global realV* restrict xgm, const __global realV* restrict ygm,
    __global realV* zgm)
{
  #pragma unroll
  for (int _w = 0; _w < WPT; _w++) {
    const int id = _w*get_global_size(0) + get_global_id(0);
      realV xvec = xgm[id];
      realV yvec = ygm[id];
      realV zvec;
      XformVector(zvec, xvec, yvec);
      zgm[id] = zvec;
  }
}

#endif

)" // End of the C++11 raw string literal
