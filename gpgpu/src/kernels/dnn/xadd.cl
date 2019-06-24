// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// A vector-vector addition function.
INLINE_FUNC realV AddVectorVector(const realV aval, const realV bvec) {
  realV cvec;
  #if VW == 1
    Add(cvec, aval, bvec);
  #elif VW == 2
    Add(cvec.x, aval.x, bvec.x);
    Add(cvec.y, aval.y, bvec.y);
  #elif VW == 4
    Add(cvec.x, aval.x, bvec.x);
    Add(cvec.y, aval.y, bvec.y);
    Add(cvec.z, aval.z, bvec.z);
    Add(cvec.w, aval.w, bvec.w);
  #elif VW == 8
    Add(cvec.s0, aval.s0, bvec.s0);
    Add(cvec.s1, aval.s1, bvec.s1);
    Add(cvec.s2, aval.s2, bvec.s2);
    Add(cvec.s3, aval.s3, bvec.s3);
    Add(cvec.s4, aval.s4, bvec.s4);
    Add(cvec.s5, aval.s5, bvec.s5);
    Add(cvec.s6, aval.s6, bvec.s6);
    Add(cvec.s7, aval.s7, bvec.s7);
  #elif VW == 16
    Add(cvec.s0, aval.s0, bvec.s0);
    Add(cvec.s1, aval.s1, bvec.s1);
    Add(cvec.s2, aval.s2, bvec.s2);
    Add(cvec.s3, aval.s3, bvec.s3);
    Add(cvec.s4, aval.s4, bvec.s4);
    Add(cvec.s5, aval.s5, bvec.s5);
    Add(cvec.s6, aval.s6, bvec.s6);
    Add(cvec.s7, aval.s7, bvec.s7);
    Add(cvec.s8, aval.s8, bvec.s8);
    Add(cvec.s9, aval.s9, bvec.s9);
    Add(cvec.sA, aval.sA, bvec.sA);
    Add(cvec.sB, aval.sB, bvec.sB);
    Add(cvec.sC, aval.sC, bvec.sC);
    Add(cvec.sD, aval.sD, bvec.sD);
    Add(cvec.sE, aval.sE, bvec.sE);
    Add(cvec.sF, aval.sF, bvec.sF);
  #endif
  return cvec;
}

// Full version of the kernel with offsets and strided accesses
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xadd(const int n,
    const __global real* restrict xgm, const int x_offset, const int x_inc,
    const __global real* restrict ygm, const int y_offset, const int y_inc,
    __global real* zgm, const int z_offset, const int z_inc)
{
  // Loops over the work that needs to be done (allows for an arbitrary number of threads)
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x_value = xgm[id*x_inc + x_offset];
    real y_value = ygm[id*y_inc + y_offset];
    real z_value;
    Add(z_value, x_value, y_value);
    zgm[id*z_inc + z_offset] = z_value;
  }
}

// Faster version of the kernel without offsets and strided access but with
// if-statement. Also assumes that 'n' is dividable by 'VW' and 'WPT'.
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XaddFaster(const int n,
    const __global realV* restrict xgm,
    const __global realV* restrict ygm,
    __global realV* zgm)
{
  if (get_global_id(0) < n / (VW)) {
    #pragma unroll
    for (int _w = 0; _w < WPT; _w++) {
      const int id = _w*get_global_size(0) + get_global_id(0);
      zgm[id] = AddVectorVector(xgm[id], ygm[id]);
    }
  }
}

// Faster version of the kernel without offsets and strided accesses.
// Also assumes that 'n' is dividable by 'VW', 'WGS' and 'WPT'.
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XaddFastest(const int n,
    const __global realV* restrict xgm,
    const __global realV* restrict ygm,
    __global realV* zgm)
{
  #pragma unroll
  for (int _w = 0; _w < WPT; _w++) {
    const int id = _w*get_global_size(0) + get_global_id(0);
      zgm[id] = AddVectorVector(xgm[id], ygm[id]);
  }
}

)" // End of the C++11 raw string literal
