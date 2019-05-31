R"(

// Parameters set by the tuner or by database. Here they are given a basic
// value in case this kernel file is used outside of the GPGPU library.

#ifndef WGS1
  #define WGS1 64       // The local work-group size of the main kernel
#endif
#ifndef WGS2
  #define WGS2 64       // The local work-group size of the reduce kernel
#endif

static INLINE_FUNC long psi(long n) {
  return ((n & 3) == 1 || (n & 3) == 2) ? 1 : 0;
}

__kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))
void Euler210(const long n, const long u, __global long* output) {
  __local long lm[WGS1];
  const int lid = get_local_id(0);
  const int wgid = get_group_id(0);
  const int num_groups = get_num_groups(0);

  // Performs the first steps of the reduction
  long acc = 0;
  int a = wgid*WGS1 + lid + 1;
  while (a <= u) {
    long b = n / a;
    acc += psi(b);
    if ((a & 3) == 1)
        acc += b;
    if ((a & 3) == 3)
        acc -= b;
    a += WGS1*num_groups;
  }
  lm[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Perform reduction in local memory
  for (int s=WGS1/2; s>0; s>>=1) {
    if (lid < s)
      lm[lid] += lm[lid+s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Stores the per-workgroup result
  if (lid == 0) {
    output[wgid] = lm[0];
  }
}

// The epilogue reduction kernel, performing the final bit of the operation.
// This kernel has to be launched with a single workgoup only.
__kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
void Epilogue(const __global long* restrict input, __global long* ans) {
  __local long lm[WGS2];
  const int lid = get_local_id(0);

  // Performs the first step of the reduction while loading the data
  lm[lid] = input[lid] + input[lid + WGS2];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Performs reduction in local memory
  for (int s=WGS2/2; s>0; s>>=1) {
    if (lid < s)
      lm[lid] += lm[lid+s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Computes the final result
  if (lid == 0) {
    ans[0] = lm[0];
  }
}

)"