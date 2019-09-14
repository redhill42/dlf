// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#define DEFINE_REDUCE_OP(name, identity, accum, accum2, post)           \
__kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))              \
void X##name(const int n,                                               \
    const __global real* restrict xgm, const int x_offset,              \
    __global real* ygm, const int y_offset)                             \
{                                                                       \
  const int batch = get_global_id(1);                                   \
  const int lid = get_local_id(0);                                      \
  const int wgid = get_group_id(0);                                     \
  const int num_groups = get_num_groups(0);                             \
  __local real lm[WGS1];                                                \
                                                                        \
  /* Perform loading and the first steps of the reduction */            \
  real acc; SetReal(acc, identity);                                     \
  int id = wgid*WGS1 + lid;                                             \
  while (id < n) {                                                      \
    real x = xgm[x_offset + batch*n + id];                              \
    accum(acc, acc, x);                                                 \
    id += WGS1*num_groups;                                              \
  }                                                                     \
  lm[lid] = acc;                                                        \
  barrier(CLK_LOCAL_MEM_FENCE);                                         \
                                                                        \
  /* Perform reduction in local memory */                               \
  for (int s = WGS1/2; s > 0; s >>= 1) {                                \
    if (lid < s)                                                        \
      accum2(lm[lid], lm[lid], lm[lid + s]);                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                       \
  }                                                                     \
                                                                        \
  /* Stores the per-workgroup result */                                 \
  if (lid == 0) {                                                       \
    ygm[y_offset + batch*WGS2*2 + wgid] = lm[0];                        \
  }                                                                     \
}                                                                       \
                                                                        \
__kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))              \
void X##name##Strided(                                                  \
    const int n, const int rank, __constant int* shape,                 \
    const __global real* restrict xgm, const int x_offset,              \
    __global real* ygm, const int y_offset)                             \
{                                                                       \
  const int batch = get_global_id(1);                                   \
  const int lid = get_local_id(0);                                      \
  const int wgid = get_group_id(0);                                     \
  const int num_groups = get_num_groups(0);                             \
  __local real lm[WGS1];                                                \
                                                                        \
  /* Perform loading and the first steps of the reduction */            \
  real acc; SetReal(acc, identity);                                     \
  int id = wgid*WGS1 + lid;                                             \
  while (id < n) {                                                      \
    int xid = unravel(batch*n + id, rank, shape);                       \
    real x = xgm[x_offset + xid];                                       \
    accum(acc, acc, x);                                                 \
    id += WGS1*num_groups;                                              \
  }                                                                     \
  lm[lid] = acc;                                                        \
  barrier(CLK_LOCAL_MEM_FENCE);                                         \
                                                                        \
  /* Perform reduction in local memory */                               \
  for (int s = WGS1/2; s > 0; s >>= 1) {                                \
    if (lid < s)                                                        \
      accum2(lm[lid], lm[lid], lm[lid + s]);                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                       \
  }                                                                     \
                                                                        \
  /* Stores the per-workgroup result */                                 \
  if (lid == 0) {                                                       \
    ygm[y_offset + batch*WGS2*2 + wgid] = lm[0];                        \
  }                                                                     \
}                                                                       \
                                                                        \
__kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))              \
void X##name##Epilogue(const int n,                                     \
    const __global real* restrict xgm, const int x_offset,              \
    __global real* ygm, const int y_offset)                             \
{                                                                       \
  const int batch = get_global_id(1);                                   \
  const int lid = get_local_id(0);                                      \
  __local real lm[WGS2];                                                \
                                                                        \
  /* Performs the first step of the reduction while loading the data */ \
  xgm = &xgm[x_offset + batch*WGS2*2];                                  \
  accum2(lm[lid], xgm[lid], xgm[lid + WGS2]);                           \
  barrier(CLK_LOCAL_MEM_FENCE);                                         \
                                                                        \
  /* Perform reduction in local memory */                               \
  for (int s = WGS2/2; s > 0; s >>= 1) {                                \
    if (lid < s)                                                        \
      accum2(lm[lid], lm[lid], lm[lid + s]);                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                       \
  }                                                                     \
                                                                        \
  /* Computes the final result */                                       \
  if (lid == 0) {                                                       \
    post(ygm[y_offset + batch], lm[0], n);                              \
  }                                                                     \
}                                                                       \
                                                                        \
__kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))              \
void X##name##StridedEpilogue(                                          \
    const int n, const int rank, __constant int* shape,                 \
    const __global real* restrict xgm, const int x_offset,              \
    __global real* ygm, const int y_offset)                             \
{                                                                       \
  const int batch = get_global_id(1);                                   \
  const int lid = get_local_id(0);                                      \
  __local real lm[WGS2];                                                \
                                                                        \
  /* Performs the first step of the reduction while loading the data */ \
  xgm = &xgm[x_offset + batch*WGS2*2];                                  \
  accum2(lm[lid], xgm[lid], xgm[lid + WGS2]);                           \
  barrier(CLK_LOCAL_MEM_FENCE);                                         \
                                                                        \
  /* Perform reduction in local memory */                               \
  for (int s = WGS2/2; s > 0; s >>= 1) {                                \
    if (lid < s)                                                        \
      accum2(lm[lid], lm[lid], lm[lid + s]);                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                       \
  }                                                                     \
                                                                        \
  /* Computes the final result */                                       \
  if (lid == 0) {                                                       \
    int yid = unravel(batch, rank, shape);                              \
    post(ygm[y_offset + yid], lm[0], n);                                \
  }                                                                     \
}                                                                       \

//---------------------------------------------------------------------------

#if defined(CUDA)
#if PRECISION == 16
  #define log  hlog
  #define exp  hexp
  #define sqrt hsqrt
#elif PRECISION == 32 || PRECISION == 3232
  #define log  logf
  #define exp  expf
  #define sqrt sqrtf
#endif
#endif

#define Ident(c,a,n)    c = a
#define Log(c,a,n)      SetReal(c, log(GetReal(a)))  /* FIXME */
#define Sqrt(c,a,n)     SetReal(c, sqrt(GetReal(a))) /* FIXME */

#if PRECISION == 3232 || PRECISION == 6464
  #define Mean(c,a,n) c.x = a.x / n; c.y = a.y / n
#else
  #define Mean(c,a,n) c = a / n
#endif

#define AbsoluteMax(c,a,x) SetToAbsoluteValue(x); Max(c,a,x)
#define AbsoluteMin(c,a,x) SetToAbsoluteValue(x); Min(c,a,x)
#define AbsoluteAdd(c,a,x) SetToAbsoluteValue(x); Add(c,a,x)

/* Note: a is unused */
#define AddSquare(c,a,x) MultiplyAdd(c,x,x)

#define AddNrm2(c,a,x1)     \
  do {                      \
    real x2 = x1;           \
    COMPLEX_CONJUGATE(x2);  \
    MultiplyAdd(c,x1,x2);   \
  } while (0)

#define AddExp(c,a,b) c = a + exp(b)

DEFINE_REDUCE_OP(reduce_max, SMALLEST, Max, Max, Ident)
DEFINE_REDUCE_OP(reduce_amax, ZERO, AbsoluteMax, Max, Ident)
DEFINE_REDUCE_OP(reduce_min, -SMALLEST, Min, Min, Ident)
DEFINE_REDUCE_OP(reduce_amin, -SMALLEST, AbsoluteMin, Min, Ident)
DEFINE_REDUCE_OP(reduce_sum, ZERO, Add, Add, Ident)
DEFINE_REDUCE_OP(reduce_asum, ZERO, AbsoluteAdd, Add, Ident)
DEFINE_REDUCE_OP(reduce_mean, ZERO, Add, Add, Mean)
DEFINE_REDUCE_OP(reduce_sum_square, ZERO, AddSquare, Add, Ident)
DEFINE_REDUCE_OP(reduce_prod, ONE, Multiply, Multiply, Ident)

#if PRECISION < 10000
DEFINE_REDUCE_OP(reduce_log_sum, ZERO, Add, Add, Log)
DEFINE_REDUCE_OP(reduce_log_sum_exp, ZERO, AddExp, Add, Log)
DEFINE_REDUCE_OP(reduce_nrm2, ZERO, AddNrm2, Add, Sqrt)
#endif

)"
