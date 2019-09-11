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
  real acc = identity;                                                  \
  int id = wgid*WGS1 + lid;                                             \
  while (id < n) {                                                      \
    real x = xgm[x_offset + batch*n + id];                              \
    acc = accum(acc, x);                                                \
    id += WGS1*num_groups;                                              \
  }                                                                     \
  lm[lid] = acc;                                                        \
  barrier(CLK_LOCAL_MEM_FENCE);                                         \
                                                                        \
  /* Perform reduction in local memory */                               \
  for (int s = WGS1/2; s > 0; s >>= 1) {                                \
    if (lid < s)                                                        \
      lm[lid] = accum2(lm[lid], lm[lid + s]);                           \
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
  real acc = identity;                                                  \
  int id = wgid*WGS1 + lid;                                             \
  while (id < n) {                                                      \
    int xid = unravel(batch*n + id, rank, &shape[rank], shape);         \
    real x = xgm[x_offset + xid];                                       \
    acc = accum(acc, x);                                                \
    id += WGS1*num_groups;                                              \
  }                                                                     \
  lm[lid] = acc;                                                        \
  barrier(CLK_LOCAL_MEM_FENCE);                                         \
                                                                        \
  /* Perform reduction in local memory */                               \
  for (int s = WGS1/2; s > 0; s >>= 1) {                                \
    if (lid < s)                                                        \
      lm[lid] = accum2(lm[lid], lm[lid + s]);                           \
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
  lm[lid] = accum2(xgm[lid], xgm[lid + WGS2]);                          \
  barrier(CLK_LOCAL_MEM_FENCE);                                         \
                                                                        \
  /* Perform reduction in local memory */                               \
  for (int s = WGS2/2; s > 0; s >>= 1) {                                \
    if (lid < s)                                                        \
      lm[lid] = accum2(lm[lid], lm[lid + s]);                           \
    barrier(CLK_LOCAL_MEM_FENCE);                                       \
  }                                                                     \
                                                                        \
  /* Computes the final result */                                       \
  if (lid == 0) {                                                       \
    ygm[y_offset + batch] = post(lm[0], n);                             \
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
  lm[lid] = accum2(xgm[lid], xgm[lid + WGS2]);                          \
  barrier(CLK_LOCAL_MEM_FENCE);                                         \
                                                                        \
  /* Perform reduction in local memory */                               \
  for (int s = WGS2/2; s > 0; s >>= 1) {                                \
    if (lid < s)                                                        \
      lm[lid] = accum2(lm[lid], lm[lid + s]);                           \
    barrier(CLK_LOCAL_MEM_FENCE);                                       \
  }                                                                     \
                                                                        \
  /* Computes the final result */                                       \
  if (lid == 0) {                                                       \
    int yid = unravel(batch, rank, &shape[rank], shape);                \
    ygm[y_offset + yid] = post(lm[0], n);                               \
  }                                                                     \
}                                                                       \

//---------------------------------------------------------------------------

#if defined(CUDA)
#if PRECISION == 16
  #define log  hlog
  #define exp  hexp
  #define sqrt hsqrt
#elif PRECISION == 32
  #define log  logf
  #define exp  expf
  #define sqrt sqrtf
#endif
#endif

#if PRECISION >= 10000
  #define xabs abs
#else
  #define xabs fabs
#endif

#define add(x,y)            ((x)+(y))
#define add_abs(x,y)        ((x)+xabs(y))
#define add_square(x,y)     ((x)+(y)*(y))
#define add_exp(x,y)        ((x)+exp(y))
#define mul(x,y)            ((x)*(y))

#define ident(x,n)          (x)
#define mean(x,n)           ((x)/(n))
#define log_r(x,n)          log(x)
#define sqrt_r(x,n)         sqrt(x)

DEFINE_REDUCE_OP(reduce_max, SMALLEST, max, max, ident)
DEFINE_REDUCE_OP(reduce_min, -SMALLEST, min, min, ident)
DEFINE_REDUCE_OP(reduce_sum, ZERO, add, add, ident)
DEFINE_REDUCE_OP(reduce_mean, ZERO, add, add, mean)
DEFINE_REDUCE_OP(reduce_sum_square, ZERO, add_square, add, ident)
DEFINE_REDUCE_OP(reduce_prod, ONE, mul, mul, ident)
DEFINE_REDUCE_OP(reduce_l1, ZERO, add_abs, add, ident)

#if PRECISION < 10000
DEFINE_REDUCE_OP(reduce_log_sum, ZERO, add, add, log_r)
DEFINE_REDUCE_OP(reduce_log_sum_exp, ZERO, add_exp, add, log_r)
DEFINE_REDUCE_OP(reduce_l2, ZERO, add_square, add, sqrt_r)
#endif

)"
