// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#define DEFINE_REDUCE_OP(name, identity, accum, post)                   \
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))                 \
void X##name(const int n,                                               \
    const __global real* restrict xgm, const int x_offset,              \
    __global real* ygm, const int y_offset)                             \
{                                                                       \
  const int id = get_global_id(0);                                      \
  real acc = identity;                                                  \
  for (int i = 0, xid = id*n+x_offset; i < n; ++i, ++xid) {             \
    real x = xgm[xid];                                                  \
    acc = accum;                                                        \
  }                                                                     \
  ygm[id + y_offset] = post;                                            \
}                                                                       \
                                                                        \
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))                 \
void X##name##Strided(                                                  \
    const int n, const int rank, __constant int* shape,                 \
    const __global real* restrict xgm, const int x_offset,              \
    __global real* ygm, const int y_offset)                             \
{                                                                       \
  const int id = get_global_id(0);                                      \
  real acc = identity;                                                  \
  for (int i = 0, xid = id*n+x_offset; i < n; ++i, ++xid) {             \
    real x = xgm[unravel(xid, rank, &shape[rank], shape) + x_offset];   \
    acc = accum;                                                        \
  }                                                                     \
  ygm[id + y_offset] = post;                                            \
}

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

DEFINE_REDUCE_OP(reduce_max, SMALLEST, max(acc,x), acc)
DEFINE_REDUCE_OP(reduce_min, -SMALLEST, min(acc,x), acc)
DEFINE_REDUCE_OP(reduce_sum, ZERO, acc+x, acc)
DEFINE_REDUCE_OP(reduce_mean, ZERO, acc+x, acc/n)
DEFINE_REDUCE_OP(reduce_sum_square, ZERO, acc+x*x, acc)
DEFINE_REDUCE_OP(reduce_log_sum, ZERO, acc+x, log(acc))
DEFINE_REDUCE_OP(reduce_log_sum_exp, ZERO, acc+exp(x), log(acc))
DEFINE_REDUCE_OP(reduce_prod, ONE, acc*x, acc)
DEFINE_REDUCE_OP(reduce_l1, ZERO, acc+fabs(x), acc)
DEFINE_REDUCE_OP(reduce_l2, ZERO, acc+x*x, sqrt(acc))

)" // End of the C++11 raw string literal
