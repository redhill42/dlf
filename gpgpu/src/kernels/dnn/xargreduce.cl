// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#define ARG_REDUCE(name, op)                                            \
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))               \
void X##name(                                                           \
    const int n, const int k,                                           \
    const int rank, __constant int* shape,                              \
    const __global real* restrict xgm, const int x_offset,              \
    __global int* ygm)                                                  \
{                                                                       \
  const int id = get_global_id(0);                                      \
  if (id < n) {                                                         \
    const int x_off = unravel(id*k, rank, &shape[rank], shape);         \
    const int x_inc = shape[2*rank - 1];                                  \
    real acc = xgm[x_off];                                              \
    int idx = 0;                                                        \
    for (int ik = 1; ik < k; ++ik) {                                    \
      real x = xgm[x_off + ik*x_inc];                                   \
      if (x op acc) {                                                   \
        acc = x;                                                        \
        idx = ik;                                                       \
      }                                                                 \
    }                                                                   \
    ygm[id] = idx;                                                      \
  }                                                                     \
}

ARG_REDUCE(argmax, >)
ARG_REDUCE(argmin, <)

)" // End of the C++11 raw string literal
