// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#if defined(ROUTINE_argmax)
#  define op >
#elif defined(ROUTINE_argmin)
#  define op <
#endif

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xargreduce(
    const int n, const int k,
    const int x_rank, __constant int* x_shape,
    const __global real* restrict xgm, const int x_offset,
    const int y_rank, __constant int* y_shape,
    __global int* ygm, const int y_offset)
{
  const int id = get_global_id(0);
  if (id < n) {
    const int x_off = unravel(id*k, x_rank, x_shape) + x_offset;
    const int x_inc = x_shape[2*x_rank - 1];
    const int y_off = unravel(id, y_rank, y_shape) + y_offset;
    real acc = xgm[x_off];
    int idx = 0;
    for (int ik = 1; ik < k; ++ik) {
      real x = xgm[x_off + ik*x_inc];
      if (x op acc) {
        acc = x;
        idx = ik;
      }
    }
    ygm[y_off] = idx;
  }
}

)" // End of the C++11 raw string literal
