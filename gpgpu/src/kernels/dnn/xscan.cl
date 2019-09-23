// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#if defined(ROUTINE_cumsum)
#  define IDENTITY SetToZero
#  define OP Add
#elif defined(ROUTINE_cumprod)
#  define IDENTITY SetToOne
#  define OP Multiply
#elif defined(ROUTINE_cummax)
#  define IDENTITY(x) x = SMALLEST
#  define OP Max
#elif defined(ROUTINE_cummin)
#  define IDENTITY(x) x = LARGEST
#  define OP Min
#else
#  error "Unsupported scan operation"
#endif

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void Xscan(const int n, const int exclusive, const int rank, __constant int* shape,
           const __global real* restrict xgm, const int x_offset, const int x_inc,
           __global real* ygm, const int y_offset, const int y_inc)
{
  const int batch = get_global_id(0);
  int x_id = x_offset, y_id = y_offset;
  unravel2(batch*n, &x_id, &y_id, rank, shape);

  real acc;
  IDENTITY(acc);
  if (exclusive) {
    for (int i = 0; i < n; i++, x_id += x_inc, y_id += y_inc) {
      real x = xgm[x_id];
      ygm[y_id] = acc;
      OP(acc, acc, x);
    }
  } else {
    for (int i = 0; i < n; i++, x_id += x_inc, y_id += y_inc) {
      real x = xgm[x_id];
      OP(acc, acc, x);
      ygm[y_id] = acc;
    }
  }
}

)"
