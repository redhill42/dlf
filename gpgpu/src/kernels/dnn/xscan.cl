// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void Xcumsum(const int n, const int exclusive, const int rank, __constant int* shape,
             const __global real* restrict xgm, const int x_offset, const int x_inc,
             __global real* ygm, const int y_offset, const int y_inc)
{
  const int i = get_global_id(0);
  int x_id = x_offset, y_id = y_offset;
  unravel2(i*n, &x_id, &y_id, rank, shape);

  real acc;
  SetToZero(acc);
  if (exclusive) {
    for (int j = 0; j < n; j++, x_id += x_inc, y_id += y_inc) {
      real x = xgm[x_id];
      ygm[y_id] = acc;
      Add(acc, acc, x);
    }
  } else {
    for (int j = 0; j < n; j++, x_id += x_inc, y_id += y_inc) {
      real x = xgm[x_id];
      Add(acc, acc, x);
      ygm[y_id] = acc;
    }
  }
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void Xcumprod(const int n, const int exclusive, const int rank, __constant int* shape,
              const __global real* restrict xgm, const int x_offset, const int x_inc,
              __global real* ygm, const int y_offset, const int y_inc)
{
  const int i = get_global_id(0);
  int x_id = x_offset, y_id = y_offset;
  unravel2(i*n, &x_id, &y_id, rank, shape);

  real acc;
  SetToOne(acc);
  if (exclusive) {
    for (int j = 0; j < n; j++, x_id += x_inc, y_id += y_inc) {
      real x = xgm[x_id];
      ygm[y_id] = acc;
      Multiply(acc, acc, x);
    }
  } else {
    for (int j = 0; j < n; j++, x_id += x_inc, y_id += y_inc) {
      real x = xgm[x_id];
      Multiply(acc, acc, x);
      ygm[y_id] = acc;
    }
  }
}

)"
