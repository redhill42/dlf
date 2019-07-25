// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
void Xargmax(const int m, const int k, const int n,
             const __global real* restrict xgm,
             __global int* ygm)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  if (i < m && j < n) {
    const int x_off = i * k * n + j;
    real max = xgm[x_off];
    int idx = 0;
    for (int ik = 1; ik < k; ik++) {
      if (xgm[x_off + ik*n] > max) {
        max = xgm[x_off + ik*n];
        idx = ik;
      }
    }
    ygm[i * n + j] = idx;
  }
}

__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
void Xargmin(const int m, const int k, const int n,
             const __global real* restrict xgm,
             __global int* ygm)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  if (i < m && j < n) {
    const int x_off = i * k * n + j;
    real min = xgm[x_off];
    int idx = 0;
    for (int ik = 1; ik < k; ik++) {
      if (xgm[x_off + ik*n] < min) {
        min = xgm[x_off + ik*n];
        idx = ik;
      }
    }
    ygm[i * n + j] = idx;
  }
}

)" // End of the C++11 raw string literal
