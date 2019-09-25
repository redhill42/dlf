// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void Xsoftmax(const int n, const __global real* restrict xgm, __global real* ygm) {
  const int batch = get_group_id(0);
  const size_t data_off = batch * n;

  real max = xgm[data_off];
  for (int i = 1; i < n; ++i) {
    real x = xgm[data_off + i];
    max = maxval(max, x);
  }

  real denom = ZERO;
  for (int i = 0; i < n; ++i) {
    denom += ygm[data_off + i] = exp(xgm[data_off + i] - max);
  }

  for (int i = 0; i < n; ++i) {
    ygm[data_off + i] /= denom;
  }
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void Xlogsoftmax(const int n, const __global real* restrict xgm, __global real* ygm) {
  const int batch = get_group_id(0);
  const size_t data_off = batch * n;

  real max = xgm[data_off];
  for (int i = 1; i < n; ++i) {
    real x = xgm[data_off + i];
    max = maxval(max, x);
  }

  real sum = ZERO;
  for (int i = 0; i < n; ++i)
    sum += exp(xgm[data_off + i] - max);
  sum = log(sum);

  for (int i = 0; i < n; ++i) {
    ygm[data_off + i] = xgm[data_off + i] - max - sum;
  }
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void Xhardmax(const int n, const __global real* restrict xgm, __global real* ygm) {
  const int batch = get_group_id(0);
  const size_t data_off = batch * n;

  real amax = xgm[data_off];
  int  imax = 0;
  for (int i = 0; i < n; ++i) {
    if (xgm[data_off + i] > amax) {
      amax = xgm[data_off + i];
      imax = i;
    }
    ygm[data_off + i] = ZERO;
  }
  ygm[data_off + imax] = ONE;
}

)" // End of the C++11 raw string literal
