// Enable loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#line 5
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1))) \
void Xrelu(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, const int x_offset, const int x_inc,
    __global real* ygm, const int y_offset, const int y_inc)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id*x_inc + x_offset];
    ygm[id*y_inc + y_offset] = max(ZERO, x);
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1))) \
void Xleaky_relu(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, const int x_offset, const int x_inc,
    __global real* ygm, const int y_offset, const int y_inc)
{
  const real alpha = GetRealArg(alpha_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id*x_inc + x_offset];
    if (x < ZERO)
      x *= alpha;
    ygm[id*y_inc + y_offset] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1))) \
void Xthresholded_relu(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, const int x_offset, const int x_inc,
    __global real* ygm, const int y_offset, const int y_inc)
{
  const real alpha = GetRealArg(alpha_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id*x_inc + x_offset];
    if (x <= alpha)
      x = ZERO;
    ygm[id*y_inc + y_offset] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1))) \
void Xselu(const int n, const real_arg alpha_arg, const real_arg gamma_arg,
    const __global real* restrict xgm, const int x_offset, const int x_inc,
    __global real* ygm, const int y_offset, const int y_inc)
{
  const real alpha = GetRealArg(alpha_arg);
  const real gamma = GetRealArg(gamma_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id*x_inc + x_offset];
    if (x < ZERO)
      x = alpha * (exp(x) - 1);
    x *= gamma;
    ygm[id*y_inc + y_offset] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1))) \
void Xelu(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, const int x_offset, const int x_inc,
    __global real* ygm, const int y_offset, const int y_inc)
{
  const real alpha = GetRealArg(alpha_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id*x_inc + x_offset];
    if (x < ZERO)
      x = alpha * (exp(x) - ONE);
    ygm[id*y_inc + y_offset] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1))) \
void Xhard_sigmoid(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, const int x_offset, const int x_inc,
    __global real* ygm, const int y_offset, const int y_inc)
{
  const real alpha = GetRealArg(alpha_arg);
  const real beta = GetRealArg(beta_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id*x_inc + x_offset];
    ygm[id*y_inc + y_offset] = max(ZERO, min(ONE, alpha * x + beta));
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1))) \
void Xsoftsign(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, const int x_offset, const int x_inc,
    __global real* ygm, const int y_offset, const int y_inc)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id*x_inc + x_offset];
    ygm[id*y_inc + y_offset] = x / (ONE + fabs(x));
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1))) \
void Xsoftplus(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, const int x_offset, const int x_inc,
    __global real* ygm, const int y_offset, const int y_inc)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id*x_inc + x_offset];
    ygm[id*y_inc + y_offset] = log(exp(x) + ONE);
  }
}

)" // End of C++11 raw string literal
