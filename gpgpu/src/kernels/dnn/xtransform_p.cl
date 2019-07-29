// Enable loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(
#line 4

#ifdef CUDA
#define clamp(x,minval,maxval) (x<minval ? minval : x>maxval ? maxval : x)
#endif

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xclip(const int n, const real_arg low_arg, const real_arg high_arg,
    const __global real* restrict xgm, __global real* ygm)
{
  const real low = GetRealArg(low_arg);
  const real high = GetRealArg(high_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id];
    ygm[id] = clamp(x, low, high);
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xshrink(const int n, const real_arg lambd_arg, const real_arg bias_arg,
    const __global real* restrict xgm, __global real* ygm)
{
  const real lambd = GetRealArg(lambd_arg);
  const real bias = GetRealArg(bias_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id];
    ygm[id] = x < -lambd ? x + bias : x > lambd ? x - bias : 0;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xrelu(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm,__global real* ygm)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id];
    ygm[id] = max(ZERO, x);
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xleaky_relu(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, __global real* ygm)
{
  const real alpha = GetRealArg(alpha_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id];
    if (x < ZERO)
      x *= alpha;
    ygm[id] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xthresholded_relu(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, __global real* ygm)
{
  const real alpha = GetRealArg(alpha_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id];
    if (x <= alpha)
      x = ZERO;
    ygm[id] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xselu(const int n, const real_arg alpha_arg, const real_arg gamma_arg,
    const __global real* restrict xgm, __global real* ygm)
{
  const real alpha = GetRealArg(alpha_arg);
  const real gamma = GetRealArg(gamma_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id];
    if (x < ZERO)
      x = alpha * (exp(x) - 1);
    x *= gamma;
    ygm[id] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xelu(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, __global real* ygm)
{
  const real alpha = GetRealArg(alpha_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id];
    if (x < ZERO)
      x = alpha * (exp(x) - ONE);
    ygm[id] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xhard_sigmoid(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, __global real* ygm)
{
  const real alpha = GetRealArg(alpha_arg);
  const real beta = GetRealArg(beta_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id];
    ygm[id] = max(ZERO, min(ONE, alpha * x + beta));
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xsoftsign(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, __global real* ygm)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id];
    ygm[id] = x / (ONE + fabs(x));
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xsoftplus(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, __global real* ygm)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id];
    ygm[id] = log(exp(x) + ONE);
  }
}

)" // End of C++11 raw string literal
