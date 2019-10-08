// Enable loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
PROGRAM_STRING_DEBUG_INFO R"(

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xclip(const int n, const real_arg low_arg, const real_arg high_arg,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  const real low = GetRealArg(low_arg);
  const real high = GetRealArg(high_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id + x_offset];
    ygm[id + y_offset] = clamp(x, low, high);
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XclipStrided(
    const real_arg low_arg, const real_arg high_arg,
    const int n, const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  const real low = GetRealArg(low_arg);
  const real high = GetRealArg(high_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    int x_id = x_offset, y_id = y_offset;
    unravel2(id, &x_id, &y_id, rank, shape);
    real x = xgm[x_id];
    ygm[y_id] = clamp(x, low, high);
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xshrink(const int n, const real_arg lambd_arg, const real_arg bias_arg,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  const real lambd = GetRealArg(lambd_arg);
  const real bias = GetRealArg(bias_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id + x_offset];
    ygm[id + y_offset] = x < -lambd ? x + bias : x > lambd ? x - bias : ZERO;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XshrinkStrided(
    const real_arg lambd_arg, const real_arg bias_arg,
    const int n, const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  const real lambd = GetRealArg(lambd_arg);
  const real bias = GetRealArg(bias_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    int x_id = x_offset, y_id = y_offset;
    unravel2(id, &x_id, &y_id, rank, shape);
    real x = xgm[x_id];
    ygm[y_id] = x < -lambd ? x + bias : x > lambd ? x - bias : ZERO;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xrelu(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id + x_offset];
    ygm[id + y_offset] = maxval(ZERO, x);
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XreluStrided(
    const real_arg alpha_arg, const real_arg beta_arg,
    const int n, const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    int x_id = x_offset, y_id = y_offset;
    unravel2(id, &x_id, &y_id, rank, shape);
    real x = xgm[x_id];
    ygm[y_id] = maxval(ZERO, x);
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xleaky_relu(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  const real alpha = GetRealArg(alpha_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id + x_offset];
    if (x < ZERO)
      x *= alpha;
    ygm[id + y_offset] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xleaky_reluStrided(
    const real_arg alpha_arg, const real_arg beta_arg,
    const int n, const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  const real alpha = GetRealArg(alpha_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    int x_id = x_offset, y_id = y_offset;
    unravel2(id, &x_id, &y_id, rank, shape);
    real x = xgm[x_id];
    if (x < ZERO)
      x *= alpha;
    ygm[y_id] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xthresholded_relu(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  const real alpha = GetRealArg(alpha_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id + x_offset];
    if (x <= alpha)
      x = ZERO;
    ygm[id + y_offset] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xthresholded_reluStrided(
    const real_arg alpha_arg, const real_arg beta_arg,
    const int n, const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  const real alpha = GetRealArg(alpha_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    int x_id = x_offset, y_id = y_offset;
    unravel2(id, &x_id, &y_id, rank, shape);
    real x = xgm[x_id];
    if (x <= alpha)
      x = ZERO;
    ygm[x_id] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xselu(const int n, const real_arg alpha_arg, const real_arg gamma_arg,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  const real alpha = GetRealArg(alpha_arg);
  const real gamma = GetRealArg(gamma_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id + x_offset];
    if (x < ZERO)
      x = alpha * (exp(x) - ONE);
    x *= gamma;
    ygm[id + y_offset] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XseluStrided(
    const real_arg alpha_arg, const real_arg gamma_arg,
    const int n, const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  const real alpha = GetRealArg(alpha_arg);
  const real gamma = GetRealArg(gamma_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    int x_id = x_offset, y_id = y_offset;
    unravel2(id, &x_id, &y_id, rank, shape);
    real x = xgm[x_id];
    if (x < ZERO)
      x = alpha * (exp(x) - ONE);
    x *= gamma;
    ygm[y_id] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xelu(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  const real alpha = GetRealArg(alpha_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id + x_offset];
    if (x < ZERO)
      x = alpha * (exp(x) - ONE);
    ygm[id + y_offset] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XeluStrided(
    const real_arg alpha_arg, const real_arg beta_arg,
    const int n, const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  const real alpha = GetRealArg(alpha_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    int x_id = x_offset, y_id = y_offset;
    unravel2(id, &x_id, &y_id, rank, shape);
    real x = xgm[x_id];
    if (x < ZERO)
      x = alpha * (exp(x) - ONE);
    ygm[y_id] = x;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xhard_sigmoid(const int n, const real_arg alpha_arg, const real_arg beta_arg,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  const real alpha = GetRealArg(alpha_arg);
  const real beta = GetRealArg(beta_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x = xgm[id + x_offset];
    ygm[id + y_offset] = clamp(alpha * x + beta, ZERO, ONE);
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xhard_sigmoidStrided(
    const real_arg alpha_arg, const real_arg beta_arg,
    const int n, const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
  const real alpha = GetRealArg(alpha_arg);
  const real beta = GetRealArg(beta_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    int x_id = x_offset, y_id = y_offset;
    unravel2(id, &x_id, &y_id, rank, shape);
    real x = xgm[x_id];
    ygm[y_id] = clamp(alpha * x + beta, ZERO, ONE);
  }
}

)" // End of C++11 raw string literal
