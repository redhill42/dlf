CL_PROGRAM R"(

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xbatch_norm(const int batches, const int channels, const int spatial,
                 const __global real* restrict xgm,
                       __global real*          ygm,
                 const __global real* restrict scale,
                 const __global real* restrict bias,
                 const __global real* restrict mean,
                 const __global real* restrict var,
                 const real_arg epsilon_arg)
{
    const real epsilon = GetRealArg(epsilon_arg);
    for (int b = 0; b < batches; b++) {
        for (int c = 0; c < channels; c++) {
            int offset = (b * channels + c) * spatial;
            for (int id = get_global_id(0); id < spatial; id += get_global_size(0)) {
                ygm[offset+id] = scale[c] * (xgm[offset+id] - mean[c]) / sqrt(var[c] + epsilon) + bias[c];
            }
        }
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xlrn(const int batches, const int channels, const int spatial,
          const __global real* restrict xgm, __global real* ygm,
          const int nsize, const real_arg alpha_arg, const real_arg beta_arg, const real_arg bias_arg)
{
    const real alpha = GetRealArg(alpha_arg);
    const real beta  = -GetRealArg(beta_arg);
    const real bias  = GetRealArg(bias_arg);

    for (int b = 0; b < batches; b++) {
        for (int c = 0; c < channels; c++) {
            int offset = (b * channels + c) * spatial;
            const int L = max(0, c - (nsize-1)/2);
            const int H = min(channels-1, c + nsize/2);
            for (int id = get_global_id(0); id < spatial; id += get_global_size(0)) {
                real val = ZERO;
                for (int j = L; j <= H; j++) {
                    real x = xgm[offset + id + (j-c)*spatial];
                    val += x * x;
                }
                ygm[offset+id] = xgm[offset+id] * pow(alpha*val/nsize + bias, beta);
            }
        }
  }
}

)"
