// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

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

)" // End of the C++11 raw string literal
