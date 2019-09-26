// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

__kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))
void Xsoftmax(const int n,
              __global real* restrict xgm, const int x_offset,
              const __global real* restrict bias, const int bias_offset,
              __global real* ygm, const int y_offset)
{
    const int batch = get_global_id(1);
    const int lid = get_local_id(0);
    const int wgid = get_group_id(0);
    const int wgs = get_local_size(0);
    const int num_groups = get_num_groups(0);

    __local real lm[WGS1];

    xgm  += x_offset + batch * n;
    bias += bias_offset;
    ygm  += y_offset + batch * num_groups;

    real acc = ZERO;
    int id = wgid * wgs + lid;
    while (id < n) {
        #if defined(ROUTINE_softmax)
          real x = exp(xgm[id] - bias[batch]);
          xgm[id] = x;
          acc += x;
        #else
          real x = xgm[id] - bias[batch];
          xgm[id] = x;
          acc += exp(x);
        #endif
        id += wgs * num_groups;
    }
    lm[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = wgs/2; s > 0; s >>= 1) {
        if (lid < s)
            lm[lid] += lm[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        ygm[wgid] = lm[0];
    }
}

__kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
void XsoftmaxEpilogue(const int n,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
    const int batch = get_global_id(1);
    const int lid = get_local_id(0);
    __local real lm[WGS2];

    xgm += x_offset + batch * WGS2*2;
    ygm += y_offset;

    lm[lid] = xgm[lid] + xgm[lid + WGS2];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = WGS2/2; s > 0; s >>= 1) {
        if (lid < s)
            lm[lid] += lm[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
      #if defined(ROUTINE_softmax)
        ygm[batch] = lm[0];
      #else
        ygm[batch] = log(lm[0]);
      #endif
    }
}

)" // End of the C++11 raw string literal
