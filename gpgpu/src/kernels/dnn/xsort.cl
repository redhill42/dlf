CL_PROGRAM R"(

INLINE_FUNC void comparator(LOCAL_PTR real* A, LOCAL_PTR real* B, int dir) {
    if ((*A < *B) == dir) {
        real t = *A;
        *A = *B;
        *B = t;
    }
}

INLINE_FUNC void comparator2(LOCAL_PTR real* A, LOCAL_PTR int* iA,
                             LOCAL_PTR real* B, LOCAL_PTR int* iB,
                             int dir)
{
    if ((*A < *B) == dir) {
        real t = *A;
        *A = *B;
        *B = t;

        int i = *iA;
        *iA = *iB;
        *iB = i;
    }
}

// Monolithic bitonic sort kernel for short arrays fitting into local memory
__kernel void DirectSort(
    const int n, const int dir, const int rank, __constant int* shape,
    const __global real* restrict xgm, int x_offset, const int x_inc,
          __global real*          ygm, int y_offset, const int y_inc
#ifndef CUDA
    , __local real* lm) {
#else
    ) { extern __shared__ real lm[];
#endif

    const int i = get_local_id(0);
    const int L = get_local_size(0);

    unravel2(get_group_id(0)*n + i, &x_offset, &y_offset, rank, shape);
    xgm += x_offset;
    ygm += y_offset;

    const real pad = dir ? SMALLEST : LARGEST;
    lm[i + 0] = (i + 0 < n) ? xgm[0        ] : pad;
    lm[i + L] = (i + L < n) ? xgm[L * x_inc] : pad;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 2; k <= L; k <<= 1) {
        int ddd = dir ^ ((i & (k / 2)) != 0);
        for (int j = k >> 1; j > 0; j >>= 1) {
            int pos = 2*i - (i & (j - 1));
            comparator(&lm[pos], &lm[pos + j], ddd);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for (int j = L; j > 0; j >>= 1) {
        int pos = 2*i - (i & (j - 1));
        comparator(&lm[pos], &lm[pos + j], dir);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < n)
        ygm[0] = lm[i];
    if (i + L < n)
        ygm[L * y_inc] = lm[i + L];
}

__kernel void DirectArgSort(
    const int n, const int dir, const int rank, __constant int* shape,
    const __global real* restrict xgm, int x_offset, const int x_inc,
          __global int*           ygm, int y_offset, const int y_inc
#ifndef CUDA
    , __local real* lm) {
#else
    ) { extern __shared__ real lm[];
#endif

    const int i = get_local_id(0);
    const int L = get_local_size(0);

    LOCAL_PTR real* s_val = lm;
    LOCAL_PTR int*  s_idx = (LOCAL_PTR int*)(lm + L*2);

    unravel2(get_group_id(0)*n + i, &x_offset, &y_offset, rank, shape);
    xgm += x_offset;
    ygm += y_offset;

    const real pad = dir ? SMALLEST : LARGEST;
    s_val[i + 0] = (i + 0 < n) ? xgm[0        ] : pad;
    s_val[i + L] = (i + L < n) ? xgm[L * x_inc] : pad;
    s_idx[i + 0] = i;
    s_idx[i + L] = i + L;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 2; k <= L; k <<= 1) {
        int ddd = dir ^ ((i & (k / 2)) != 0);
        for (int j = k >> 1; j > 0; j >>= 1) {
            int pos = 2*i - (i & (j - 1));
            comparator2(&s_val[pos], &s_idx[pos], &s_val[pos+j], &s_idx[pos+j], ddd);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for (int j = L; j > 0; j >>= 1) {
        int pos = 2*i - (i & (j - 1));
        comparator2(&s_val[pos], &s_idx[pos], &s_val[pos+j], &s_idx[pos+j], dir);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < n)
        ygm[0] = s_idx[i] < n ? s_idx[i] : s_idx[s_idx[i]];
    if (i + L < n)
        ygm[L * y_inc] = s_idx[i+L] < n ? s_idx[i+L] : s_idx[s_idx[i+L]];
}

)"
