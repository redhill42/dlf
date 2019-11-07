CL_PROGRAM R"(

INLINE_FUNC void comparator2(
    LOCAL_PTR real* A, LOCAL_PTR int* iA,
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

INLINE_FUNC int get_index(LOCAL_PTR int* indices, const int i) {
    int res = indices[i];
    if (res < 0)
        res = indices[-res];
    return res;
}

INLINE_FUNC void LocalSort(
    const int i, const int L, const int dir,
    LOCAL_PTR real* s_val, LOCAL_PTR int* s_idx)
{
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
}

// Monolithic bitonic sort kernel for short arrays fitting into local memory
__kernel void DirectTopK(
    const int n, const int limit, const int dir,
    const int rank, __constant int* x_shape, __constant int* y_shape,
    const __global real* restrict xgm, int x_offset, const int x_inc,
          __global real* restrict ygm, int y_offset, const int y_inc,
          __global int*           igm, int i_offset, const int i_inc
#ifndef CUDA
    , __local real* lm) {
#else
    ) { extern __shared__ real lm[];
#endif

    const int i = get_local_id(0);
    const int L = get_local_size(0);

    LOCAL_PTR real* s_val = lm;
    LOCAL_PTR int*  s_idx = (LOCAL_PTR int*)(lm + L*2);

    unravel2(get_group_id(0)*limit + i, &y_offset, &i_offset, rank, y_shape);
    xgm += x_offset + unravel(get_group_id(0)*n + i, rank, x_shape);
    ygm += y_offset;
    igm += i_offset;

    const real pad = dir ? SMALLEST : LARGEST;
    s_val[i + 0] = (i + 0 < n) ? xgm[0        ] : pad;
    s_val[i + L] = (i + L < n) ? xgm[L * x_inc] : pad;
    s_idx[i + 0] = (i + 0 < n) ? (i + 0) : -(i + 0);
    s_idx[i + L] = (i + L < n) ? (i + L) : -(i + L);
    barrier(CLK_LOCAL_MEM_FENCE);

    LocalSort(i, L, dir, s_val, s_idx);

    if (i < limit) {
        ygm[0] = s_val[i];
        igm[0] = get_index(s_idx, i);
    }
    if (i + L < limit) {
        ygm[L * y_inc] = s_val[i+L];
        igm[L * i_inc] = get_index(s_idx, i + L);
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void BlockTopK(
    const int n, const int limit, const int y_len,
    const int dir, const int rank, __constant int* shape,
    const __global real* xgm, int x_offset, const int x_inc,
          __global real* ygm, int y_offset,
          __global int*  igm, int i_offset)
{
    const int batch = get_global_id(1);
    const int lid   = get_local_id(0);
    const int wgid  = get_group_id(0);
    const int xgid  = wgid*WGS*2 + lid;
    const int ygid  = batch*y_len + wgid*limit + lid;

    xgm += x_offset + unravel(batch*n + xgid, rank, shape);
    ygm += y_offset + ygid;
    igm += i_offset + ygid;

    __local real s_val[WGS*2];
    __local int  s_idx[WGS*2];

    const real pad = dir ? SMALLEST : LARGEST;
    s_val[lid +   0] = (xgid +   0 < n) ? xgm[0          ] : pad;
    s_val[lid + WGS] = (xgid + WGS < n) ? xgm[WGS * x_inc] : pad;
    s_idx[lid +   0] = (xgid +   0 < n) ? xgid +   0 : -(lid +   0);
    s_idx[lid + WGS] = (xgid + WGS < n) ? xgid + WGS : -(lid + WGS);
    barrier(CLK_LOCAL_MEM_FENCE);

    LocalSort(lid, WGS, dir, s_val, s_idx);

    if (lid < limit && xgid < n) {
        ygm[0] = s_val[lid];
        igm[0] = get_index(s_idx, lid);
    }
    if (lid + WGS < limit && xgid + WGS < n) {
        ygm[WGS] = s_val[lid + WGS];
        igm[WGS] = get_index(s_idx, lid + WGS);
    }
}

__kernel void CompactTopK(
    const int n, const int limit, const int dir,
    const int rank, __constant int* shape,
    const __global real* restrict xgm, int x_offset,
    const __global int*  restrict igm, int i_offset,
          __global real*          ygm, int y_offset, const int y_inc,
          __global int*           jgm, int j_offset, const int j_inc
#ifndef CUDA
    , __local real* lm) {
#else
    ) { extern __shared__ real lm[];
#endif

    const int i = get_local_id(0);
    const int L = get_local_size(0);

    LOCAL_PTR real* s_val = lm;
    LOCAL_PTR int*  s_idx = (LOCAL_PTR int*)(lm + L*2);

    const int batch = get_group_id(0);
    xgm += x_offset + batch*n + i;
    igm += i_offset + batch*n + i;

    unravel2(batch*limit + i, &y_offset, &j_offset, rank, shape);
    ygm += y_offset;
    jgm += j_offset;

    const real pad = dir ? SMALLEST : LARGEST;
    s_val[i + 0] = (i + 0 < n) ? xgm[0] : pad;
    s_val[i + L] = (i + L < n) ? xgm[L] : pad;
    s_idx[i + 0] = (i + 0 < n) ? igm[0] : -(i + 0);
    s_idx[i + L] = (i + L < n) ? igm[L] : -(i + L);
    barrier(CLK_LOCAL_MEM_FENCE);

    LocalSort(i, L, dir, s_val, s_idx);

    if (i < limit) {
        ygm[0] = s_val[i];
        jgm[0] = get_index(s_idx, i);
    }
    if (i + L < limit) {
        ygm[L * y_inc] = s_val[i + L];
        jgm[L * j_inc] = get_index(s_idx, i + L);
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void BlockCompactTopK(
    const int n, const int limit, const int y_len, const int dir,
    const __global real* restrict xgm, int x_offset,
    const __global int*  restrict igm, int i_offset,
          __global real*          ygm, int y_offset,
          __global int*           jgm, int j_offset)
{
    const int batch = get_global_id(1);
    const int lid   = get_local_id(0);
    const int wgid  = get_group_id(0);
    const int xgid  = wgid*WGS*2 + lid;
    const int ygid  = batch*y_len + wgid*limit + lid;

    xgm += x_offset + batch*n + xgid;
    igm += i_offset + batch*n + xgid;
    ygm += y_offset + ygid;
    jgm += j_offset + ygid;

    __local real s_val[WGS*2];
    __local int  s_idx[WGS*2];

    const real pad = dir ? SMALLEST : LARGEST;
    s_val[lid +   0] = (xgid +   0 < n) ? xgm[  0] : pad;
    s_val[lid + WGS] = (xgid + WGS < n) ? xgm[WGS] : pad;
    s_idx[lid +   0] = (xgid +   0 < n) ? igm[  0] : -(lid +   0);
    s_idx[lid + WGS] = (xgid + WGS < n) ? igm[WGS] : -(lid + WGS);
    barrier(CLK_LOCAL_MEM_FENCE);

    LocalSort(lid, WGS, dir, s_val, s_idx);

    if (lid < limit && xgid < n) {
        ygm[0] = s_val[lid];
        jgm[0] = get_index(s_idx, lid);
    }
    if (lid + WGS < limit && xgid + WGS < n) {
        ygm[WGS] = s_val[lid + WGS];
        jgm[WGS] = get_index(s_idx, lid + WGS);
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void SelectTopK(
    const int n, const int limit, const int rank, __constant int* y_shape,
    const __global real* restrict xgm, const int x_offset,
    const __global int*  restrict igm, const int i_offset,
          __global real* restrict ygm, const int y_offset, const int y_inc,
          __global int*  restrict jgm, const int j_offset, const int j_inc)
{
    unravel2(get_group_id(0)*limit, &y_offset, &j_offset, rank, y_shape);
    xgm += x_offset + get_group_id(0)*n;
    igm += i_offset + get_group_id(0)*n;
    ygm += y_offset;
    jgm += j_offset;

    for (int lid = get_local_id(0); lid < limit; lid += get_local_size(0)) {
        ygm[lid * y_inc] = xgm[lid];
        jgm[lid * j_inc] = igm[lid];
    }
}

)"
