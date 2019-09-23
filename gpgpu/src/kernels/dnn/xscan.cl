// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#define WGS WGS1

#if defined(ROUTINE_cumsum)
#  define IDENTITY SetToZero
#  define OP Add
#elif defined(ROUTINE_cumprod)
#  define IDENTITY SetToOne
#  define OP Multiply
#elif defined(ROUTINE_cummax)
#  define IDENTITY(x) x = SMALLEST
#  define OP Max
#elif defined(ROUTINE_cummin)
#  define IDENTITY(x) x = LARGEST
#  define OP Min
#else
#  error "Unsupported scan operation"
#endif

#if defined(CUDA)
__device__ void LocalScan(const int lid, real* lm, real* sums)
#else
INLINE_FUNC void LocalScan(const int lid, __local real* lm, __global real* sums)
#endif
{
    int offset = 1;

    // Build sum in place up the tree
    for (int d = WGS; d > 0; d >>= 1) {
        if (lid < d*2) {
            int ai = (lid+1)*offset-1;
            int bi = (lid+2)*offset-1;
            OP(lm[bi], lm[bi], lm[ai]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        offset <<= 1;
    }

    // Clear the last element and store total sum
    if (lid == 0) {
        sums[get_group_id(0)] = lm[WGS*2] = lm[WGS*2-1];
        IDENTITY(lm[WGS*2-1]);
    }

    // Traverse down tree and build scan
    for (int d = 1; d < WGS*2; d <<= 1) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < d*2) {
            int ai = (lid+1)*offset-1;
            int bi = (lid+2)*offset-1;
            real t = lm[ai];
            lm[ai] = lm[bi];
            OP(lm[bi], lm[bi], t);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void PreScan(const int n, const int inclusive, const int rank, __constant int* shape,
             const __global real* restrict xgm, int x_offset, const int x_inc,
             __global real* ygm, int y_offset, const int y_inc,
             __global real* sums, const int sums_offset)
{
    const int batch = get_global_id(1);
    unravel2(batch*n, &x_offset, &y_offset, rank, shape);
    xgm  += x_offset;
    ygm  += y_offset;
    sums += sums_offset + batch * get_num_groups(0);

    const int lid = get_local_id(0) * 2;
    const int gid = get_group_id(0) * WGS * 2 + lid;

    // Load input data into shared memory
    __local real lm[WGS*2 + 1];
    IDENTITY(lm[lid]);
    if (gid < n)
        lm[lid] = xgm[gid*x_inc];
    IDENTITY(lm[lid+1]);
    if (gid+1 < n)
        lm[lid+1] = xgm[(gid+1)*x_inc];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform local scan on shared memory
    LocalScan(lid, lm, sums);

    // Write results to device memory
    if (gid < n)
        ygm[gid*y_inc] = lm[lid + inclusive];
    if (gid+1 < n)
        ygm[(gid+1)*y_inc] = lm[lid+1 + inclusive];
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void ScanPartialSums(const int n,
                     __global real* xgm, const int x_offset,
                     __global real* sums, const int sums_offset)
{
    const int batch = get_global_id(1);
    xgm  += x_offset + batch * n;
    sums += sums_offset + batch * get_num_groups(0);

    const int lid = get_local_id(0) * 2;
    const int gid = get_group_id(0) * WGS * 2 + lid;

    // Load input data into shared memory
    __local real lm[WGS*2 + 1];
    IDENTITY(lm[lid]);
    if (gid < n)
        lm[lid] = xgm[gid];
    IDENTITY(lm[lid+1]);
    if (gid+1 < n)
        lm[lid+1] = xgm[gid+1];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform local scan on shared memory
    LocalScan(lid, lm, sums);

    // Write results to device memory
    if (gid < n)
        xgm[gid] = lm[lid];
    if (gid+1 < n)
        xgm[gid+1] = lm[lid+1];
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void AddPartialSums(const int n,
                    __global real* xgm, const int x_offset,
                    __global real* sums, const int sums_offset)
{
    const int batch = get_global_id(1);
    const int wgid = get_group_id(0);
    const int gid = wgid * WGS * 2 + get_local_id(0) * 2;

    xgm  += x_offset + batch*n + gid;
    sums += sums_offset + batch*get_num_groups(0);

    if (wgid > 0) {
        real seed = sums[wgid];
        if (gid < n)
            OP(xgm[0], xgm[0], seed);
        if (gid+1 < n)
            OP(xgm[1], xgm[1], seed);
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void FinalScan(const int n, const int rank, __constant int* shape,
               __global real* ygm, int y_offset, const int y_inc,
               const __global real* sums, const int sums_offset)
{
    const int batch = get_global_id(1);
    const int wgid = get_group_id(0);
    const int gid = wgid * WGS * 2 + get_local_id(0) * 2;

    int x_offset = 0;
    unravel2(batch*n, &x_offset, &y_offset, rank, shape);
    ygm += y_offset + gid*y_inc;
    sums += sums_offset + batch*get_num_groups(0);

    if (wgid > 0) {
        real seed = sums[wgid];
        if (gid < n)
            OP(ygm[0], ygm[0], seed);
        if (gid+1 < n)
            OP(ygm[y_inc], ygm[y_inc], seed);
    }
}

)"
