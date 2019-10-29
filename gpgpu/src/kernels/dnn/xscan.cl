CL_PROGRAM R"(

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
#elif defined(ROUTINE_nonzero)
#  define IDENTITY(x) x = 0
#  define OP(c,a,b)   c = a + b
#else
#  error "Unsupported scan operation"
#endif

#if defined(ROUTINE_nonzero)
  #define MAP(x) ((x) != 0)
  typedef int realR;
#else
  #define MAP(x) (x)
  typedef real realR;
#endif

//---------------------------------------------------------------------------

// Note: lid = get_local_id(0) * 2
STATIC void DirectLocalScan(const int lid, const int lsz, LOCAL_PTR realR* lm) {
    int offset = 1;

    // Build sum in place up the tree
    for (int d = lsz; d > 0; d >>= 1) {
        if (lid < d*2) {
            int ai = (lid + 1)*offset - 1;
            int bi = (lid + 2)*offset - 1;
            OP(lm[bi], lm[bi], lm[ai]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        offset <<= 1;
    }

    // Clear the last element
    if (lid == 0) {
        lm[lsz*2] = lm[lsz*2-1];
        IDENTITY(lm[lsz*2-1]);
    }

    // Traverse down tree and build scan
    for (int d = 1; d < lsz*2; d <<= 1) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < d*2) {
            int   ai = (lid + 1)*offset - 1;
            int   bi = (lid + 2)*offset - 1;
            realR t  = lm[ai];
            lm[ai]   = lm[bi];
            OP(lm[bi], lm[bi], t);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void DirectScan(
    const int n, const int inclusive, const int rank, __constant int* shape,
    const __global real* restrict xgm, int x_offset, const int x_inc,
    __global realR* ygm, int y_offset, const int y_inc
#ifndef CUDA
    , __local realR* lm) {
#else
    ) { extern __shared__ realR lm[];
#endif

    const int lid = get_local_id(0)*2;
    const int gid = get_group_id(0)*n + lid;

    unravel2(gid, &x_offset, &y_offset, rank, shape);
    xgm += x_offset;
    ygm += y_offset;

    // Load input data into shared memory
    IDENTITY(lm[lid]);
    if (lid < n)
        lm[lid] = MAP(xgm[0]);
    IDENTITY(lm[lid+1]);
    if (lid+1 < n)
        lm[lid+1] = MAP(xgm[x_inc]);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform local scan on shared memory
    DirectLocalScan(lid, get_local_size(0), lm);

    // Write result to device memory
    if (lid < n)
        ygm[0] = lm[lid + inclusive];
    if (lid+1 < n)
        ygm[y_inc] = lm[lid+1 + inclusive];
}

//---------------------------------------------------------------------------

// Note: lid = get_local_id(0) * 2
STATIC void LocalScan(const int lid, LOCAL_PTR realR* lm, __global realR* sums) {
    int offset = 1;

    // Build sum in place up the tree
    for (int d = WGS; d > 0; d >>= 1) {
        if (lid < d*2) {
            int ai = (lid + 1)*offset - 1;
            int bi = (lid + 2)*offset - 1;
            OP(lm[bi], lm[bi], lm[ai]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        offset <<= 1;
    }

    // Clear the last element and store total sum
    if (lid == 0) {
        *sums = lm[WGS*2] = lm[WGS*2-1];
        IDENTITY(lm[WGS*2-1]);
    }

    // Traverse down tree and build scan
    for (int d = 1; d < WGS*2; d <<= 1) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < d*2) {
            int   ai = (lid + 1)*offset - 1;
            int   bi = (lid + 2)*offset - 1;
            realR t  = lm[ai];
            lm[ai]   = lm[bi];
            OP(lm[bi], lm[bi], t);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void PreScan(const int n, const int inclusive, const int rank, __constant int* shape,
             const __global real* restrict xgm, int x_offset, const int x_inc,
             __global realR* ygm, int y_offset, const int y_inc,
             __global realR* sums, const int sums_offset)
{
    const int batch = get_global_id(1);
    const int gid = get_global_id(0) * 2;
    const int lid = get_local_id(0) * 2;

    unravel2(batch*n + gid, &x_offset, &y_offset, rank, shape);
    xgm  += x_offset;
    ygm  += y_offset;
    sums += sums_offset + batch*get_num_groups(0) + get_group_id(0);

    // Load input data into shared memory
    __local realR lm[WGS*2 + 1];
    IDENTITY(lm[lid]);
    if (gid < n)
        lm[lid] = MAP(xgm[0]);
    IDENTITY(lm[lid+1]);
    if (gid+1 < n)
        lm[lid+1] = MAP(xgm[x_inc]);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform local scan on shared memory
    LocalScan(lid, lm, sums);

    // Write results to device memory
    if (gid < n)
        ygm[0] = lm[lid + inclusive];
    if (gid+1 < n)
        ygm[y_inc] = lm[lid+1 + inclusive];
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void ScanPartialSums(const int n,
                     __global realR* xgm, const int x_offset,
                     __global realR* sums, const int sums_offset)
{
    const int batch = get_global_id(1);
    const int gid = get_global_id(0) * 2;
    const int lid = get_local_id(0) * 2;

    xgm  += x_offset + batch*n + gid;
    sums += sums_offset + batch*get_num_groups(0) + get_group_id(0);

    // Load input data into shared memory
    __local realR lm[WGS*2 + 1];
    IDENTITY(lm[lid]);
    if (gid < n)
        lm[lid] = xgm[0];
    IDENTITY(lm[lid+1]);
    if (gid+1 < n)
        lm[lid+1] = xgm[1];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform local scan on shared memory
    LocalScan(lid, lm, sums);

    // Write results to device memory
    if (gid < n)
        xgm[0] = lm[lid];
    if (gid+1 < n)
        xgm[1] = lm[lid+1];
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void AddPartialSums(const int n,
                    __global realR* xgm, const int x_offset,
                    const __global realR* sums, const int sums_offset)
{
    const int batch = get_global_id(1);
    const int gid = get_global_id(0) * 2;

    xgm  += x_offset + batch*n + gid;
    sums += sums_offset + batch*get_num_groups(0) + get_group_id(0);

    if (gid < n)
        OP(xgm[0], xgm[0], *sums);
    if (gid+1 < n)
        OP(xgm[1], xgm[1], *sums);
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void FinalScan(const int n, const int rank, __constant int* shape,
               __global realR* ygm, int y_offset, const int y_inc,
               const __global realR* sums, const int sums_offset)
{
    const int batch = get_global_id(1);
    const int gid = get_global_id(0) * 2;

    int x_offset = 0;
    unravel2(batch*n + gid, &x_offset, &y_offset, rank, shape);
    ygm  += y_offset;
    sums += sums_offset + batch*get_num_groups(0) + get_group_id(0);

    if (gid < n)
        OP(ygm[0], ygm[0], *sums);
    if (gid+1 < n)
        OP(ygm[y_inc], ygm[y_inc], *sums);
}

)"
