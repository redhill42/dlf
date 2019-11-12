CL_PROGRAM R"(

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void DirectReverse(const int n, const int rank, __constant int* shape,
                   __global real* xgm, int x_offset, const int x_inc)
{
    const int lid = get_local_id(0);
    __local real lm[WGS];

    xgm += x_offset + unravel(get_group_id(0)*n, rank, shape);
    if (lid < n)
        lm[lid] = xgm[lid * x_inc];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < n)
        xgm[lid * x_inc] = lm[n - lid - 1];
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void BatchedReverse(const int n, const int block_size,
                    const int rank, __constant int* shape,
                    __global real* xgm, int x_offset, const int x_inc)
{
    const int batch = get_group_id(0) / block_size;
    const int block = get_group_id(0) % block_size;
    const int gid   = block * WGS + get_local_id(0);

    xgm += x_offset + unravel(batch*n, rank, shape);
    if (gid < n/2) {
        __global real*  left = xgm + gid*x_inc;
        __global real* right = xgm + (n - gid - 1)*x_inc;

        real t = *left;
        *left  = *right;
        *right = t;
    }
}

)"
