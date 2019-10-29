CL_PROGRAM R"(

#define BATCH_WGS 32

INLINE_FUNC void ArgReduce(
    const int n, LOCAL_PTR real* xlm, LOCAL_PTR int* ilm,
    const __global real* restrict xgm,
    __global real* ygm, __global int* igm)
{
    const int lid = get_local_id(0);
    const int wgid = get_group_id(0);
    const int wgs = get_local_size(0);
    const int num_groups = get_num_groups(0);

    // Performs loading and the first steps of the reduction
    int id = wgid * wgs + lid;
    real max = SMALLEST;
    int imax = 0;
    while (id < n) {
        real x = xgm[id];
        #if defined(ROUTINE_argmin)
          x = -x;
        #endif
        if (x > max) {
            max = x;
            imax = id;
        }
        id += wgs * num_groups;
    }
    xlm[lid] = max;
    ilm[lid] = imax;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Performs reduction in local memory
    for (int s = wgs/2; s > 0; s >>= 1) {
        if (lid < s && xlm[lid + s] > xlm[lid]) {
            xlm[lid] = xlm[lid + s];
            ilm[lid] = ilm[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Stores the per-workgroup result
    if (lid == 0) {
        ygm[wgid] = xlm[0];
        igm[wgid] = ilm[0];
    }
}

__kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))
void Xargreduce(
    const int n,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset,
    __global int* igm, const int i_offset)
{
    __local real xlm[WGS1];
    __local int  ilm[WGS1];
    ArgReduce(n, xlm, ilm, xgm + x_offset, ygm + y_offset, igm + i_offset);
}

__kernel __attribute__((reqd_work_group_size(BATCH_WGS, 1, 1)))
void XargreduceBatched(
    const int n,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset,
    __global int* igm, const int i_offset)
{
    __local real xlm[BATCH_WGS];
    __local int  ilm[BATCH_WGS];

    const int batch = get_global_id(1);
    const int num_groups = get_num_groups(0);

    xgm += x_offset + batch*n;
    ygm += y_offset + batch * num_groups;
    igm += i_offset + batch * num_groups;
    ArgReduce(n, xlm, ilm, xgm, ygm, igm);
}

//---------------------------------------------------------------------------

INLINE_FUNC void StridedArgReduce(
    const int n, LOCAL_PTR real* xlm, LOCAL_PTR int* ilm,
    const int rank, __constant int* shape,
    const __global real* restrict xgm,
    __global real* ygm, __global int* igm)
{
    const int batch = get_global_id(1);
    const int lid = get_local_id(0);
    const int wgid = get_group_id(0);
    const int wgs = get_local_size(0);
    const int num_groups = get_num_groups(0);

    // Performs loading and the first steps of the reduction
    int id = wgid * wgs + lid;
    real max = SMALLEST;
    int imax = 0;
    while (id < n) {
        real x = xgm[unravel(batch*n + id, rank, shape)];
        #if defined(ROUTINE_argmin)
          x = -x;
        #endif
        if (x > max) {
            max = x;
            imax = id;
        }
        id += wgs * num_groups;
    }
    xlm[lid] = max;
    ilm[lid] = imax;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    for (int s = wgs/2; s > 0; s >>= 1) {
        if (lid < s && xlm[lid + s] > xlm[lid]) {
            xlm[lid] = xlm[lid + s];
            ilm[lid] = ilm[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Stores the per-workgroup result
    if (lid == 0) {
        ygm[wgid] = xlm[0];
        igm[wgid] = ilm[0];
    }
}

__kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))
void XargreduceStrided(
    const int n, const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset,
    __global int* igm, const int i_offset)
{
    __local real xlm[WGS1];
    __local int  ilm[WGS1];

    xgm += x_offset;
    ygm += y_offset;
    igm += i_offset;
    StridedArgReduce(n, xlm, ilm, rank, shape, xgm, ygm, igm);
}

__kernel __attribute__((reqd_work_group_size(BATCH_WGS, 1, 1)))
void XargreduceStridedBatched(
    const int n, const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset,
    __global int* igm, const int i_offset)
{
    __local real xlm[BATCH_WGS];
    __local int  ilm[BATCH_WGS];

    const int batch = get_global_id(1);
    const int num_groups = get_num_groups(0);

    xgm += x_offset;
    ygm += y_offset + batch * num_groups;
    igm += i_offset + batch * num_groups;

    StridedArgReduce(n, xlm, ilm, rank, shape, xgm, ygm, igm);
}

//---------------------------------------------------------------------------

__kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
void XargreduceEpilogue(
    const __global real* restrict xgm, const int x_offset,
    const __global int* restrict igm, const int i_offset,
    __global int* ygm, const int y_offset)
{
    const int batch = get_global_id(1);
    const int lid = get_local_id(0);
    __local real xlm[WGS2];
    __local int  ilm[WGS2];

    xgm += x_offset + batch * WGS2*2;
    igm += i_offset + batch * WGS2*2;
    ygm += y_offset;

    // Performs the first step of the reduction while loading the data
    if (xgm[lid + WGS2] > xgm[lid]) {
        xlm[lid] = xgm[lid + WGS2];
        ilm[lid] = igm[lid + WGS2];
    } else {
        xlm[lid] = xgm[lid];
        ilm[lid] = igm[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Performs reduction in local memory
    for (int s = WGS2/2; s > 0; s >>= 1) {
        if (lid < s && xlm[lid + s] > xlm[lid]) {
            xlm[lid] = xlm[lid + s];
            ilm[lid] = ilm[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Stores the final result
    if (lid == 0) {
        ygm[batch] = ilm[0];
    }
}

__kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
void XargreduceEpilogueStrided(
    const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    const __global int* restrict igm, const int i_offset,
    __global real* ygm, const int y_offset)
{
    const int batch = get_global_id(1);
    const int lid = get_local_id(0);
    __local real xlm[WGS2];
    __local int  ilm[WGS2];

    xgm += x_offset + batch * WGS2*2;
    igm += i_offset + batch * WGS2*2;
    ygm += y_offset;

    // Performs the first step of the reduction while loading the data
    if (xgm[lid + WGS2] > xgm[lid]) {
        xlm[lid] = xgm[lid + WGS2];
        ilm[lid] = igm[lid + WGS2];
    } else {
        xlm[lid] = xgm[lid];
        ilm[lid] = igm[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Performs reduction in local memory
    for (int s = WGS2/2; s > 0; s >>= 1) {
        if (lid < s && xlm[lid + s] > xlm[lid]) {
            xlm[lid] = xlm[lid + s];
            ilm[lid] = ilm[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // stores the final result
    if (lid == 0) {
        ygm[unravel(batch, rank, shape)] = ilm[0];
    }
}

)"
