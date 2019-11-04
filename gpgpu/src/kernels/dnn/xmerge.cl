CL_PROGRAM R"(

/*=========================================================================*/

__kernel void DirectMerge(
    const int dir,
    const int x_rank, __constant int* x_shape,
    const __global real* restrict xgm, const int x_len, const int x_offset, const int x_inc,
    const int y_rank, __constant int* y_shape,
    const __global real* restrict ygm, const int y_len, const int y_offset, const int y_inc,
    const int z_rank, __constant int* z_shape,
    __global real* zgm, const int z_offset, const int z_inc
#ifndef CUDA
    , __local real* lm) {
#else
    ) { extern __shared__ real lm[];
#endif

    const int batch = get_group_id(0);
    const int z_len = x_len + y_len;

    xgm += x_offset + unravel(batch * x_len, x_rank, x_shape);
    ygm += y_offset + unravel(batch * y_len, y_rank, y_shape);
    zgm += z_offset + unravel(batch * z_len, z_rank, z_shape);

    LocalMerge(dir,
               xgm, x_len, x_inc,
               ygm, y_len, y_inc,
               zgm, z_len, z_inc,
               lm, lm + get_local_size(0)*WPT + 1);
}

/*=========================================================================*/

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void MergePath(
    const int dir,
    const int x_rank, __constant int* x_shape,
    const __global real* restrict xgm, const int x_len, const int x_offset, const int x_inc,
    const int y_rank, __constant int* y_shape,
    const __global real* restrict ygm, const int y_len, const int y_offset, const int y_inc,
    __global int* diag, const int diag_offset)
{
    const int batch        = get_global_id(1);
    const int wgid         = get_group_id(0);
    const int blocks       = get_num_groups(0);
    const int lid          = get_local_id(0);
    const int local_offset = lid - 16;

    __local int x_start, y_start, x_end, y_end, found;
    __local int lm[32];

    xgm  += x_offset + unravel(batch * x_len, x_rank, x_shape);
    ygm  += y_offset + unravel(batch * y_len, y_rank, y_shape);
    diag += diag_offset + batch * 2*(blocks + 1);

    // Figure out the coordinates of our diagonal
    if (lid == 0) {
        int k   = wgid * WGS * WPT;
        x_start = min(k, x_len);
        x_end   = max(0, k - y_len);
        y_start = max(0, k - x_len);
        y_end   = min(k, y_len);
        found   = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Search the diagonal
    while (!found) {
        // Update coordinates within the 32-wide section of the diagonal
        int current_x = x_start - ((x_start - x_end) >> 1) - local_offset;
        int current_y = y_start + ((y_end - y_start) >> 1) + local_offset;

        // Are we a '1' or '0' with respect to A[x] <= B[x]
        if (current_x >= x_len || current_y < 0) {
            lm[lid] = 0;
        } else if (current_y >= y_len || current_x < 1) {
            lm[lid] = 1;
        } else {
            lm[lid] = comp(xgm[(current_x-1)*x_inc], ygm[current_y*y_inc], dir);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // If we find the meeting of the '1's and '0's, we found the
        // intersection of the path and diagonal
        if (lid > 0 && lm[lid] != lm[lid-1]) {
            found = 1;
            diag[wgid*2 + 0] = current_x;
            diag[wgid*2 + 1] = current_y;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Adjust the search window on the diagonal
        if (lid == 16) {
            if (lm[31] != 0) {
                x_end = current_x;
                y_end = current_y;
            } else {
                x_start = current_x;
                y_start = current_y;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Set the boundary diagonal (through 0,0 and x_len,y_len)
    if (lid == 0 && wgid == 0) {
        diag[0] = 0;
        diag[1] = 0;
        diag[blocks*2 + 0] = x_len;
        diag[blocks*2 + 1] = y_len;
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Merge(const int dir,
           const int x_rank, __constant int* x_shape,
           const __global real* restrict xgm, const int x_len, const int x_offset, const int x_inc,
           const int y_rank, __constant int* y_shape,
           const __global real* restrict ygm, const int y_len, const int y_offset, const int y_inc,
           const int z_rank, __constant int* z_shape,
           __global real* zgm, const int z_offset, const int z_inc,
           const __global int* diag, const int diag_offset)
{
    const int batch  = get_global_id(1);
    const int wgid   = get_group_id(0);
    const int blocks = get_num_groups(0);
    const int z_len  = x_len + y_len;

    // Storage space for local merge window
    __local int x_block_start, y_block_start, x_block_end, y_block_end;
    __local real xlm[WGS*WPT + 1];
    __local real ylm[WGS*WPT + 1];

    xgm += x_offset + unravel(batch * x_len, x_rank, x_shape);
    ygm += y_offset + unravel(batch * y_len, y_rank, y_shape);
    zgm += z_offset + unravel(batch * z_len, z_rank, z_shape);

    // Define global window and create sentinels
    if (get_local_id(0) == 0) {
        diag += diag_offset + batch * 2*(blocks + 1) + wgid*2;
        x_block_start = *diag++;
        y_block_start = *diag++;
        x_block_end   = *diag++;
        y_block_end   = *diag;

        xlm[WGS * WPT] = ylm[WGS * WPT] = dir ? SMALLEST : LARGEST;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    while (x_block_start < x_block_end || y_block_start < y_block_end) {
        LocalMultiMerge(dir,
                        xgm, x_len, x_inc,
                        ygm, y_len, y_inc,
                        zgm, z_len, z_inc,
                        &x_block_start, xlm,
                        &y_block_start, ylm);
    }
}

)"
