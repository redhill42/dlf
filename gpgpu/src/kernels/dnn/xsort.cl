CL_PROGRAM R"(

#undef WPT
#define WPT 4
#define BLOCK_SIZE (WGS*WPT)

/*=========================================================================*/

INLINE_FUNC void comparator(LOCAL_PTR real* A, LOCAL_PTR real* B, int dir) {
    if ((*A < *B) == dir) {
        real t = *A;
        *A = *B;
        *B = t;
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

/*-------------------------------------------------------------------------*/

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void BlockSort(
    const int n, const int dir, const int rank, __constant int* shape,
    const __global real* restrict xgm, int x_offset, const int x_inc,
          __global real*          ygm, int y_offset, const int y_inc)
{
    const int batch = get_global_id(1);
    const int lid   = get_local_id(0);
    const int gid   = get_group_id(0)*WGS*2 + lid;

    unravel2(batch*n + gid, &x_offset, &y_offset, rank, shape);
    xgm += x_offset;
    ygm += y_offset;

    __local real lm[WGS*2];

    const real pad = dir ? SMALLEST : LARGEST;
    lm[lid +   0] = (gid +   0 < n) ? xgm[0          ] : pad;
    lm[lid + WGS] = (gid + WGS < n) ? xgm[WGS * x_inc] : pad;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 2; k <= WGS; k <<= 1) {
        int ddd = dir ^ ((lid & (k / 2)) != 0);
        for (int j = k >> 1; j > 0; j >>= 1) {
            int pos = 2*lid - (lid & (j - 1));
            comparator(&lm[pos], &lm[pos + j], ddd);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for (int j = WGS; j > 0; j >>= 1) {
        int pos = 2*lid - (lid & (j - 1));
        comparator(&lm[pos], &lm[pos + j], dir);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gid < n)
        ygm[0] = lm[lid];
    if (gid + WGS < n)
        ygm[WGS * y_inc] = lm[lid + WGS];
}

__kernel void DirectMerge(
    const int n, const int dir, const int rank, __constant int* shape,
    const __global real* restrict xgm, int x_offset, const int x_inc,
          __global real*          zgm, int z_offset, const int z_inc
#ifndef CUDA
    , __local real* lm) {
#else
    ) { extern __shared__ real lm[];
#endif

    const int lsz    = get_local_size(0) * WPT;
    const int wgid   = get_group_id(0);
    const int blocks = (n - 1) / (lsz*2) + 1;
    const int batch  = wgid / blocks;
    const int gid    = (wgid % blocks) * lsz*2;

    const int z_len  = min(n - gid, lsz*2);
    const int x_len  = min(z_len, lsz);
    const int y_len  = z_len - x_len;

    unravel2(batch*n + gid, &x_offset, &z_offset, rank, shape);
    xgm += x_offset;
    zgm += z_offset;

    LocalMerge(dir,
               xgm, x_len, x_inc,
               xgm + x_len*x_inc, y_len, x_inc,
               zgm, z_len, z_inc,
               lm, lm + lsz + 1);
}

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void MergePath(
    const int n, const int k, const int dir, const int rank, __constant int* shape,
    const __global real* restrict xgm, int x_offset, const int x_inc,
          __global int* diag, const int diag_offset)
{
    const int batch = get_global_id(1);
    const int wgid  = get_group_id(0);

    const int num_blocks = k*2 / BLOCK_SIZE;
    const int num_splits = get_num_groups(0) / num_blocks;
    const int cur_block  = wgid % num_blocks;
    const int cur_split  = wgid / num_blocks;

    const int gid = cur_split * k * 2;

    const int z_len = min(n - gid, k*2);
    const int x_len = min(z_len, k);
    const int y_len = z_len - x_len;

    xgm  += x_offset + unravel(batch*n + gid, rank, shape);
    const __global real* ygm = xgm + x_len*x_inc;
    diag += diag_offset + (batch*num_splits + cur_split) * (num_blocks+1)*2;

    __local int x_start, y_start, x_end, y_end, found;
    __local int lm[32];

    // Figure out the coordinates of our diagonal
    const int lid = get_local_id(0);
    const int local_offset = lid - 16;
    if (lid == 0) {
        int d = cur_block * BLOCK_SIZE;
        x_start = min(d, x_len);
        x_end   = max(0, d - y_len);
        y_start = max(0, d - x_len);
        y_end   = min(d, y_len);
        found   = 0;

        if (z_len <= k || cur_block*BLOCK_SIZE >= z_len) {
            diag[cur_block*2 + 0] = x_start;
            diag[cur_block*2 + 1] = y_len;
            found = 1;
        }
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
            lm[lid] = comp(xgm[(current_x-1)*x_inc], ygm[current_y*x_inc], dir);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // If we find the meeting of the '1's and '0's, we found the
        // intersection of the path and diagonal
        if (lid > 0 && lm[lid] != lm[lid-1]) {
            found = 1;
            diag[cur_block*2 + 0] = current_x;
            diag[cur_block*2 + 1] = current_y;
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
    if (lid == 0 && cur_block == 0) {
        diag[0] = 0;
        diag[1] = 0;
        diag[num_blocks*2 + 0] = x_len;
        diag[num_blocks*2 + 1] = y_len;
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void IndirectMerge(
    const int n, const int k, const int dir, const int rank, __constant int* shape,
    const __global real* restrict xgm, int x_offset, const int x_inc,
          __global real*          zgm, int z_offset, const int z_inc,
    const __global int*  restrict diag, const int diag_offset)
{
    const int batch = get_global_id(1);
    const int wgid  = get_group_id(0);

    const int num_blocks = k*2 / BLOCK_SIZE;
    const int num_splits = get_num_groups(0) / num_blocks;
    const int cur_block  = wgid % num_blocks;
    const int cur_split  = wgid / num_blocks;

    const int gid = cur_split * k * 2;

    const int z_len = min(n - gid, k*2);
    const int x_len = min(z_len, k);
    const int y_len = z_len - x_len;

    unravel2(batch*n + gid, &x_offset, &z_offset, rank, shape);
    xgm += x_offset;
    zgm += z_offset;

    // Storage space for local merge window
    __local int x_block_start, y_block_start, x_block_end, y_block_end;
    __local real xlm[BLOCK_SIZE + 1];
    __local real ylm[BLOCK_SIZE + 1];

    // Define global window and create sentinels
    if (get_local_id(0) == 0) {
        diag += diag_offset + (batch*num_splits + cur_split) * (num_blocks+1)*2 + cur_block*2;
        x_block_start = diag[0];
        y_block_start = diag[1];
        x_block_end   = diag[2];
        y_block_end   = diag[3];
        xlm[BLOCK_SIZE] = ylm[BLOCK_SIZE] = dir ? SMALLEST : LARGEST;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    while (x_block_start < x_block_end || y_block_start < y_block_end) {
        LocalMultiMerge(dir,
                        xgm, x_len, x_inc,
                        xgm + x_len*x_inc, y_len, x_inc,
                        zgm, z_len, z_inc,
                        &x_block_start, xlm,
                        &y_block_start, ylm);
    }
}

/*=========================================================================*/

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

/*=========================================================================*/

)"
