CL_PROGRAM R"(

#undef WPT
#define WPT 4

// dir = 0: ascending, dir = 1: descending
#define comp(A, B, dir) (((A) <= (B)) != (dir))

#ifdef ROUTINE_ARGSORT
#define ARG(x) x
#else
#define ARG(x)
#endif

#ifndef ROUTINE_ARGSORT
INLINE_FUNC void LocalMerge(
    const int dir,
    const __global real* restrict xgm, const int x_len, const int x_inc,
    const __global real* restrict ygm, const int y_len, const int y_inc,
          __global real*          zgm, const int z_len, const int z_inc,
    LOCAL_PTR real* xlm, LOCAL_PTR real* ylm)
#else
INLINE_FUNC void LocalArgMerge(
    const int dir,
    const int x_len, const __global real* restrict xgm, const int  x_inc,
                     const __global int* restrict ixgm, const int ix_inc,
    const int y_len, const __global real* restrict ygm, const int  y_inc,
                     const __global int* restrict iygm, const int iy_inc,
    const int z_len,       __global real*          zgm, const int  z_inc,
                           __global int*          izgm, const int iz_inc,
    LOCAL_PTR real* xlm, LOCAL_PTR real* ylm)
#endif
{
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    // Load current local window
    const real pad = dir ? SMALLEST : LARGEST;
    #pragma unroll
    for (int _w = 0, i = lid; _w < WPT; _w++, i += lsz) {
        xlm[i] = (i < x_len) ? xgm[i * x_inc] : pad;
        ylm[i] = (i < y_len) ? ygm[i * y_inc] : pad;
    }
    if (lid == 0) {
        xlm[lsz * WPT] = ylm[lsz * WPT] = pad;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Binary search diagonal in the local window
    int ix = 0, iy = 0;
    if (lid != 0) {
        const int k          = lid * WPT * 2;
        const int L          = lsz * WPT;
        const int diag_start = min(k, L) - 1;
        const int diag_end   = max(0, k - L);

        int start = 0, end = diag_start - diag_end;
        while (start <= end) {
            int m = (start + end) >> 1;
            ix = diag_start - m;
            iy = diag_end   + m;
            if (comp(xlm[ix], ylm[iy], dir)) {
                end = m - 1;
                ++ix;
            } else {
                start = m + 1;
                ++iy;
            }
        }
    }

    // Merge elements at the found path intersection
    for (int i = lid*WPT*2, z_end = min(i + WPT*2, z_len); i < z_end; i++) {
        if (comp(xlm[ix], ylm[iy], dir)) {
            zgm[i * z_inc] = xlm[ix];
            ARG(izgm[i * iz_inc] = ixgm[ix * ix_inc]);
            ++ix;
        } else {
            zgm[i * z_inc] = ylm[iy];
            ARG(izgm[i * iz_inc] = iygm[iy * iy_inc]);
            ++iy;
        }
    }
}

#ifndef ROUTINE_ARGSORT
INLINE_FUNC void LocalMultiMerge(
    const int dir,
    const __global real* restrict xgm, const int x_len, const int x_inc,
    const __global real* restrict ygm, const int y_len, const int y_inc,
          __global real*          zgm, const int z_len, const int z_inc,
    LOCAL_PTR int* x_block_start, LOCAL_PTR real* xlm,
    LOCAL_PTR int* y_block_start, LOCAL_PTR real* ylm)
#else
INLINE_FUNC void LocalMultiArgMerge(
    const int dir,
    const int x_len, const __global real* restrict xgm, const int  x_inc,
                     const __global int* restrict ixgm, const int ix_inc,
    const int y_len, const __global real* restrict ygm, const int  y_inc,
                     const __global int* restrict iygm, const int iy_inc,
    const int z_len,       __global real*          zgm, const int  z_inc,
                           __global int*          izgm, const int iz_inc,
    LOCAL_PTR int* x_block_start, LOCAL_PTR real* xlm,
    LOCAL_PTR int* y_block_start, LOCAL_PTR real* ylm)
#endif
{
    const int lid = get_local_id(0);

    // Make sure this is before the sync
    int gix = *x_block_start;
    int giy = *y_block_start;
    int giz = gix + giy + lid*WPT;

    // Load current local window
    const real pad = dir ? SMALLEST : LARGEST;
    #pragma unroll
    for (int _w = 0, i = lid; _w < WPT; _w++, i += get_local_size(0)) {
        xlm[i] = (gix + i < x_len) ? xgm[(gix + i) * x_inc] : pad;
        ylm[i] = (giy + i < y_len) ? ygm[(giy + i) * y_inc] : pad;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Binary search diagonal in the local window for path
    int ix = 0, iy = 0;
    if (lid != 0) {
        const int k = lid * WPT;
        int start = 0, end = k - 1;
        while (start <= end) {
            int m = (start + end) >> 1;
            ix = k - m - 1;
            iy = m;
            if (comp(xlm[ix], ylm[iy], dir)) {
                end = m - 1;
                ++ix;
            } else {
                start = m + 1;
                ++iy;
            }
        }
    }

    // Merge elements at the found path intersection
    #pragma unroll
    for (int _w = 0; _w < WPT; _w++, giz++) {
        real val; ARG(int idx);
        if (comp(xlm[ix], ylm[iy], dir)) {
            val = xlm[ix];
            ARG(idx = ixgm[(gix + ix) * ix_inc]);
            ++ix;
        } else {
            val = ylm[iy];
            ARG(idx = iygm[(giy + iy) * iy_inc]);
            ++iy;
        }
        if (giz < z_len) {
            zgm[giz * z_inc] = val;
            ARG(izgm[giz * iz_inc] = idx);
        }
    }

    // Update for next window
    if (lid == get_local_size(0) - 1) {
        *x_block_start += ix;
        *y_block_start += iy;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

#undef ARG

)"
