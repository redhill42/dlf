CL_PROGRAM R"(

INLINE_FUNC void iamax(const int n, __global real* A, const int lda,
                       LOCAL_PTR singlereal* lmmaxA, LOCAL_PTR int* lmimax)
{
    const int lid = get_local_id(0);
    singlereal maxA = ZERO;
    int imax = 0;

    for (int id = lid; id < n; id += get_local_size(0)) {
        singlereal absA = AbsoluteValue(A[id*lda]);
        if (absA > maxA) {
            maxA = absA;
            imax = id;
        }
    }

    lmmaxA[lid] = maxA;
    lmimax[lid] = imax;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0)/2; s > 0; s >>= 1) {
        if (lid < s) {
            if (lmmaxA[lid + s] >= lmmaxA[lid]) {
                lmmaxA[lid] = lmmaxA[lid + s];
                lmimax[lid] = lmimax[lid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void pivot(const int m, const int n, const int i,
           __global real* A, const int a_offset, const int lda,
           __global int* ipiv, const int ipiv_offset)
{
    const int lid = get_local_id(0);
    int j;

    __local singlereal lmmaxA[WGS];
    __local int lmimax[WGS];

    A += a_offset;
    ipiv += ipiv_offset;

    if (lid == 0 && i == 0) {
        ipiv[0] = 0;
    }

    iamax(m - i, A + i*lda + i, lda, lmmaxA, lmimax);
    singlereal maxA = lmmaxA[0];
    int imax = lmimax[0] + i;

    if (lid == 0) {
        ipiv[i + 1] = imax + 1;
        if (maxA == ZERO) {
            ipiv[0] = i + 1;
        }
    }

    if (i != imax) {
        for (j = lid; j < n; j += get_local_size(0)) {
            real temp = A[i*lda + j];
            A[i*lda + j] = A[imax*lda + j];
            A[imax*lda + j] = temp;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (maxA != ZERO) {
        real aii = A[i*lda + i];
        for (j = lid+i+1; j < m; j += get_local_size(0)) {
            real aji = A[j*lda + i];
            DivideFull(aji, aji, aii);
            A[j*lda + i] = aji;
        }
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void laswp(const int n, __global real* A, const int a_offset, const int lda,
           const int k1, const int k2,
           const __global int* ipiv, const int ip_offset, const int ip_inc)
{
    if (ip_inc == 1) {
        for (int i = k1; i < k2; ++i) {
            int ip = ipiv[i + 1] - 1;
            if (ip != i) {
                for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
                    real temp = A[i*lda + id];
                    A[i*lda + id] = A[ip*lda + id];
                    A[ip*lda + id] = temp;
                }
            }
        }
    } else {
        for (int i = k2-1; i >= k1; --i) {
            int ip = ipiv[i + 1] - 1;
            if (ip != i) {
                for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
                    real temp = A[i*lda + id];
                    A[i*lda + id] = A[ip*lda + id];
                    A[ip*lda + id] = temp;
                }
            }
        }
    }
}

)"
