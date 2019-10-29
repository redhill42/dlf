CL_PROGRAM R"(

INLINE_FUNC int norm_index(int id, int max_item) {
    if (id < 0)
        id += max_item;
    return clamp(id, 0, max_item-1);
}

__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
void Xgather(const int m, const int n, const int chunk, const int max_item,
             const __global real* restrict xgm, const int x_offset,
             const __global int* restrict igm, const int i_offset,
             __global real* ygm, const int y_offset)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    if (i < m && j < n) {
        const int id = norm_index(igm[i_offset + j], max_item);
        xgm += x_offset + i*chunk*max_item + id*chunk;
        ygm += y_offset + (i*n + j)*chunk;
        for (int k = 0; k < chunk; k++) {
            ygm[k] = xgm[k];
        }
    }
}

__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))
void XgatherStrided(
    const int m, const int n, const int chunk, const int max_item,
    const int x_rank, __constant int* x_shape, const __global real* restrict xgm, const int x_offset,
    const int i_rank, __constant int* i_shape, const __global int* restrict igm, const int i_offset,
    const int y_rank, __constant int* y_shape, __global real* ygm, const int y_offset)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    if (i < m && j < n) {
        const int id = norm_index(igm[unravel(j, i_rank, i_shape) + i_offset], max_item);
        const int x_off = i*chunk*max_item + id*chunk;
        const int y_off = (i*n + j)*chunk;
        for (int k = 0; k < chunk; k++) {
            const int x_id = unravel(x_off+k, x_rank, x_shape) + x_offset;
            const int y_id = unravel(y_off+k, y_rank, y_shape) + y_offset;
            ygm[y_id] = xgm[x_id];
        }
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xgather_elements(
    const int n, const int max_item,
    const int i_stride1, const int i_stride2,
    const int x_stride1, const int x_stride2,
    const __global real* restrict xgm, const int x_offset,
    const __global int* restrict igm, const int i_offset,
    __global real* ygm, const int y_offset)
{
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
        const int tmp = norm_index(igm[id + i_offset], max_item);
        const int x_id = (id % i_stride1) + (tmp * x_stride1) + (id / i_stride2 * x_stride2);
        ygm[id + y_offset] = xgm[x_id + x_offset];
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xgather_elementsStrided(
    const int n, const int max_item, const int rank,
    const int i_stride1, const int i_stride2,
    const int x_stride1, const int x_stride2,
    __constant int* x_shape,
    const __global real* restrict xgm, const int x_offset,
    __constant int* i_shape,
    const __global int* restrict igm, const int i_offset,
    __global real* ygm, const int y_offset)
{
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
        int i_id = i_offset, y_id = y_offset;
        unravel2(id, &i_id, &y_id, rank, i_shape);

        const int tmp = norm_index(igm[i_id], max_item);
        const int x_id = (id % i_stride1) + (tmp * x_stride1) + (id / i_stride2 * x_stride2);
        ygm[y_id] = xgm[unravel(x_id, rank, x_shape) + x_offset];
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xscatter_elements(
    const int n, const int max_item,
    const int i_stride1, const int i_stride2,
    const int x_stride1, const int x_stride2,
    __global real* xgm, const int x_offset,
    const __global int* restrict igm, const int i_offset,
    const __global real* restrict ygm, const int y_offset)
{
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
        const int tmp = norm_index(igm[id + i_offset], max_item);
        const int x_id = (id % i_stride1) + (tmp * x_stride1) + (id / i_stride2 * x_stride2);
        xgm[x_id + x_offset] = ygm[id + y_offset];
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xscatter_elementsStrided(
    const int n, const int max_item, const int rank,
    const int i_stride1, const int i_stride2,
    const int x_stride1, const int x_stride2,
    __constant int* x_shape,
    __global real* xgm, const int x_offset,
    __constant int* i_shape,
    const __global int* restrict igm, const int i_offset,
    const __global real* restrict ygm, const int y_offset)
{
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
        int i_id = i_offset, y_id = y_offset;
        unravel2(id, &i_id, &y_id, rank, i_shape);

        const int tmp = norm_index(igm[i_id], max_item);
        const int x_id = (id % i_stride1) + (tmp * x_stride1) + (id / i_stride2 * x_stride2);
        xgm[unravel(x_id, rank, x_shape) + x_offset] = ygm[y_id];
    }
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void Xgather_nd(
    const int k, const int chunk,
    const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    const __global int* restrict igm, const int i_offset,
    __global real* ygm, const int y_offset)
{
    const int i = get_global_id(0);

    int offset = 0, dim = 1;
    igm = &igm[i*k + i_offset];
    for (int j = 0; j < k; j++) {
        offset = offset*dim + norm_index(igm[j], shape[j]);
        dim = shape[j];
    }
    offset *= chunk;

    xgm += x_offset + offset;
    ygm += y_offset + i*chunk;
    for (int j = 0; j < chunk; j++) {
        ygm[j] = xgm[j];
    }
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void Xgather_ndStrided(
    const int k, const int chunk,
    const int x_rank, __constant int* x_shape,
    const __global real* restrict xgm, const int x_offset,
    const int i_rank, __constant int* i_shape,
    const __global int* restrict igm, const int i_offset,
    const int y_rank, __constant int* y_shape,
    __global real* ygm, const int y_offset)
{
    const int i = get_global_id(0);

    int offset = 0, dim = 1;
    for (int j = 0; j < k; j++) {
        const int tmp = igm[unravel(i*k+j, i_rank, i_shape) + i_offset];
        offset = offset*dim + norm_index(tmp, x_shape[j]);
        dim = x_shape[j];
    }
    offset *= chunk;

    for (int j = 0; j < chunk; j++) {
        const int x_id = unravel(offset+j, x_rank, x_shape) + x_offset;
        const int y_id = unravel(i*chunk+j, y_rank, y_shape) + y_offset;
        ygm[y_id] = xgm[x_id];
    }
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void Xscatter_nd(
    const int k, const int chunk,
    const int rank, __constant int* shape,
    __global real* xgm, const int x_offset,
    const __global int* restrict igm, const int i_offset,
    const __global real* ygm, const int y_offset)
{
    const int i = get_global_id(0);

    int offset = 0, dim = 1;
    igm += i_offset + i*k;
    for (int j = 0; j < k; j++) {
        offset = offset*dim + norm_index(igm[j], shape[j]);
        dim = shape[j];
    }
    offset *= chunk;

    xgm += x_offset + offset;
    ygm += y_offset + i*chunk;
    for (int j = 0; j < chunk; j++) {
        xgm[j] = ygm[j];
    }
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void Xscatter_ndStrided(
    const int k, const int chunk,
    const int x_rank, __constant int* x_shape,
    __global real* xgm, const int x_offset,
    const int i_rank, __constant int* i_shape,
    const __global int* restrict igm, const int i_offset,
    const int y_rank, __constant int* y_shape,
    const __global real* restrict ygm, const int y_offset)
{
    const int i = get_global_id(0);

    int offset = 0, dim = 1;
    for (int j = 0; j < k; j++) {
        const int tmp = igm[unravel(i*k+j, i_rank, i_shape) + i_offset];
        offset = offset*dim + norm_index(tmp, x_shape[j]);
        dim = x_shape[j];
    }
    offset *= chunk;

    for (int j = 0; j < chunk; j++) {
        const int x_id = unravel(offset+j, x_rank, x_shape) + x_offset;
        const int y_id = unravel(i*chunk+j, y_rank, y_shape) + y_offset;
        xgm[y_id] = ygm[x_id];
    }
}

INLINE_FUNC void Xgather_indices_row_major(
    const int m, const int n, const int rank, __constant int* dims,
    __global const int* restrict indices, __global int* output)
{
    for (int i = get_global_id(0); i < m; i += get_global_size(0)) {
        int prev = (i == 0) ? 0 : indices[i - 1];
        int curr = indices[i];
        if (curr != prev) {
            int temp = i;
            for (int j = rank; --j >= 0; ) {
                output[n*j + prev] = temp % dims[j];
                temp /= dims[j];
            }
        }
    }
}

INLINE_FUNC void Xgather_indices_col_major(
    const int m, const int n, const int rank, __constant int* dims,
    __global const int* restrict indices, __global int* output)
{
    for (int i = get_global_id(0); i < m; i += get_global_size(0)) {
        int prev = (i == 0) ? 0 : indices[i - 1];
        int curr = indices[i];
        if (curr != prev) {
            int temp = i;
            for (int j = rank; --j >= 0; ) {
                output[prev*rank + j] = temp % dims[j];
                temp /= dims[j];
            }
        }
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xgather_indices(
    const int m, const int n, const int row_major,
    const int rank, __constant int* dims,
    __global const int* restrict indices, const int indices_offset,
    __global int* output, const int output_offset)
{
    if (row_major)
        Xgather_indices_row_major(m, n, rank, dims, indices+indices_offset, output+output_offset);
    else
        Xgather_indices_col_major(m, n, rank, dims, indices+indices_offset, output+output_offset);
}

)"
