// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Full version of the kernel with offsets and strided accesses
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xcopy(const int x_size, const __global real* restrict xgm, const int x_offset,
           const int y_size, __global real* ygm, const int y_offset)
{
  if (x_size == 1) {
    real x_value = xgm[x_offset];
    for (int id = get_global_id(0); id < y_size; id += get_global_size(0)) {
      ygm[id + y_offset] = x_value;
    }
  } else if (x_size < y_size) {
    for (int id = get_global_id(0); id < y_size; id += get_global_size(0)) {
      ygm[id + y_offset] = xgm[id % x_size + x_offset];
    }
  } else {
    for (int id = get_global_id(0); id < y_size; id += get_global_size(0)) {
      ygm[id + y_offset] = xgm[id + x_offset];
    }
  }
}

// Faster version of the kernel without offsets and strided accesses. Also assumes that 'n' is
// dividable by 'VW', 'WGS' and 'WPT'.
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XcopyFast(const int n, const __global realV* restrict xgm, __global realV* ygm) {
  #pragma unroll
  for (int _w = 0; _w < WPT; _w++) {
    const int id = _w*get_global_size(0) + get_global_id(0);
    ygm[id] = xgm[id];
  }
}

// Strided version of the kernel with offsets and non-standard stride access
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XcopyStridedL(const int n,
                   const __global real* restrict xgm, const int x_offset,
                   const int x_rank, __constant int* x_shape,
                  __global real* ygm, const int y_offset)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    ygm[id + y_offset] = xgm[unravel(id, x_rank, &x_shape[x_rank], x_shape) + x_offset];
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XcopyStridedR(const int n,
                   const __global real* restrict xgm, const int x_offset,
                   const int y_rank, __constant int* y_shape,
                   __global real* ygm, const int y_offset)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    ygm[unravel(id, y_rank, &y_shape[y_rank], y_shape) + y_offset] = xgm[id + x_offset];
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XcopyStridedLR(const int n,
                    const int x_rank, __constant int* x_shape,
                    const __global real* restrict xgm, const int x_offset,
                    const int y_rank, __constant int* y_shape,
                    __global real* ygm, const int y_offset)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    ygm[unravel(id, y_rank, &y_shape[y_rank], y_shape) + y_offset] =
    xgm[unravel(id, x_rank, &x_shape[x_rank], x_shape) + x_offset];
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xconcat_copy(const int n, const int offset, const int block, const int stride,
                  const __global real* restrict xgm, __global real* ygm)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    const int tmp = id / block;
    const int yid = tmp*stride + (id-tmp*block) + offset;
    ygm[yid] = xgm[id];
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xsplit_copy(const int n, const int offset, const int block, const int stride,
                 const __global real* restrict xgm, __global real* ygm)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    const int tmp = id / block;
    const int xid = tmp*stride + (id-tmp*block) + offset;
    ygm[id] = xgm[xid];
  }
}

)" // End of the C++11 raw string literal
