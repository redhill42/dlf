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
void XcopyStrided(const int n, const int rank, __constant int* shape,
                  const __global real* restrict xgm, const int x_offset,
                  __global real* ygm, const int y_offset)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    int x_id = x_offset, y_id = y_offset;
    unravel2(id, &x_id, &y_id, rank, shape);
    ygm[y_id] = xgm[x_id];
  }
}

)" // End of the C++11 raw string literal
