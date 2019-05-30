// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// Full version of the kernel with offsets and strided accesses
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xclamp(const int n, const real_arg arg_min, const real_arg arg_max,
            __global real* xgm, const int x_offset, const int x_inc) {
  const real min = GetRealArg(arg_min);
  const real max = GetRealArg(arg_max);

  // Loops over the work that needs to be done (allows for an arbitrary number of threads)
  for (int id = get_global_id(0); id<n; id += get_global_size(0)) {
    const int gid = id * x_inc + x_offset;
    xgm[gid] = clamp(xgm[gid], min, max);
  }
}

// Faster version of the kernel without offsets and strided accesses. Also assumes that 'n' is
// dividable by 'VW', 'WGS' and 'WPT'.
__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XclampFast(const int n, const real_arg arg_min, const real_arg arg_max,
               __global realV* xgm) {
  const real min = GetRealArg(arg_min);
  const real max = GetRealArg(arg_max);

  #pragma unroll
  for (int _w = 0; _w < WPT; _w += 1) {
    const int id = _w*get_global_size(0) + get_global_id(0);
    xgm[id] = clamp(xgm[id], min, max);
  }
}

)" // End of the C++11 raw string literal
