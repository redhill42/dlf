// Enable loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xwhere(const int n, const int rank,
            const __global char* restrict cgm,
            __constant int* c_shape, const int c_offset,
            const __global real* restrict xgm,
            __constant int* x_shape, const int x_offset,
            const __global real* restrict ygm,
            __constant int* y_shape, const int y_offset,
            __global real* zgm, const int z_offset)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    zgm[id + z_offset] =
      cgm[unravel(id, rank, &c_shape[rank], c_shape) + c_offset]
        ? xgm[unravel(id, rank, &x_shape[rank], x_shape) + x_offset]
        : ygm[unravel(id, rank, &y_shape[rank], y_shape) + y_offset];
  }
}

)" // End of the C++11 raw string literal
