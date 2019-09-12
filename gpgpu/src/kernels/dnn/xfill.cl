// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xfill(const int n, __global real* xgm, const int x_offset, const real_arg value_arg) {
  const real value = GetRealArg(value_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    xgm[id + x_offset] = value;
  }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XfillStrided(const int n, const int rank, __constant int* shape,
                  __global real* xgm, const int x_offset,
                  const real_arg value_arg)
{
  const real value = GetRealArg(value_arg);
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    xgm[unravel(id, rank, shape) + x_offset] = value;
  }
}

)" // End of the C++11 raw string literal
