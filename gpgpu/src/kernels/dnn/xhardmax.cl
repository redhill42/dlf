// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xhardmax(const int n, __global real* xgm, const int x_offset,
              const __global int* restrict arg, const int arg_offset)
{
  const int batch = get_group_id(1);
  const int idx = arg[arg_offset + batch];
  xgm += x_offset + batch*n;

  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    xgm[id] = (id == idx) ? ONE : ZERO;
  }
}

)" // End of the C++11 raw string literal
