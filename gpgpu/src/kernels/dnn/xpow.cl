// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#ifdef CUDA
  #if PRECISION == 16
    #define xpow hpow
  #elif PRECISION == 32
    #define xpow powf
  #else
    #define xpow pow
  #endif
#else
  #define xpow pow
#endif

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xpow(const int n,
    const __global real* restrict xgm, const int x_offset, const int x_inc,
    const __global real* restrict ygm, const int y_offset, const int y_inc,
    __global real* zgm, const int z_offset, const int z_inc)
{
  for (int id = get_global_id(0); id < n; id += get_global_size(0)) {
    real x_value = xgm[id*x_inc + x_offset];
    real y_value = ygm[id*y_inc + y_offset];
    zgm[id*z_inc + z_offset] = xpow(x_value, y_value);
  }
}

)" // End of the C++11 raw string literal
