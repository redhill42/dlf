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

#define Kernel Xpow
#define Xform(c,a,b) c = xpow(a,b)
#define ALLOW_VECTOR 0

)" // End of the C++11 raw string literal
