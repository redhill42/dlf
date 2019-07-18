// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

#define DEFINE_BINARY(name, op)                                             \
__kernel __attribute__((reqd_work_group_size(COPY_DIMX, COPY_DIMY, 1)))     \
void name(const int m, const int n, const int channels,                     \
          const __global real* restrict xgm,                                \
          const __global real* restrict ygm,                                \
          __global real* zgm)                                               \
{                                                                           \
  const int rid = get_global_id(0);                                         \
  if (rid < m) {                                                            \
    const real y = ygm[rid % channels];                                     \
    for (int id = get_global_id(1); id < n; id += get_global_size(1)) {     \
      const int offset = rid*n + id;                                        \
      real x = xgm[offset];                                                 \
      real z;                                                               \
      op(z, x, y);                                                          \
      zgm[offset] = z;                                                      \
    }                                                                       \
  }                                                                         \
}

// The scalar division function
#if PRECISION == 3232 || PRECISION == 6464
  #define Divide(c,a,b) \
    do { \
      singlereal num_x = (a.x * b.x) + (a.y * b.y); \
      singlereal num_y = (a.y * b.x) - (a.x * b.y); \
      singlereal denom = (b.x * b.x) + (b.y * b.y); \
      c.x = num_x / denom; \
      c.y = num_y / denom; \
    } while (0)
#else
  #define Divide(c,a,b) c = a / b
#endif

DEFINE_BINARY(Xadd_v, Add)
DEFINE_BINARY(Xsub_v, Subtract)
DEFINE_BINARY(Xmul_v, Multiply)
DEFINE_BINARY(Xdiv_v, Divide)

#if PRECISION != 3232 && PRECISION != 6464

#if INTEGER_PRECISION
  #define xpow(x,y) pow((float)x,(float)y)
  #define xmod(x,y) (x%y)
#elif defined(CUDA) && PRECISION == 32
  #define xpow powf
  #define xmod fmodf
#else
  #define xpow pow
  #define xmod fmod
#endif

#define Max(c,a,b) c = max(a,b)
#define Min(c,a,b) c = min(a,b)
DEFINE_BINARY(Xmax, Max)
DEFINE_BINARY(Xmin, Min)

#define Mod(c,a,b) c = xmod(a,b)
DEFINE_BINARY(Xmod, Mod)

#define Pow(c,a,b) c = xpow(a,b)
DEFINE_BINARY(Xpow, Pow)

#define PRelu(c,a,b) c = a<ZERO ? a*b : a
DEFINE_BINARY(Xprelu, PRelu)

#endif

#if INTEGER_PRECISION
#define BitAnd(c,a,b) c = a & b;
#define BitOr(c,a,b)  c = a | b;
#define BitXor(c,a,b) c = a ^ b;

DEFINE_BINARY(Xbit_and, BitAnd)
DEFINE_BINARY(Xbit_or,  BitOr)
DEFINE_BINARY(Xbit_xor, BitXor)
#endif

)" // End of the C++11 raw string literal
