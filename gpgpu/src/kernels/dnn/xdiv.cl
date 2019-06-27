// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

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

#define Kernel Xdiv
#define FasterKernel XdivFaster
#define FastestKernel XdivFastest

#define Xform Divide
#define ALLOW_VECTOR 1

)" // End of the C++11 raw string literal
