// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

//---------------------------------------------------------------------------

#if defined(CUDA) && (PRECISION == 32 || PRECISION == 3232)
  #define log       logf
  #define exp       expf
  #define sqrt      sqrtf
  #define cos       cosf
  #define sin       sinf
  #define hypot     hypotf
  #define atan2     atan2f
  #define copysign  copysignf
#endif

#if defined(ROUTINE_reduce_max)
#  define INIT(x)           SetReal(x, SMALLEST)
#  define MAP_REDUCE(c,x)   Max(c,c,x)
#  define REDUCE(c,x,y)     Max(c,x,y)
#elif defined(ROUTINE_reduce_amax)
#  define INIT(x)           SetToZero(x)
#  define MAP_REDUCE(c,x)   SetToAbsoluteValue(x); Max(c,c,x)
#  define REDUCE(c,x,y)     Max(c,x,y)
#elif defined(ROUTINE_reduce_min)
#  define INIT(x)           SetReal(x, LARGEST)
#  define MAP_REDUCE(c,x)   Min(c,c,x)
#  define REDUCE(c,x,y)     Min(c,x,y)
#elif defined(ROUTINE_reduce_amin)
#  define INIT(x)           SetReal(x, LARGEST)
#  define MAP_REDUCE(c,x)   SetToAbsoluteValue(x); Min(c,c,x)
#  define REDUCE(c,x,y)     Min(c,x,y)
#elif defined(ROUTINE_reduce_sum)
#  define INIT(x)           SetToZero(x)
#  define MAP_REDUCE(c,x)   Add(c,c,x)
#  define REDUCE(c,x,y)     Add(c,x,y)
#elif defined(ROUTINE_reduce_asum)
#  define INIT(x)           SetToZero(x)
#  define MAP_REDUCE(c,x)   SetToAbsoluteValue(x); Add(c,c,x)
#  define REDUCE(c,x,y)     Add(c,x,y)
#elif defined(ROUTINE_reduce_mean)
#  define INIT(x)           SetToZero(x)
#  define MAP_REDUCE(c,x)   Add(c,c,x)
#  define REDUCE(c,x,y)     Add(c,x,y)
#  if PRECISION == 3232 || PRECISION == 6464
#    define FINAL(z,a,n)    z.x = a.x / n; z.y = a.y / n
#  else
#    define FINAL(c,x,n)    c = x / n
#  endif
#elif defined(ROUTINE_reduce_sum_square)
#  define INIT(x)           SetToZero(x)
#  define MAP_REDUCE(c,x)   MultiplyAdd(c,x,x)
#  define REDUCE(c,x,y)     Add(c,x,y)
#elif defined(ROUTINE_reduce_prod)
#  define INIT(x)           SetToOne(x)
#  define MAP_REDUCE(c,x)   Multiply(c,c,x)
#  define REDUCE(c,x,y)     Multiply(c,x,y)

#elif defined(ROUTINE_reduce_log_sum)
#  define INIT(x)           SetToZero(x)
#  define MAP_REDUCE(c,x)   Add(c,c,x)
#  define REDUCE(c,x,y)     Add(c,x,y)
#  if PRECISION == 3232 || PRECISION == 6464
#    define FINAL(z,a,n)            \
       z.x = log(hypot(a.x, a.y));  \
       z.y = atan2(a.y, a.x);
#  else
#    define FINAL(c,x,n)    c = log(x)
#  endif

#elif defined(ROUTINE_reduce_log_sum_exp)
#  define INIT(x)           SetToZero(x)
#  define REDUCE(c,x,y)     Add(c,x,y)
#  if PRECISION == 3232 || PRECISION == 6464
#    define MAP_REDUCE(z,a)         \
       do {                         \
          singlereal e = exp(a.x);  \
          z.x += e * cos(a.x);      \
          z.y += e * sin(a.y);      \
       } while (0)
#    define FINAL(z,a,n)            \
       z.x = log(hypot(a.x, a.y));  \
       z.y = atan2(a.y, a.x);
#  else
#    define MAP_REDUCE(c,x) c += exp(x)
#    define FINAL(c,x,n)    c = log(x)
#  endif

#elif defined(ROUTINE_reduce_nrm2)
#  define INIT(x)           SetToZero(x)
#  define REDUCE(c,x,y)     Add(c,x,y)
#  define MAP_REDUCE(c,a)           \
     do {                           \
       real b = a;                  \
       COMPLEX_CONJUGATE(b);        \
       MultiplyAdd(c,a,b);          \
     } while (0)
#  if PRECISION == 3232 || PRECISION == 6464
#    define FINAL(z,a,n)                            \
       do {                                         \
         singlereal m = hypot(a.x,a.y);             \
         z.x = sqrt((m + a.x) / 2);                 \
         z.y = copysign(sqrt((m - a.x) / 2), a.y);  \
       } while (0)
#  else
#    define FINAL(c,x,n)    c = sqrt(x)
#  endif
#else
#  error "Unsupported reduce routine"
#endif

#ifndef FINAL
#  define FINAL(c,x,n)  c = x
#endif

//---------------------------------------------------------------------------

#define BATCH_WGS 32

INLINE_FUNC void Reduce(const int n, LOCAL_PTR real* lm,
                        const __global real* restrict xgm,
                        __global real* ygm)
{
    const int lid = get_local_id(0);
    const int wgid = get_group_id(0);
    const int wgs = get_local_size(0);
    const int num_groups = get_num_groups(0);

    /* Perform loading and the first steps of the reduction */
    real acc; INIT(acc);
    int id = wgid * wgs + lid;
    while (id < n) {
        real x = xgm[id];
        MAP_REDUCE(acc, x);
        id += wgs * num_groups;
    }
    lm[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Perform reduction in local memory */
    for (int s = wgs/2; s > 0; s >>= 1) {
        if (lid < s)
            REDUCE(lm[lid], lm[lid], lm[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Stores the per-workgroup result */
    if (lid == 0) {
        ygm[wgid] = lm[0];
    }
}

__kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))
void Xreduce(const int n,
             const __global real* restrict xgm, const int x_offset,
             __global real* ygm, const int y_offset)
{
    __local real lm[WGS1];
    Reduce(n, lm, xgm + x_offset, ygm + y_offset);
}

__kernel __attribute__((reqd_work_group_size(BATCH_WGS, 1, 1)))
void XreduceBatched(const int n,
                    const __global real* restrict xgm, const int x_offset,
                    __global real* ygm, const int y_offset)
{
    __local real lm[BATCH_WGS];
    const int batch = get_global_id(1);
    xgm += x_offset + batch*n;
    ygm += y_offset + batch*get_num_groups(0);
    Reduce(n, lm, xgm, ygm);
}

//---------------------------------------------------------------------------

INLINE_FUNC void StridedReduce(
    const int n, LOCAL_PTR real* lm, const int rank, __constant int* shape,
    const __global real* restrict xgm, __global real* ygm)
{
    const int batch = get_global_id(1);
    const int lid = get_local_id(0);
    const int wgid = get_group_id(0);
    const int wgs = get_local_size(0);
    const int num_groups = get_num_groups(0);

    /* Perform loading and the first steps of the reduction */
    real acc; INIT(acc);
    int id = wgid * wgs + lid;
    while (id < n) {
        real x = xgm[unravel(batch*n + id, rank, shape)];
        MAP_REDUCE(acc, x);
        id += wgs * num_groups;
    }
    lm[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Perform reduction in local memory */
    for (int s = wgs/2; s > 0; s >>= 1) {
        if (lid < s)
            REDUCE(lm[lid], lm[lid], lm[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Stores the per-workgroup result */
    if (lid == 0) {
        ygm[wgid] = lm[0];
    }
}

__kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))
void XreduceStrided(
    const int n, const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
    __local real lm[WGS1];
    StridedReduce(n, lm, rank, shape, xgm + x_offset, ygm + y_offset);
}

__kernel __attribute__((reqd_work_group_size(BATCH_WGS, 1, 1)))
void XreduceStridedBatched(
    const int n, const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
    __local real lm[BATCH_WGS];
    xgm += x_offset;
    ygm += y_offset + get_global_id(1) * get_num_groups(0);
    StridedReduce(n, lm, rank, shape, xgm, ygm);
}

//---------------------------------------------------------------------------

__kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
void XreduceEpilogue(const int n,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
    const int batch = get_global_id(1);
    const int lid = get_local_id(0);
    __local real lm[WGS2];

    xgm += x_offset + batch * WGS2*2;
    ygm += y_offset;

    /* Performs the first step of the reduction while loading the data */
    REDUCE(lm[lid], xgm[lid], xgm[lid + WGS2]);
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Perform reduction in local memory */
    for (int s = WGS2/2; s > 0; s >>= 1) {
        if (lid < s)
            REDUCE(lm[lid], lm[lid], lm[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Computes the final result */
    if (lid == 0) {
        FINAL(ygm[batch], lm[0], n);
    }
}

__kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
void XreduceEpilogueStrided(
    const int n, const int rank, __constant int* shape,
    const __global real* restrict xgm, const int x_offset,
    __global real* ygm, const int y_offset)
{
    const int batch = get_global_id(1);
    const int lid = get_local_id(0);
    __local real lm[WGS2];

    xgm += x_offset + batch * WGS2*2;
    ygm += y_offset;

    /* Performs the first step of the reduction while loading the data */
    REDUCE(lm[lid], xgm[lid], xgm[lid + WGS2]);
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Perform reduction in local memory */
    for (int s = WGS2/2; s > 0; s >>= 1) {
        if (lid < s)
            REDUCE(lm[lid], lm[lid], lm[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Computes the final result */
    if (lid == 0) {
        int yid = unravel(batch, rank, shape);
        FINAL(ygm[yid], lm[0], n);
    }
}

)"
