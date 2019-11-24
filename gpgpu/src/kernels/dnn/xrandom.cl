CL_PROGRAM R"(

#if defined(CUDA)
  typedef unsigned int uint;
  typedef unsigned long ulong;
  #define STATIC __device__
#else
  #define STATIC static
#endif

struct pcg_state_setseq_64 {    // Internals are *Private*
    ulong mult;                 // The multiplier
    ulong plus;                 // Controls which RNG sequence (stream) is
                                // selected. Must *always* be odd.
    ulong state;                // RNG state. All values are possible.
};

typedef struct pcg_state_setseq_64 pcg32_random_t;

STATIC uint pcg32_next(pcg32_random_t* rng);

INLINE_FUNC void pcg32_seed(pcg32_random_t* rng, ulong seed, ulong stream) {
    rng->mult = 6364136223846793005UL;
    rng->plus = (stream << 1) | 1;
    rng->state = seed + rng->plus;
    pcg32_next(rng);
}

INLINE_FUNC uint pcg32_next(pcg32_random_t* rng) {
    ulong x = rng->state;
    rng->state = x * rng->mult + rng->plus;
    uint y = ((x >> 18u) ^ x) >> 27u;
    uint rot = x >> 59u;
    return (y >> rot) | (y << ((-rot) & 31));
}

STATIC void pcg32_advance(uint delta, ulong* p_mult, ulong* p_plus) {
    ulong cur_mult = *p_mult;
    ulong cur_plus = *p_plus;
    ulong acc_mult = 1;
    ulong acc_plus = 0;
    while (delta > 0) {
        if (delta & 1) {
            acc_mult *= cur_mult;
            acc_plus = acc_plus*cur_mult + cur_plus;
        }
        cur_plus = (cur_mult+1)*cur_plus;
        cur_mult *= cur_mult;
        delta >>= 1;
    }
    *p_mult = acc_mult;
    *p_plus = acc_plus;
}

INLINE_FUNC void pcg32_skip(pcg32_random_t* rng, uint delta) {
    ulong cur_mult = rng->mult;
    ulong cur_plus = rng->plus;
    pcg32_advance(delta, &cur_mult, &cur_plus);
    rng->state = rng->state * cur_mult + cur_plus;
}

INLINE_FUNC void pcg32_leapfrog(pcg32_random_t* rng, uint delta) {
    pcg32_advance(delta, &rng->mult, &rng->plus);
}

INLINE_FUNC void pcg32_seed_streams(pcg32_random_t* rng, ulong seed, ulong stream) {
    pcg32_seed(rng, seed, stream);
    pcg32_skip(rng, get_global_id(0));
    pcg32_leapfrog(rng, get_global_size(0));
}

/*=========================================================================*/

#if PRECISION == 32 || PRECISION == 3232
#  define UNIFORM_FACTOR 2.328306437e-10f
#  define PI 3.1415927f
#elif PRECISION == 64 || PRECISION == 6464
#  define UNIFORM_FACTOR 2.3283064365386962890625e-10
#  define PI 3.141592653589793
#endif

#if defined(CUDA)
#define _sincospi(x, s, c)  sincospi(x, &s, &c)
#else
#define _sincospi(x, s, c)  s = sincos(x*PI, &c)
#endif

#if PRECISION == 10016
#define UNIFORM_DISTRIBUTION(index)                                         \
    uint r = (uint)high - (uint)low + 1;                                    \
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {     \
        xgm[index] = (short)(pcg32_next(&rng) % r) + low;                   \
    }

#elif PRECISION == 10032
#define UNIFORM_DISTRIBUTION(index)                                         \
    uint r = (uint)high - (uint)low + 1;                                    \
    if (r != 0) {                                                           \
        for (int id = get_global_id(0); id < n; id += get_global_size(0)) { \
            xgm[index] = (int)(pcg32_next(&rng) % r) + low;                 \
        }                                                                   \
    } else {                                                                \
        for (int id = get_global_id(0); id < n; id += get_global_size(0)) { \
            xgm[index] = (int)pcg32_next(&rng);                             \
        }                                                                   \
    }

#elif PRECISION == 10064
#define UNIFORM_DISTRIBUTION(index)                                         \
    ulong r = (ulong)high - (ulong)low + 1;                                 \
    if (r != 0) {                                                           \
        for (int id = get_global_id(0); id < n; id += get_global_size(0)) { \
            ulong u = ((ulong)pcg32_next(&rng) << 32) | pcg32_next(&rng);   \
            xgm[index] = (long)(u % r) + low;                               \
        }                                                                   \
    } else {                                                                \
        for (int id = get_global_id(0); id < n; id += get_global_size(0)) { \
            ulong u = ((ulong)pcg32_next(&rng) << 32) | pcg32_next(&rng);   \
            xgm[index] = u;                                                 \
        }                                                                   \
    }

#elif PRECISION == 32 || PRECISION == 64
#define UNIFORM_DISTRIBUTION(index)                                         \
    real factor = (high - low) * UNIFORM_FACTOR;                            \
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {     \
        xgm[index] = pcg32_next(&rng) * factor + low;                       \
    }

// https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
#define NORMAL_DISTRIBUTION(index)                                          \
    real u, v, r, s, t;                                                     \
    int v_hot = 0;                                                          \
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {     \
        if (v_hot) {                                                        \
            v_hot = 0;                                                      \
            u = v;                                                          \
        } else {                                                            \
            u = pcg32_next(&rng) * UNIFORM_FACTOR;                          \
            v = pcg32_next(&rng) * UNIFORM_FACTOR;                          \
            r = sqrt(-2 * log(u));                                          \
            _sincospi(2*v, s, t);                                           \
            u = r * s;                                                      \
            v = r * t;                                                      \
            v_hot = 1;                                                      \
        }                                                                   \
        xgm[index] = u*stdev + mean;                                        \
    }

#elif PRECISION == 3232 || PRECISION == 6464
#define UNIFORM_DISTRIBUTION(index)                                         \
    singlereal factor = (high.x - low.x) * UNIFORM_FACTOR;                  \
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {     \
        int idx = index;                                                    \
        xgm[idx].x = pcg32_next(&rng) * factor + low.x;                     \
        xgm[idx].y = pcg32_next(&rng) * factor + low.x;                     \
    }

#define NORMAL_DISTRIBUTION(index)                                          \
    singlereal u, v, r, s, t;                                               \
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {     \
        int idx = index;                                                    \
        u = pcg32_next(&rng) * UNIFORM_FACTOR;                              \
        v = pcg32_next(&rng) * UNIFORM_FACTOR;                              \
        r = sqrt(-2 * log(u));                                              \
        _sincospi(2*v, s, t);                                               \
        xgm[idx].x = r*s*stdev.x + mean.x;                                  \
        xgm[idx].y = r*t*stdev.x + mean.x;                                  \
    }

#endif

#ifndef NORMAL_DISTRIBUTION
#define NORMAL_DISTRIBUTION(index)
#endif

/*=========================================================================*/

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xrandom(const int n, __global real* xgm, const int x_offset,
             const ulong seed, const ulong stream,
             const real low, const real high)
{
    if (get_global_id(0) < n) {
        pcg32_random_t rng;
        pcg32_seed_streams(&rng, seed, stream);
        xgm += x_offset;
        UNIFORM_DISTRIBUTION(id)
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XrandomStrided(const int n, const int rank, __constant int* shape,
                    __global real* xgm, const int x_offset,
                    const ulong seed, const ulong stream,
                    const real low, const real high)
{
    if (get_global_id(0) < n) {
        pcg32_random_t rng;
        pcg32_seed_streams(&rng, seed, stream);
        xgm += x_offset;
        UNIFORM_DISTRIBUTION(unravel(id, rank, shape))
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XrandomNormal(const int n, __global real* xgm, const int x_offset,
                   const ulong seed, const ulong stream,
                   const real mean, const real stdev)
{
    if (get_global_id(0) < n) {
        pcg32_random_t rng;
        pcg32_seed_streams(&rng, seed, stream);
        xgm += x_offset;
        NORMAL_DISTRIBUTION(id);
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XrandomNormalStrided(const int n, const int rank, __constant int* shape,
                          __global real* xgm, const int x_offset,
                          const ulong seed, const ulong stream,
                          const real mean, const real stdev)
{
    if (get_global_id(0) < n) {
        pcg32_random_t rng;
        pcg32_seed_streams(&rng, seed, stream);
        xgm += x_offset;
        NORMAL_DISTRIBUTION(unravel(id, rank, shape))
    }
}

)"
