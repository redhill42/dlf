CL_PROGRAM R"(

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
#define UNIFORM(index)                                                      \
    uint r = (uint)high - (uint)low + 1;                                    \
    mwc64x_state_t rng;                                                     \
    MWC64X_SeedStreams(&rng, seed, (n-1)/get_global_size(0)+1);             \
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {     \
        xgm[index + x_offset] = (short)(MWC64X_NextUint(&rng) % r) + low;   \
    }

#elif PRECISION == 10032
#define UNIFORM(index)                                                      \
    mwc64x_state_t rng;                                                     \
    MWC64X_SeedStreams(&rng, seed, (n-1)/get_global_size(0)+1);             \
    uint r = (uint)high - (uint)low + 1;                                    \
    if (r != 0) {                                                           \
        for (int id = get_global_id(0); id < n; id += get_global_size(0)) { \
            xgm[index + x_offset] = (int)(MWC64X_NextUint(&rng) % r) + low; \
        }                                                                   \
    } else {                                                                \
        for (int id = get_global_id(0); id < n; id += get_global_size(0)) { \
            xgm[index + x_offset] = (int)MWC64X_NextUint(&rng);             \
        }                                                                   \
    }

#elif PRECISION == 10064
#define UNIFORM(index)                                                      \
    ulong r = (ulong)high - (ulong)low + 1;                                 \
    mwc64x_state_t rng;                                                     \
    MWC64X_SeedStreams(&rng, seed, 2*((n-1)/get_global_size(0)+1));         \
    for (int id = get_global_id(0); id < n; id += get_global_size(0)) {     \
        ulong u;                                                            \
        u  = (ulong)MWC64X_NextUint(&rng) << 32;                            \
        u |= (ulong)MWC64X_NextUint(&rng);                                  \
        xgm[index + x_offset] = (long)(u % r) + low;                        \
    }

#elif PRECISION == 32 || PRECISION == 64
#define UNIFORM(index)                                                      \
    mwc64x_state_t rng;                                                     \
    MWC64X_SeedStreams(&rng, seed, (n-1)/get_global_size(0)+1);             \
    real factor = (high - low) * UNIFORM_FACTOR;                            \
    for (size_t id = get_global_id(0); id < n; id += get_global_size(0)) {  \
        xgm[index + x_offset] = MWC64X_NextUint(&rng) * factor + low;       \
    }

// https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
#define NORMAL(index)                                                       \
    mwc64x_state_t rng;                                                     \
    MWC64X_SeedStreams(&rng, seed, 2*((n-1)/get_global_size(0)+1));         \
    real u, v, r, s, t;                                                     \
    int v_hot = 0;                                                          \
    for (size_t id = get_global_id(0); id < n; id += get_global_size(0)) {  \
        if (v_hot) {                                                        \
            v_hot = 0;                                                      \
            u = v;                                                          \
        } else {                                                            \
            u = MWC64X_NextUint(&rng) * UNIFORM_FACTOR;                     \
            v = MWC64X_NextUint(&rng) * UNIFORM_FACTOR;                     \
            r = sqrt(-2 * log(u));                                          \
            _sincospi(2*v, s, t);                                           \
            u = r * s;                                                      \
            v = r * t;                                                      \
            v_hot = 1;                                                      \
        }                                                                   \
        xgm[index + x_offset] = u*stdev + mean;                             \
    }

#elif PRECISION == 3232 || PRECISION == 6464
#define UNIFORM(index)                                                      \
    mwc64x_state_t rng;                                                     \
    MWC64X_SeedStreams(&rng, seed, 2*((n-1)/get_global_size(0)+1));         \
    singlereal factor = (high.x - low.x) * UNIFORM_FACTOR;                  \
    for (size_t id = get_global_id(0); id < n; id += get_global_size(0)) {  \
        int idx = index + x_offset;                                         \
        xgm[idx].x = MWC64X_NextUint(&rng) * factor + low.x;                \
        xgm[idx].y = MWC64X_NextUint(&rng) * factor + low.x;                \
    }

#define NORMAL(index)                                                       \
    mwc64x_state_t rng;                                                     \
    MWC64X_SeedStreams(&rng, seed, 2*((n-1)/get_global_size(0)+1));         \
    singlereal u, v, r, s, t;                                               \
    for (size_t id = get_global_id(0); id < n; id += get_global_size(0)) {  \
        int idx = index + x_offset;                                         \
        u = MWC64X_NextUint(&rng) * UNIFORM_FACTOR;                         \
        v = MWC64X_NextUint(&rng) * UNIFORM_FACTOR;                         \
        r = sqrt(-2 * log(u));                                              \
        _sincospi(2*v, s, t);                                               \
        xgm[idx].x = r*s*stdev.x + mean.x;                                  \
        xgm[idx].y = r*t*stdev.x + mean.x;                                  \
    }

#endif

#ifndef NORMAL
#define NORMAL(index)
#endif

//---------------------------------------------------------------------------

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void Xrandom(const int n, __global real* xgm, const int x_offset,
             const ulong seed, const real low, const real high)
{
    if (get_global_id(0) < n) {
        UNIFORM(id)
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XrandomStrided(const int n, const int rank, __constant int* shape,
                    __global real* xgm, const int x_offset,
                    const ulong seed, const real low, const real high)
{
    if (get_global_id(0) < n) {
        UNIFORM(unravel(id, rank, shape))
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XrandomNormal(const int n, __global real* xgm, const int x_offset,
                   const ulong seed, const real mean, const real stdev)
{
    if (get_global_id(0) < n) {
        NORMAL(id);
    }
}

__kernel __attribute__((reqd_work_group_size(WGS, 1, 1)))
void XrandomNormalStrided(const int n, const int rank, __constant int* shape,
                          __global real* xgm, const int x_offset,
                          const ulong seed, const real mean, const real stdev)
{
    if (get_global_id(0) < n) {
        NORMAL(unravel(id, rank, shape))
    }
}

)"
