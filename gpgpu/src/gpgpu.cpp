#include <numeric>
#include "gpgpu.h"
#include "gpgpu_cl.hpp"
#include "gpgpu_cu.hpp"

namespace gpgpu {

inline bool empty_bitset(std::vector<bool> bitset) {
    return std::accumulate(bitset.begin(), bitset.end(), false, std::logical_or<bool>()) == false;
}

std::vector<bool> parseDeviceFilter(int num_devices, const char* env) {
    enum state_t { Begin, Exclude, Digits, Range };
    state_t state = Begin;
    int beg = 0, num = 0;
    bool excluded = false;
    std::vector<bool> inclusion(num_devices);
    std::vector<bool> exclusion(num_devices);

    if (env == nullptr)
        env = "";
    for (const char* p = env; ; p++) {
        switch (state) {
        case Begin:
            if (*p >= '0' && *p <= '9') {
                state = Digits;
                num = num * 10 + (*p - '0');
            } else if (*p == '-') {
                state = Exclude;
                excluded = true;
            }
            break;

        case Exclude:
            if (*p >= '0' && *p <= '9') {
                state = Digits;
                num = num * 10 + (*p - '0');
            }
            break;

        case Digits:
            if (*p >= '0' && *p <= '9') {
                num = num * 10 + (*p - '0');
            } else if (*p == '-') {
                state = Range;
                beg = num;
                num = 0;
            }
            break;

        case Range:
            if (*p >= '0' && *p <= '9') {
                num = num * 10 + (*p - '0');
            }
            break;
        }

        if (*p == ',' || *p == '\0') {
            auto& v = excluded ? exclusion : inclusion;
            if (state == Digits) {
                if (num > 0 && num <= num_devices)
                    v[num-1] = true;
            } else if (state == Range) {
                auto end = std::min(num, num_devices);
                beg = std::max(1, beg);
                if (beg <= end) {
                    std::fill(v.begin()+beg-1, v.begin()+end, true);
                }
            }

            state = Begin;
            beg = num = 0;
            excluded = false;
        }

        if (*p == '\0')
            break;
    }

    if (empty_bitset(inclusion)) {
        // no explicit inclusion, includes all
        std::fill(inclusion.begin(), inclusion.end(), true);
    }

    for (int i = 0; i < num_devices; i++) {
        // exclude from exclusion
        inclusion[i] = inclusion[i] && !exclusion[i];
    }

    if (empty_bitset(inclusion)) {
        // illegal for empty device filter, includes all
        std::fill(inclusion.begin(), inclusion.end(), true);
    }

    return inclusion;
}

std::vector<bool> parseDeviceFilter(int num_devices) {
    return parseDeviceFilter(num_devices, getenv("GPGPU_DEVICES"));
}

Platform probe() {
    auto cu = gpgpu::cu::probe();
    if (cu != nullptr)
        return Platform(std::move(cu));

    auto cl = gpgpu::cl::probe();
    if (cl != nullptr)
        return Platform(std::move(cl));

    throw RuntimeError("No OpenCL or CUDA platform available");
}

std::vector<Device> Platform::devices(DeviceType type) const {
    auto raw_devices = m_raw->devices(type);
    auto devices = std::vector<Device>();
    for (const auto& dev : raw_devices)
        devices.emplace_back(*this, dev);
    return devices;
}

} // namespace gpgpu
