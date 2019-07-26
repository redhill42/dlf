#include <algorithm>
#include <numeric>
#include <atomic>
#include <iostream>

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
                if (num >= 0 && num < num_devices) {
                    v[num] = true;
                }
            } else if (state == Range) {
                auto end = std::min(num, num_devices-1);
                if (beg <= end) {
                    std::fill(v.begin()+beg, v.begin()+end+1, true);
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
    auto env = getenv("GPGPU");

    if (env == nullptr || strcmp(env, "CUDA") == 0) {
        auto cu = gpgpu::cu::probe();
        if (cu != nullptr)
            return Platform(std::move(cu));
    }

    if (env == nullptr || strcmp(env, "OpenCL") == 0) {
        auto cl = gpgpu::cl::probe();
        if (cl != nullptr)
            return Platform(std::move(cl));
    }

    if (env == nullptr) {
        throw RuntimeError("No OpenCL or CUDA platform available");
    } else {
        throw RuntimeError("No " + std::string(env) + " platform available");
    }
}

std::vector<Device> Platform::devices(DeviceType type) const {
    auto raw_devices = m_raw->devices(type);
    auto devices = std::vector<Device>();
    for (const auto& dev : raw_devices)
        devices.emplace_back(*this, dev);
    return devices;
}

static std::vector<Context> initialize_global_contexts() {
    auto devices = probe().devices(DeviceType::GPU);
    auto contexts = std::vector<Context>();

    for (size_t id = 0; id < devices.size(); id++) {
        try {
            contexts.push_back(devices[id].createContext());
        } catch (APIError&) {
            std::cerr << "Warning: failed to create context on device #" + std::to_string(id) << std::endl;
        }
    }
    return contexts;
}

static Context select_current_context() {
    static std::vector<Context> global_contexts = initialize_global_contexts();
    static std::atomic<size_t> context_index(0); // round-robin select context

    if (global_contexts.empty())
        throw NoDeviceFound();

    do {
        size_t id = ++context_index;
        if (id < global_contexts.size())
            return global_contexts[id];
        if (context_index.compare_exchange_strong(id, 0))
            return global_contexts[0];
    } while (true);
}

static thread_local Queue current_queue;

const Queue& current::queue() {
    if (current_queue.id() == 0) {
        Context context = select_current_context();
        context.activate();
        current_queue = context.createQueue();
    }
    return current_queue;
}

const Context& current::context() {
    return current::queue().context();
}

current::current(const Queue& queue) {
    if (current_queue.id() != 0)
        current_queue.context().deactivate();
    if (queue.id() != 0)
        queue.context().activate();
    previous_queue = current_queue;
    current_queue = queue;
}

current::~current() {
    if (current_queue.id() != 0)
        current_queue.context().deactivate();
    if (previous_queue.id() != 0)
        previous_queue.context().activate();
    current_queue = previous_queue;
}

} // namespace gpgpu
