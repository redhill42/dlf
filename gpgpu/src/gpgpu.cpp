#include <algorithm>
#include <numeric>
#include <atomic>
#include <iostream>
#include <cassert>

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

#if HAS_CUDA
    if (env == nullptr || strcmp(env, "CUDA") == 0) {
        auto cu = gpgpu::cu::probe();
        if (cu != nullptr)
            return Platform(std::move(cu));
    }
#endif

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

static Device select_current_device() {
    static std::vector<Device> global_devices = probe().devices(DeviceType::GPU);
    static std::atomic<size_t> device_index(0); // round-robin select device

    if (global_devices.empty())
        throw NoDeviceFound();

    do {
        size_t id = ++device_index;
        if (id < global_devices.size())
            return global_devices[id];
        if (device_index.compare_exchange_strong(id, 0))
            return global_devices[0];
    } while (true);
}

static thread_local Queue current_queue;

const Queue& current::queue() {
    if (current_queue.id() == 0) {
        Context context = select_current_device().createContext();
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

//---------------------------------------------------------------------------

std::shared_ptr<rawBuffer> rawContext::getTemporaryBuffer(
    size_t& size, size_t& offset, size_t item_size) const
{
    auto data_offset = m_temp_buffer_offset;
    if (data_offset > 0)
        data_offset = ((data_offset - 1)/item_size + 1) * item_size;

    auto data_size = data_offset + size*item_size;
    if (data_size > m_temp_buffer_size) {
        m_temporary_buffer = createBuffer(data_size, BufferAccess::ReadWrite);
        m_temp_buffer_size = data_size;
    }

    size = data_size / item_size;
    offset = data_offset / item_size;
    m_temp_buffer_offset = data_size;
    return m_temporary_buffer;
}

void rawContext::releaseTemporaryBuffer(
    size_t size, size_t offset, size_t item_size) const
{
    if (size * item_size != m_temp_buffer_offset)
        throw std::runtime_error("temporary buffer allocation violation");
    m_temp_buffer_offset = offset * item_size;
}

std::shared_ptr<rawBuffer> rawContext::getSharedBuffer(
    std::string&& content, const rawQueue& queue) const
{
    auto it = m_shared_buffers.find(content);
    if (it != m_shared_buffers.end()) {
        return it->second;
    }

    auto buffer = createBuffer(content.size(), BufferAccess::ReadWrite);
    buffer->write(queue, content.data(), content.size(), 0, nullptr);
    queue.finish();
    m_shared_buffers.emplace(std::move(content), buffer);
    return buffer;
}

} // namespace gpgpu
