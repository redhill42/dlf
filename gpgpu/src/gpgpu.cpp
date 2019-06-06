#include "gpgpu.h"
#include "gpgpu_cl.hpp"
#include "gpgpu_cu.hpp"

namespace gpgpu {

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
