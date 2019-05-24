#include "gpgpu.h"

using namespace gpgpu;

extern std::shared_ptr<raw::Platform> probe_cl();
extern std::shared_ptr<raw::Platform> probe_cu();

Platform gpgpu::probe() {
    auto cu = probe_cu();
    if (cu != nullptr)
        return Platform(std::move(cu));

    auto cl = probe_cl();
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
