#ifndef _GPGPU_CU_H
#define _GPGPU_CU_H

#include "gpgpu.h"
#include <cuda.h>
#include <nvrtc.h>
#include <cublas_v2.h>

namespace gpgpu::cu {

class cuPlatform;
class cuDevice;
class cuContext;
class cuQueue;
class cuEvent;
class cuProgram;
class cuKernel;

class cuPlatform final : public raw::Platform {
public:
    APITypes api() const noexcept override {
        return APITypes::CUDA;
    }

    PlatformID id() const noexcept override {
        return 0;
    }

    std::string name() const override {
        return "CUDA";
    }

    std::string vendor() const override {
        return "NVIDIA";
    }

    std::string version() const override;

    std::vector<std::shared_ptr<raw::Device>> devices(DeviceType type) const override;
    std::shared_ptr<raw::Device> device() override;
};

class cuDevice final : public raw::Device {
    CUdevice m_device;

public:
    explicit cuDevice(CUdevice device)
        : m_device(device) {}

    DeviceID id() const noexcept override {
        return static_cast<DeviceID>(m_device);
    }

    std::string version() const override;
    std::string vendor() const override;
    std::string name() const override;
    DeviceType type() const override;

    size_t maxWorkGroupSize() const override {
        return getInfo(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    }

    size_t maxWorkItemDimensions() const override {
        return 3;
    }

    std::vector<size_t> maxWorkItemSizes() const override {
        return {
            getInfo(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
            getInfo(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
            getInfo(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
        };
    }

    uint64_t localMemSize() const override {
        return getInfo(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
    }

    std::string capabilities() const override {
        auto major = getInfo(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        auto minor = getInfo(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        return "SM" + std::to_string(major) + "." + std::to_string(minor);
    }

    uint32_t coreClock() const override {
        return getInfo(CU_DEVICE_ATTRIBUTE_CLOCK_RATE) / 1000;
    }

    uint32_t computeUnits() const override {
        return getInfo(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    }

    uint64_t memorySize() const override;

    uint64_t maxAllocSize() const override {
        return memorySize();
    }

    std::shared_ptr<raw::Context> createContext() override;

private:
    size_t getInfo(CUdevice_attribute info) const;
};

class cuContext final : public raw::Context {
    CUcontext m_context;

public:
    explicit cuContext(CUcontext context)
        : m_context(context) {}

    ~cuContext() override;

    ContextID id() const noexcept override {
        return reinterpret_cast<ContextID>(m_context);
    }

    std::shared_ptr<raw::Program> compileProgram(
        const char *source, const std::vector<std::string> &options) override;
    std::shared_ptr<raw::Program> loadProgram(const std::string& binary) override ;

    std::shared_ptr<raw::Queue> createQueue() override;
    std::shared_ptr<raw::Event> createEvent() override;
    std::shared_ptr<raw::Buffer> createBuffer(size_t size, BufferAccess access) override;
};

class cuEvent final : public raw::Event {
    CUevent m_start, m_end;

public:
    cuEvent(CUevent start, CUevent end)
        : m_start(start), m_end(end) {}

    ~cuEvent() override;

    // Waits for completion of this event (not implemented by CUDA)
    void waitForCompletion() override {}

    float getElapsedTime() override {
        float result = 0.0f;
        cuEventElapsedTime(&result, m_start, m_end);
        return result;
    }

    CUevent start() const noexcept { return m_start; }
    CUevent end() const noexcept { return m_end; }

    static CUevent start(const raw::Event& event) {
        return reinterpret_cast<const cuEvent&>(event).start();
    }

    static CUevent end(const raw::Event& event) {
        return reinterpret_cast<const cuEvent&>(event).end();
    }
};

class cuQueue final : public raw::Queue {
    CUstream m_queue;
    cublasHandle_t m_cublas;

public:
    explicit cuQueue(CUstream queue)
        : m_queue(queue), m_cublas(nullptr) {}

    ~cuQueue() override;

    QueueID id() const noexcept override {
        return reinterpret_cast<QueueID>(m_queue);
    }

    cublasHandle_t getCublasHandle();

    void finish(raw::Event& event) override;
    void finish() override;

    static CUstream* unwrap(raw::Queue& queue) {
        return &reinterpret_cast<cuQueue&>(queue).m_queue;
    }

    static const CUstream* unwrap(const raw::Queue& queue) {
        return &reinterpret_cast<const cuQueue&>(queue).m_queue;
    }
};

class cuBuffer final : public raw::Buffer {
    BufferAccess m_access;
    CUdeviceptr m_buffer;

public:
    explicit cuBuffer(BufferAccess access, CUdeviceptr buffer)
        : m_access(access), m_buffer(buffer) {}

    ~cuBuffer() override;

    void read(raw::Queue& queue, void* host, size_t size, size_t offset) override;
    void write(raw::Queue& queue, const void* host, size_t size, size_t offset) override;
    void copy(raw::Queue& queue, raw::Buffer& dest, size_t size) override;

    static CUdeviceptr* unwrap(raw::Buffer& buffer) {
        return &reinterpret_cast<cuBuffer&>(buffer).m_buffer;
    }

    static const CUdeviceptr* unwrap(const raw::Buffer& buffer) {
        return &reinterpret_cast<const cuBuffer&>(buffer).m_buffer;
    }

    template <typename T>
    static CUdeviceptr* unwrap(gpgpu::Buffer<T>& buffer) {
        return unwrap(buffer.raw());
    }

    template <typename T>
    static const CUdeviceptr* unwrap(const gpgpu::Buffer<T>& buffer) {
        return unwrap(buffer.raw());
    }
};

class cuProgram final : public raw::Program {
    CUmodule m_module;
    std::string m_ir;

public:
    explicit cuProgram(CUmodule module, std::string ir)
        : m_module(module), m_ir(std::move(ir)) {}

    ~cuProgram() override;

    std::string getIR() override { return m_ir; }
    std::shared_ptr<raw::Kernel> getKernel(const char* name) override;
};

class cuKernel final : public raw::Kernel {
    CUfunction m_kernel;
    std::vector<size_t> m_arguments_indices;
    std::vector<char> m_arguments_data;

public:
    explicit cuKernel(CUfunction kernel) : m_kernel(kernel) {}

    void setArgument(size_t index, const void* value, size_t size) override;
    void setArgument(size_t index, const raw::Buffer& buffer) override;
    void launch(raw::Queue& queue,
                const std::vector<size_t>& global,
                const std::vector<size_t>& local,
                raw::Event* event) override;
};

std::shared_ptr<raw::Platform> probe();

} // namespace gpgpu::cu

#endif //_GPGPU_CU_H