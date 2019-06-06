#ifndef _GPGPU_CL_H
#define _GPGPU_CL_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings
#define CL_SILENCE_DEPRECATION
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "gpgpu.h"

namespace gpgpu { namespace cl {

class clPlatform;
class clDevice;
class clContext;
class clQueue;
class clEvent;
class clProgram;
class clKernel;

class clPlatform final : public raw::Platform {
    cl_platform_id m_platform;
    mutable std::shared_ptr<raw::Device> m_default_device;
public:
    explicit clPlatform(cl_platform_id id) : m_platform(id) {}

    PlatformID id() const noexcept override {
        return reinterpret_cast<PlatformID>(m_platform);
    }

    APIType api() const noexcept override {
        return APIType::OpenCL;
    }

    std::string name() const override {
        return getInfo(CL_PLATFORM_NAME);
    }

    std::string vendor() const override {
        return getInfo(CL_PLATFORM_VENDOR);
    }

    std::string version() const override {
        return getInfo(CL_PLATFORM_VERSION);
    }

    std::vector<std::shared_ptr<raw::Device>> devices(DeviceType type) const override;
    std::shared_ptr<raw::Device> device() const override;

private:
    std::string getInfo(cl_device_info info) const;
};

class clDevice final : public raw::Device {
    cl_device_id m_device;
public:
    explicit clDevice(cl_device_id id) : m_device(id) {}

    ~clDevice() override;

    DeviceID id() const noexcept override {
        return reinterpret_cast<DeviceID>(m_device);
    }

    std::string version() const override {
        return getInfoStr(CL_DEVICE_VERSION);
    }

    std::string vendor() const override {
        return getInfoStr(CL_DEVICE_VENDOR);
    }

    std::string name() const override {
        return getInfoStr(CL_DEVICE_NAME);
    }

    DeviceType type() const override {
        switch (getInfo<cl_device_type>(CL_DEVICE_TYPE)) {
        case CL_DEVICE_TYPE_CPU:
            return DeviceType::CPU;
        case CL_DEVICE_TYPE_GPU:
            return DeviceType::GPU;
        case CL_DEVICE_TYPE_ACCELERATOR:
            return DeviceType::Accelerator;
        default:
            return DeviceType::Unknown;
        }
    }

    size_t maxWorkGroupSize() const override {
        return getInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE);
    }

    size_t maxWorkItemDimensions() const override {
        return getInfo<cl_uint>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
    }

    std::vector<size_t> maxWorkItemSizes() const override {
        return getInfoVec<size_t>(CL_DEVICE_MAX_WORK_ITEM_SIZES);
    }

    uint64_t localMemSize() const override {
        return getInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE);
    }

    bool supportsFP64() const override {
        return capabilities().find("cl_khr_fp64") != std::string::npos;
    }

    bool supportsFP16() const override {
        if (name() == "Mali-T628")
            return true; // supports fp16 but not cl_khr_fp16 officially
        return capabilities().find("cl_khr_fp16");
    }

    std::string capabilities() const override {
        return getInfoStr(CL_DEVICE_EXTENSIONS);
    }

    uint32_t coreClock() const override {
        return getInfo<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY);
    }

    uint32_t computeUnits() const override {
        return getInfo<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS);
    }

    uint64_t memorySize() const override {
        return getInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE);
    }

    uint64_t maxAllocSize() const override {
        return getInfo<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    }

    std::shared_ptr<raw::Context> createContext() const override;

private:
    template <typename T>
    T getInfo(cl_device_info info) const;

    std::string getInfoStr(cl_device_info info) const;

    template <typename T>
    std::vector<T> getInfoVec(cl_device_info info) const;
};

class clContext final : public raw::Context {
    cl_device_id m_device;
    cl_context m_context;
public:
    explicit clContext(cl_device_id device, cl_context context)
        : m_device(device), m_context(context) {}

    ~clContext() override;

    ContextID id() const noexcept override {
        return reinterpret_cast<ContextID>(m_context);
    }

    std::shared_ptr<raw::Program> compileProgram(
        const char* source, const std::vector<std::string>& options) const override;
    std::shared_ptr<raw::Program> loadProgram(const std::string& binary) const override;

    std::shared_ptr<raw::Queue> createQueue() const override;
    std::shared_ptr<raw::Event> createEvent() const override;
    std::shared_ptr<raw::Buffer> createBuffer(size_t size, BufferAccess access) const override;
};

class clQueue final : public raw::Queue {
    cl_command_queue m_queue;
public:
    explicit clQueue(cl_command_queue queue)
        : m_queue(queue) {}

    ~clQueue() override;

    QueueID id() const noexcept override {
        return reinterpret_cast<QueueID>(m_queue);
    }

    void finish(raw::Event&) const override;
    void finish() const override;

    static cl_command_queue* unwrap(raw::Queue& queue) {
        return &reinterpret_cast<clQueue&>(queue).m_queue;
    }

    static const cl_command_queue* unwrap(const raw::Queue& queue) {
        return &reinterpret_cast<const clQueue&>(queue).m_queue;
    }
};

class clEvent final : public raw::Event {
    cl_event m_event = nullptr;
public:
    ~clEvent() override;

    void waitForCompletion() const override;
    float getElapsedTime() const override;

    static cl_event* unwrap(raw::Event* event) {
        return event== nullptr ? nullptr : &reinterpret_cast<clEvent*>(event)->m_event;
    }

    static const cl_event* unwrap(const raw::Event* event) {
        return event==nullptr ? nullptr : &reinterpret_cast<const clEvent*>(event)->m_event;
    }
};

class clBuffer final : public raw::Buffer {
    BufferAccess m_access;
    cl_mem m_buffer;
public:
    clBuffer(BufferAccess access, cl_mem buffer)
        : m_access(access), m_buffer(buffer) {}

    ~clBuffer() override;

    void read(const raw::Queue& queue, void* host, size_t size, size_t offset, raw::Event* event) const override;
    void write(const raw::Queue& queue, const void* host, size_t size, size_t offset, raw::Event* event) override;
    void copyTo(const raw::Queue& queue, raw::Buffer& dest, size_t size, raw::Event* event) const override;

    static cl_mem* unwrap(raw::Buffer& raw) {
        return &reinterpret_cast<clBuffer&>(raw).m_buffer;
    }

    static const cl_mem* unwrap(const raw::Buffer& raw) {
        return &reinterpret_cast<const clBuffer&>(raw).m_buffer;
    }

    template <typename T>
    static cl_mem* unwrap(gpgpu::Buffer<T>& buffer) {
        return unwrap(buffer.raw());
    }

    template <typename T>
    static const cl_mem* unwrap(const gpgpu::Buffer<T>& buffer) {
        return unwrap(buffer.raw());
    }
};

class clProgram final : public raw::Program {
    cl_program m_program;
public:
    explicit clProgram(cl_program program) : m_program(program) {}

    ~clProgram() override;

    std::string getIR() const override;
    std::shared_ptr<raw::Kernel> getKernel(const char* name) const override;
};

class clKernel final : public raw::Kernel {
    cl_kernel m_kernel;
public:
    explicit clKernel(cl_kernel kernel)
        : m_kernel(kernel) {}

    ~clKernel() override;

    uint64_t localMemoryUsage(const raw::Device& device) const override;

    void setArgument(size_t index, const void* value, size_t size) const override;
    void setArgument(size_t index, const raw::Buffer& buffer) const override;

    void launch(const raw::Queue& queue,
                const std::vector<size_t>& global,
                const std::vector<size_t>& local,
                raw::Event* event) const override;
};

std::shared_ptr<raw::Platform> probe();

}} // namespace gpgpu::cl

#endif //_GPGPU_CL_H
