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

class clPlatform final : public rawPlatform {
    cl_platform_id m_platform;
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

    std::vector<std::shared_ptr<rawDevice>> devices(DeviceType type) const override;

private:
    std::string getInfo(cl_device_info info) const;
};

class clDevice final : public rawDevice {
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

    std::shared_ptr<rawContext> createContext() const override;

private:
    template <typename T>
    T getInfo(cl_device_info info) const;

    std::string getInfoStr(cl_device_info info) const;

    template <typename T>
    std::vector<T> getInfoVec(cl_device_info info) const;
};

class clContext final : public rawContext {
    cl_device_id m_device;
    cl_context m_context;
public:
    explicit clContext(cl_device_id device, cl_context context)
        : m_device(device), m_context(context) {}

    ~clContext() override;

    ContextID id() const noexcept override {
        return reinterpret_cast<ContextID>(m_context);
    }

    void activate() const override {}
    void deactivate() const override {}

    std::shared_ptr<rawProgram> compileProgram(
        const char* source, const std::vector<std::string>& options) const override;
    std::shared_ptr<rawProgram> loadProgram(const std::string& binary) const override;

    std::shared_ptr<rawQueue> createQueue() const override;
    std::shared_ptr<rawEvent> createEvent() const override;
    std::shared_ptr<rawBuffer> createBuffer(size_t size, BufferAccess access) const override;
};

class clQueue final : public rawQueue {
    cl_command_queue m_queue;
public:
    explicit clQueue(cl_command_queue queue)
        : m_queue(queue) {}

    ~clQueue() override;

    QueueID id() const noexcept override {
        return reinterpret_cast<QueueID>(m_queue);
    }

    void finish(rawEvent&) const override;
    void finish() const override;

    static cl_command_queue* unwrap(rawQueue& queue) {
        return &reinterpret_cast<clQueue&>(queue).m_queue;
    }

    static const cl_command_queue* unwrap(const rawQueue& queue) {
        return &reinterpret_cast<const clQueue&>(queue).m_queue;
    }
};

class clEvent final : public rawEvent {
    cl_event m_event = nullptr;
public:
    ~clEvent() override;

    void waitForCompletion() const override;
    float getElapsedTime() const override;

    static cl_event* unwrap(rawEvent* event) {
        return event== nullptr ? nullptr : &reinterpret_cast<clEvent*>(event)->m_event;
    }

    static const cl_event* unwrap(const rawEvent* event) {
        return event==nullptr ? nullptr : &reinterpret_cast<const clEvent*>(event)->m_event;
    }
};

class clBuffer final : public rawBuffer {
    BufferAccess m_access;
    cl_mem m_buffer;
public:
    clBuffer(BufferAccess access, cl_mem buffer)
        : m_access(access), m_buffer(buffer) {}

    ~clBuffer() override;

    void read(const rawQueue& queue, void* host, size_t size,
        size_t offset, rawEvent* event) const override;
    void write(const rawQueue& queue, const void* host, size_t size,
        size_t offset, rawEvent* event) override;
    void copyTo(const rawQueue& queue, rawBuffer& dest, size_t size,
        size_t src_offset, size_t dst_offset, rawEvent* event) const override;

    static cl_mem* unwrap(rawBuffer& raw) {
        return &reinterpret_cast<clBuffer&>(raw).m_buffer;
    }

    static const cl_mem* unwrap(const rawBuffer& raw) {
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

class clProgram final : public rawProgram {
    cl_program m_program;
public:
    explicit clProgram(cl_program program) : m_program(program) {}

    ~clProgram() override;

    std::string getIR() const override;
    std::shared_ptr<rawKernel> getKernel(const char* name) const override;
};

class clKernel final : public rawKernel {
    cl_kernel m_kernel;
public:
    explicit clKernel(cl_kernel kernel)
        : m_kernel(kernel) {}

    ~clKernel() override;

    uint64_t localMemoryUsage(const rawDevice& device) const override;

    void setArgument(size_t index, const void* value, size_t size) const override;
    void setArgument(size_t index, const rawBuffer& buffer) const override;
    void setLocalMemorySize(size_t size) const override;

    void launch(const rawQueue& queue,
                const std::vector<size_t>& global,
                const std::vector<size_t>& local,
                rawEvent* event) const override;
};

std::shared_ptr<rawPlatform> probe();

}} // namespace gpgpu::cl

#endif //_GPGPU_CL_H
