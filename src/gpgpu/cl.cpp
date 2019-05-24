#include <numeric>
#include <iostream>
#include "gpgpu.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings
#define CL_SILENCE_DEPRECATION
#if defined(__APPLE__) || defined(__MACOSX)
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

using namespace gpgpu;

static void check(cl_int status, const std::string& where) {
    if (status != CL_SUCCESS) {
        throw APIError(static_cast<int>(status), where,
            "OpenCL error: " + where + ": " + std::to_string(status));
    }
}

static void checkDtor(cl_int status, const std::string& where) {
    if (status != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error: %s: %d (ignoring)\n", where.c_str(), status);
    }
}

static std::string trimCallString(const char* where) {
    const char* paren = strchr(where, '(');
    if (paren) {
        return std::string(where, paren);
    } else {
        return std::string(where);
    }
}

#define CheckError(call) check(call, trimCallString(#call))
#define CheckErrorDtor(call) checkDtor(call, trimCallString(#call))

class clPlatform;
class clDevice;
class clContext;
class clQueue;
class clEvent;
class clProgram;
class clKernel;

class clPlatform final : public raw::Platform {
    cl_platform_id m_platform;
    std::shared_ptr<raw::Device> m_default_device;

public:
    explicit clPlatform(cl_platform_id id) : m_platform(id) {}

    PlatformID id() const noexcept override {
        return reinterpret_cast<PlatformID>(m_platform);
    }

    APITypes api() const noexcept override {
        return APITypes::OpenCL;
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
    std::shared_ptr<raw::Device> device() override;

private:
    std::string getInfo(cl_device_info info) const {
        size_t bytes = 0;
        CheckError(clGetPlatformInfo(m_platform, info, 0, nullptr, &bytes));
        std::string result;
        result.resize(bytes);
        CheckError(clGetPlatformInfo(m_platform, info, bytes, &result[0], nullptr));
        result.resize(strlen(result.c_str())); // remove trailing '\0'
        return result;
    }
};

class clDevice final : public raw::Device {
    cl_device_id m_device;

public:
    explicit clDevice(cl_device_id id)
        : m_device(id) {}

    ~clDevice() override {
        if (m_device) {
            CheckErrorDtor(clReleaseDevice(m_device));
        }
    }

    DeviceID id() const noexcept override {
        return reinterpret_cast<DeviceID>(m_device);
    }

    std::string version() const override {
        return getInfo<std::string>(CL_DEVICE_VERSION);
    }

    std::string vendor() const override {
        return getInfo<std::string>(CL_DEVICE_VENDOR);
    }

    std::string name() const override {
        return getInfo<std::string>(CL_DEVICE_NAME);
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

    std::string capabilities() const override {
        return getInfo<std::string>(CL_DEVICE_EXTENSIONS);
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

    std::shared_ptr<raw::Context> createContext() override;

private:
    template <typename T>
    T getInfo(cl_device_info info) const {
        T result{};
        CheckError(clGetDeviceInfo(m_device, info, sizeof(T), &result, nullptr));
        return result;
    }

    template <>
    std::string getInfo(cl_device_info info) const {
        size_t bytes = 0;
        CheckError(clGetDeviceInfo(m_device, info, 0, nullptr, &bytes));
        std::string result;
        result.resize(bytes);
        CheckError(clGetDeviceInfo(m_device, info, bytes, &result[0], nullptr));
        result.resize(strlen(result.c_str())); // removes any trailing '\0'
        return result;
    }

    template <typename T>
    std::vector<T> getInfoVec(cl_device_info info) const {
        size_t bytes = 0;
        CheckError(clGetDeviceInfo(m_device, info, 0, nullptr, &bytes));
        std::vector<T> result(bytes/sizeof(T));
        CheckError(clGetDeviceInfo(m_device, info, bytes, result.data(), nullptr));
        return result;
    }
};

class clContext final : public raw::Context {
    cl_device_id m_device;
    cl_context m_context;

public:
    explicit clContext(cl_device_id device, cl_context context)
        : m_device(device), m_context(context) {}

    ~clContext() override {
        if (m_context) {
            CheckErrorDtor(clReleaseContext(m_context));
        }
    }

    ContextID id() const noexcept override {
        return reinterpret_cast<ContextID>(m_context);
    }

    std::shared_ptr<raw::Program> compile(const char *source, const std::vector<std::string> &options) override;
    std::shared_ptr<raw::Queue> createQueue() override;
    std::shared_ptr<raw::Event> createEvent() override;
    std::shared_ptr<raw::Buffer> createBuffer(BufferAccess access, size_t size) override;
};

class clQueue final : public raw::Queue {
    cl_command_queue m_queue;

public:
    explicit clQueue(cl_command_queue queue)
        : m_queue(queue) {}

    ~clQueue() override {
        if (m_queue != nullptr) {
            clReleaseCommandQueue(m_queue);
        }
    }

    static cl_command_queue* unwrap(raw::Queue& queue) {
        return &reinterpret_cast<clQueue&>(queue).m_queue;
    }
    static const cl_command_queue* unwrap(const raw::Queue& queue) {
        return &reinterpret_cast<const clQueue&>(queue).m_queue;
    }

    void finish(raw::Event&) override {
        finish();
    }

    void finish() override {
        CheckError(clFinish(m_queue));
    }
};

class clEvent final : public raw::Event {
    cl_event m_event = nullptr;

public:
    ~clEvent() override {
        if (m_event) {
            CheckErrorDtor(clReleaseEvent(m_event));
        }
    }

    static cl_event* unwrap(raw::Event* event) {
        return event== nullptr ? nullptr : &reinterpret_cast<clEvent*>(event)->m_event;
    }
    static const cl_event* unwrap(const raw::Event* event) {
        return event==nullptr ? nullptr : &reinterpret_cast<const clEvent*>(event)->m_event;
    }

    void waitForCompletion() override {
        CheckError(clWaitForEvents(1, &m_event));
    }

    float getElapsedTime() override {
        waitForCompletion();

        cl_ulong time_start = 0, time_end = 0;
        const auto bytes = sizeof(cl_ulong);
        CheckError(clGetEventProfilingInfo(m_event, CL_PROFILING_COMMAND_START, bytes, &time_start, nullptr));
        CheckError(clGetEventProfilingInfo(m_event, CL_PROFILING_COMMAND_END, bytes, &time_end, nullptr));
        return static_cast<float>(time_end - time_start) * 1.0e-6f;
    }
};

class clBuffer final : public raw::Buffer {
    BufferAccess m_access;
    cl_mem m_buffer;

public:
    clBuffer(BufferAccess access, cl_mem buffer)
        : m_access(access), m_buffer(buffer) {}

    ~clBuffer() override {
        if (m_buffer) {
            CheckErrorDtor(clReleaseMemObject(m_buffer));
        }
    }

    static cl_mem* unwrap(raw::Buffer& raw) {
        return &reinterpret_cast<clBuffer&>(raw).m_buffer;
    }
    static const cl_mem* unwrap(const raw::Buffer& raw) {
        return &reinterpret_cast<const clBuffer&>(raw).m_buffer;
    }

    void read(raw::Queue& queue, void* host, size_t size, size_t offset) override {
        if (m_access == BufferAccess::kWriteOnly)
            throw LogicError("Buffer: reading from a write-only buffer");
        CheckError(clEnqueueReadBuffer(
            *clQueue::unwrap(queue), m_buffer, CL_FALSE,
            offset, size, host, 0, nullptr, nullptr));
    }

    void write(raw::Queue& queue, const void* host, size_t size, size_t offset) override {
        if (m_access == BufferAccess::kReadOnly)
            throw LogicError("Buffer: writing to a read-only buffer");
        CheckError(clEnqueueWriteBuffer(
            *clQueue::unwrap(queue), m_buffer, CL_FALSE,
            offset, size, host, 0, nullptr, nullptr));
    }

    void copy(raw::Queue& queue, raw::Buffer& dest, size_t size) override {
        CheckError(clEnqueueCopyBuffer(
            *clQueue::unwrap(queue),
            m_buffer, *clBuffer::unwrap(dest),
            0, 0, size, 0, nullptr, nullptr));
    }
};

class clProgram final : public raw::Program {
    cl_program m_program;

public:
    explicit clProgram(cl_program program)
        : m_program(program) {}

    ~clProgram() override {
        if (m_program) {
            CheckErrorDtor(clReleaseProgram(m_program));
        }
    }

    std::shared_ptr<raw::Kernel> getKernel(const char* name) override;
};

class clKernel final : public raw::Kernel {
    cl_kernel m_kernel;

public:
    explicit clKernel(cl_kernel kernel)
        : m_kernel(kernel) {}

    ~clKernel() override {
        if (m_kernel) {
            clReleaseKernel(m_kernel);
        }
    }

    void setArgument(size_t index, const void* value, size_t size) override {
        CheckError(clSetKernelArg(m_kernel, index, size, value));
    }

    void setArgument(size_t index, const raw::Buffer& buffer) override {
        setArgument(index, clBuffer::unwrap(buffer), sizeof(cl_mem));
    }

    void launch(raw::Queue& queue,
                const std::vector<size_t>& global,
                const std::vector<size_t>& local,
                raw::Event* event) override
    {
        CheckError(clEnqueueNDRangeKernel(
            *clQueue::unwrap(queue), m_kernel,
            global.size(), nullptr, global.data(),
            local.data(), 0, nullptr, clEvent::unwrap(event)));
    }
};

std::shared_ptr<raw::Platform> probe_cl() {
    static std::shared_ptr<raw::Platform> platform;

    if (platform == nullptr) {
        cl_int status;
        cl_platform_id id;
        cl_uint num_platforms;

        status = clGetPlatformIDs(1, &id, &num_platforms);
        if (status != CL_SUCCESS || num_platforms == 0)
            return nullptr;
        platform = std::make_shared<clPlatform>(id);
    }

    return platform;
}

std::vector<std::shared_ptr<raw::Device>> clPlatform::devices(DeviceType type) const {
    cl_device_type cl_type;
    switch (type) {
    case DeviceType::Default:
        cl_type = CL_DEVICE_TYPE_DEFAULT;
        break;
    case DeviceType::CPU:
        cl_type = CL_DEVICE_TYPE_CPU;
        break;
    case DeviceType::GPU:
        cl_type = CL_DEVICE_TYPE_GPU;
        break;
    case DeviceType::Accelerator:
        cl_type = CL_DEVICE_TYPE_ACCELERATOR;
        break;
    default:
        cl_type = CL_DEVICE_TYPE_ALL;
        break;
    }

    cl_uint num_devices = 0;
    CheckError(clGetDeviceIDs(m_platform, cl_type, 0, nullptr, &num_devices));
    std::vector<cl_device_id > device_ids(static_cast<size_t>(num_devices));
    CheckError(clGetDeviceIDs(m_platform, cl_type, num_devices, device_ids.data(), nullptr));

    std::vector<std::shared_ptr<raw::Device>> devices;
    devices.reserve(static_cast<size_t>(num_devices));
    for (auto id : device_ids)
        devices.push_back(std::make_shared<clDevice>(id));
    return devices;
}

std::shared_ptr<raw::Device> clPlatform::device() {
    if (m_default_device == nullptr) {
        cl_uint num_devices = 0;
        cl_device_id device_id;
        CheckError(clGetDeviceIDs(m_platform, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &num_devices));
        if (num_devices == 0)
            throw RuntimeError("OpenCL default device not found");
        m_default_device = std::make_shared<clDevice>(device_id);
    }
    return m_default_device;
}

std::shared_ptr<raw::Context> clDevice::createContext() {
    auto status = CL_SUCCESS;
    auto context = clCreateContext(nullptr, 1, &m_device, nullptr, nullptr, &status);
    check(status, "clCreateContext");
    return std::make_shared<clContext>(m_device, context);
}

std::shared_ptr<raw::Queue> clContext::createQueue() {
    auto status = CL_SUCCESS;
    auto queue = clCreateCommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &status);
    check(status, "clCreateCommandQueue");
    return std::make_shared<clQueue>(queue);
}

std::shared_ptr<raw::Event> clContext::createEvent() {
    return std::make_shared<clEvent>();
}

std::shared_ptr<raw::Buffer> clContext::createBuffer(BufferAccess access, size_t size) {
    auto flags = CL_MEM_READ_WRITE;
    if (access == BufferAccess::kReadOnly)
        flags = CL_MEM_READ_ONLY;
    if (access == BufferAccess::kWriteOnly)
        flags = CL_MEM_WRITE_ONLY;
    auto status = CL_SUCCESS;
    auto buffer = clCreateBuffer(m_context, flags, size, nullptr, &status);
    check(status, "clCreateBuffer");
    return std::make_shared<clBuffer>(access, buffer);
}

static std::string getBuildInfo(cl_program program, cl_device_id device) {
    size_t bytes = 0;
    auto query = cl_program_build_info{CL_PROGRAM_BUILD_LOG};
    CheckError(clGetProgramBuildInfo(program, device, query, 0, nullptr, &bytes));

    std::string result;
    result.resize(bytes);
    CheckError(clGetProgramBuildInfo(program, device, query, bytes, &result[0], nullptr));
    result.resize(strlen(result.c_str())); // remove trailing '\0'
    return result;
}

std::shared_ptr<raw::Program> clContext::compile(const char *source, const std::vector<std::string> &options) {
    auto status = CL_SUCCESS;
    auto program = clCreateProgramWithSource(m_context, 1, &source, nullptr, &status);
    check(status, "clCreateProgramWithSource");

    auto option_string = std::accumulate(options.begin(), options.end(), std::string{" "});
    status = clBuildProgram(program, 1, &m_device, option_string.c_str(), nullptr, nullptr);
    if (status == CL_BUILD_PROGRAM_FAILURE)
        std::cout << getBuildInfo(program, m_device);
    check(status, "clBuildProgram");

    return std::make_shared<clProgram>(program);
}

std::shared_ptr<raw::Kernel> clProgram::getKernel(const char* name) {
    auto status = CL_SUCCESS;
    auto kernel = clCreateKernel(m_program, name, &status);
    check(status, "clCreateKernel");
    return std::make_shared<clKernel>(kernel);
}
