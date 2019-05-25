#include <iostream>
#include <cuda.h>
#include <nvrtc.h>
#include "gpgpu.h"

static constexpr size_t kStringLength = 256;

using namespace gpgpu;

static void check(CUresult status, const std::string& where) {
    if (status != CUDA_SUCCESS) {
        throw APIError(static_cast<int>(status), where,
            "CUDA error: " + where + ": " + std::to_string(status));
    }
}

static void checkDtor(CUresult status, const std::string& where) {
    if (status != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA error: %s: %d (ignoring)\n", where.c_str(), status);
    }
}

static void checkNVRTC(nvrtcResult status, const std::string& where) {
    if (status != NVRTC_SUCCESS) {
        const char* status_string = nvrtcGetErrorString(status);
        throw BuildError(status, where,
            "CUDA NVRTC error: " + where + ": " + status_string);
    }
}

static void checkNVRTCDtor(nvrtcResult status, const std::string& where) {
    if (status != NVRTC_SUCCESS) {
        const char* status_string = nvrtcGetErrorString(status);
        fprintf(stderr, "CUDA NVRTC error: %s: %s (ignoring)\n", where.c_str(), status_string);
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
#define CheckNVRTC(call) checkNVRTC(call, trimCallString(#call))

#define CheckErrorDtor(call) checkDtor(call, trimCallString(#call))
#define CheckNVRTCDtor(call) checkNVRTCDtor(call, trimCallString(#call))

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
        return "NVIDIA Corporation";
    }

    std::string version() const override {
        auto result = 0;
        CheckError(cuDriverGetVersion(&result));
        return "CUDA driver " + std::to_string(result);
    }

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

    std::string version() const override {
        auto result = 0;
        CheckError(cuDriverGetVersion(&result));
        return "CUDA driver " + std::to_string(result);
    }

    std::string vendor() const override {
        return "NVIDIA Corporation";
    }

    std::string name() const override {
        std::string result;
        result.resize(kStringLength);
        CheckError(cuDeviceGetName(&result[0], result.size(), m_device));
        return result;
    }

    DeviceType type() const override {
        return DeviceType::GPU;
    }

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

    uint64_t memorySize() const override {
        size_t result = 0;
        CheckError(cuDeviceTotalMem(&result, m_device));
        return result;
    }

    uint64_t maxAllocSize() const override {
        return memorySize();
    }

    std::shared_ptr<raw::Context> createContext() override;

private:
    size_t getInfo(CUdevice_attribute info) const {
        int result = 0;
        CheckError(cuDeviceGetAttribute(&result, info, m_device));
        return result;
    }
};

class cuContext final : public raw::Context {
    CUcontext m_context;

public:
    explicit cuContext(CUcontext context)
        : m_context(context) {}

    ~cuContext() override {
        if (m_context) {
            CheckErrorDtor(cuCtxDestroy(m_context));
        }
    }

    ContextID id() const noexcept override {
        return reinterpret_cast<ContextID>(m_context);
    }

    std::shared_ptr<raw::Program> compileProgram(const char *source, const std::vector<std::string> &options) override;
    std::shared_ptr<raw::Program> loadProgram(const std::string& binary) override ;

    std::shared_ptr<raw::Queue> createQueue() override;
    std::shared_ptr<raw::Event> createEvent() override;
    std::shared_ptr<raw::Buffer> createBuffer(BufferAccess access, size_t size) override;
};

class cuEvent final : public raw::Event {
    CUevent m_start, m_end;

public:
    cuEvent(CUevent start, CUevent end)
        : m_start(start), m_end(end) {}

    ~cuEvent() override {
        if (m_start)
            CheckErrorDtor(cuEventDestroy(m_start));
        if (m_end)
            CheckErrorDtor(cuEventDestroy(m_end));
    }

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
public:
    explicit cuQueue(CUstream queue)
        : m_queue(queue) {}

    ~cuQueue() override {
        if (m_queue)
            CheckErrorDtor(cuStreamDestroy(m_queue));
    }

    static CUstream* unwrap(raw::Queue& queue) {
        return &reinterpret_cast<cuQueue&>(queue).m_queue;
    }
    static const CUstream* unwrap(const raw::Queue& queue) {
        return &reinterpret_cast<const cuQueue&>(queue).m_queue;
    }

    void finish(raw::Event& event) override {
        CheckError(cuEventSynchronize(cuEvent::end(event)));
        finish();
    }

    void finish() override {
        CheckError(cuStreamSynchronize(m_queue));
    }
};

class cuBuffer final : public raw::Buffer {
    BufferAccess m_access;
    CUdeviceptr m_buffer;

public:
    explicit cuBuffer(BufferAccess access, CUdeviceptr buffer)
        : m_access(access), m_buffer(buffer) {}

    ~cuBuffer() override {
        if (m_buffer)
            CheckErrorDtor(cuMemFree(m_buffer));
    }

    static CUdeviceptr* unwrap(raw::Buffer& buffer) {
        return &reinterpret_cast<cuBuffer&>(buffer).m_buffer;
    }
    static const CUdeviceptr* unwrap(const raw::Buffer& buffer) {
        return &reinterpret_cast<const cuBuffer&>(buffer).m_buffer;
    }

    void read(raw::Queue& queue, void* host, size_t size, size_t offset) override {
        if (m_access == BufferAccess::kWriteOnly)
            throw LogicError("Buffer: reading from a write-only buffer");
        CheckError(cuMemcpyDtoHAsync(host, m_buffer+offset, size, *cuQueue::unwrap(queue)));
    }

    void write(raw::Queue& queue, const void* host, size_t size, size_t offset) override {
        if (m_access == BufferAccess::kReadOnly)
            throw LogicError("Buffer: writing to a read-only buffer");
        CheckError(cuMemcpyHtoDAsync(m_buffer+offset, host, size, *cuQueue::unwrap(queue)));
    }

    void copy(raw::Queue& queue, raw::Buffer& dest, size_t size) override {
        CheckError(cuMemcpyDtoDAsync(*cuBuffer::unwrap(dest), m_buffer, size, *cuQueue::unwrap(queue)));
    }
};

class cuProgram final : public raw::Program {
    CUmodule m_module;
    std::string m_ir;

public:
    explicit cuProgram(CUmodule module, std::string ir)
        : m_module(module), m_ir(std::move(ir)) {}

    ~cuProgram() override {
        if (m_module) {
            cuModuleUnload(m_module);
        }
    }

    std::string getIR() override {
        return m_ir;
    }

    std::shared_ptr<raw::Kernel> getKernel(const char* name) override;
};

class cuKernel final : public raw::Kernel {
    CUfunction m_kernel;
    std::vector<size_t> m_arguments_indices;
    std::vector<char> m_arguments_data;

public:
    explicit cuKernel(CUfunction kernel) : m_kernel(kernel) {}

    /**
     * Sets a kernel argument at the indicated position. This stores both the
     * value of the argument (as raw bytes) and the index indicating where
     * this value can be found.
     */
    void setArgument(size_t index, const void* value, size_t size) override {
        if (index >= m_arguments_indices.size())
            m_arguments_indices.resize(index+1);
        m_arguments_indices[index] = m_arguments_data.size();

        char* end = &m_arguments_data[m_arguments_data.size()];
        m_arguments_data.resize(m_arguments_data.size() + size);
        memcpy(end, value, size);
    }

    void setArgument(size_t index, const raw::Buffer& buffer) override {
        setArgument(index, cuBuffer::unwrap(buffer), sizeof(CUdeviceptr));
    }

    void launch(raw::Queue& queue,
                const std::vector<size_t>& global,
                const std::vector<size_t>& local,
                raw::Event* event) override;
};

std::shared_ptr<raw::Platform> probe_cu() {
    static std::shared_ptr<raw::Platform> platform;

    if (platform == nullptr) {
        if (cuInit(0) != CUDA_SUCCESS)
            return nullptr;

        int num_devices = 0;
        auto status = cuDeviceGetCount(&num_devices);
        if (status != CUDA_SUCCESS || num_devices == 0)
            return nullptr;

        platform = std::make_shared<cuPlatform>();
    }

    return platform;
}

std::vector<std::shared_ptr<raw::Device>> cuPlatform::devices(DeviceType type) const {
    if (type != DeviceType::GPU && type != DeviceType::All)
        return {};

    int num_devices = 0;
    CheckError(cuDeviceGetCount(&num_devices));

    std::vector<std::shared_ptr<raw::Device>> devices;
    devices.reserve(num_devices);
    for (int id = 0; id < num_devices; id++) {
        CUdevice device;
        CheckError(cuDeviceGet(&device, id));
        devices.push_back(std::make_shared<cuDevice>(device));
    }

    return devices;
}

std::shared_ptr<raw::Device> cuPlatform::device() {
    CUdevice device;
    CheckError(cuDeviceGet(&device, 0));
    return std::make_shared<cuDevice>(device);
}

std::shared_ptr<raw::Context> cuDevice::createContext() {
    CUcontext context = nullptr;
    CheckError(cuCtxCreate(&context, 0, m_device));
    return std::make_shared<cuContext>(context);
}

std::shared_ptr<raw::Queue> cuContext::createQueue() {
    CUstream queue = nullptr;
    CheckError(cuStreamCreate(&queue, CU_STREAM_NON_BLOCKING));
    return std::make_shared<cuQueue>(queue);
}

std::shared_ptr<raw::Event> cuContext::createEvent() {
    CUevent start = nullptr, end = nullptr;
    CheckError(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CheckError(cuEventCreate(&end, CU_EVENT_DEFAULT));
    return std::make_shared<cuEvent>(start, end);
}

std::shared_ptr<raw::Buffer> cuContext::createBuffer(BufferAccess access, size_t size) {
    CUdeviceptr buffer = 0;
    CheckError(cuMemAlloc(&buffer, size));
    return std::make_shared<cuBuffer>(access, buffer);
}

static std::string getBuildInfo(nvrtcProgram program) {
    size_t bytes = 0;
    CheckNVRTC(nvrtcGetProgramLogSize(program, &bytes));

    std::string log;
    log.resize(bytes);
    CheckNVRTC(nvrtcGetProgramLog(program, log.data()));
    return log;
}

std::shared_ptr<raw::Program> cuContext::compileProgram(
    const char *source, const std::vector<std::string> &options)
{
    nvrtcProgram program = nullptr;
    CheckNVRTC(nvrtcCreateProgram(&program, source, nullptr, 0, nullptr, nullptr));

    std::vector<const char*> raw_options;
    raw_options.reserve(options.size());
    for (const auto& option : options)
        raw_options.push_back(option.c_str());

    auto status = nvrtcCompileProgram(program, raw_options.size(), raw_options.data());
    if (status == NVRTC_ERROR_INVALID_INPUT)
        std::cerr << getBuildInfo(program) << std::endl;
    if (status != NVRTC_SUCCESS)
        nvrtcDestroyProgram(&program);
    checkNVRTC(status, "nvrtcCompileProgram");

    size_t bytes = 0;
    CheckNVRTC(nvrtcGetPTXSize(program, &bytes));

    std::string ir;
    ir.resize(bytes);
    CheckNVRTC(nvrtcGetPTX(program, ir.data()));

    CheckNVRTC(nvrtcDestroyProgram(&program));
    return loadProgram(ir);
}

std::shared_ptr<raw::Program> cuContext::loadProgram(const std::string &ir) {
    CUmodule module;
    CheckError(cuModuleLoadDataEx(&module, ir.data(), 0, nullptr, nullptr));
    return std::make_shared<cuProgram>(module, ir);
}

std::shared_ptr<raw::Kernel> cuProgram::getKernel(const char *name) {
    CUfunction kernel = nullptr;
    CheckError(cuModuleGetFunction(&kernel, m_module, name));
    return std::make_shared<cuKernel>(kernel);
}

void cuKernel::launch(raw::Queue& queue,
                      const std::vector<size_t>& global,
                      const std::vector<size_t>& local,
                      raw::Event* event)
{
    size_t grid[]{1, 1, 1};
    size_t block[]{1, 1, 1};

    if (global.size() != local.size())
        throw LogicError("invalid thread/workgroup dimensions");

    for (size_t i = 0; i < local.size(); i++) {
        grid[i] = global[i] / local[i];
        block[i] = local[i];
    }

    // Creates the array of pointers from the arrays of indices & data
    std::vector<void*> pointers;
    for (auto& index : m_arguments_indices) {
        pointers.push_back(&m_arguments_data[index]);
    }

    auto q = *cuQueue::unwrap(queue);
    if (event != nullptr)
        CheckError(cuEventRecord(cuEvent::start(*event), q));
    CheckError(cuLaunchKernel(
        m_kernel,
        grid[0], grid[1], grid[2],
        block[0], block[1], block[2],
        0, q, pointers.data(), nullptr));
    if (event != nullptr)
        CheckError(cuEventRecord(cuEvent::end(*event), q));
}
