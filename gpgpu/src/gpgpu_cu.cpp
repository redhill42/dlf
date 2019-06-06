#include <iostream>
#include "gpgpu_cu.hpp"

namespace gpgpu { namespace cu {

static void check(CUresult status, const std::string& where) {
    if (status != CUDA_SUCCESS) {
        const char* error_name;
        const char* error_string;
        cuGetErrorName(status, &error_name);
        cuGetErrorString(status, &error_string);
        throw APIError(static_cast<int>(status), where,
            "CUDA error: " + where + ": " +
            error_name + " - " + error_string);
    }
}

static void checkDtor(CUresult status, const std::string& where) {
    if (status != CUDA_SUCCESS && status != CUDA_ERROR_DEINITIALIZED) {
        const char* error_name;
        const char* error_string;
        cuGetErrorName(status, &error_name);
        cuGetErrorString(status, &error_string);
        fprintf(stderr, "CUDA error: %s: %s - %s (ignoring)\n",
            where.c_str(), error_name, error_string);
    }
}

static void checkNVRTC(nvrtcResult status, const std::string& where) {
    if (status != NVRTC_SUCCESS) {
        const char* error_string = nvrtcGetErrorString(status);
        throw BuildError(status, where,
            "CUDA NVRTC error: " + where + ": " + error_string);
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
#define CheckNVRTC(call) checkNVRTC(call, trimCallString(#call))

std::shared_ptr<raw::Platform> probe() {
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

std::shared_ptr<raw::Device> cuPlatform::device() const {
    CUdevice device;
    CheckError(cuDeviceGet(&device, 0));
    return std::make_shared<cuDevice>(device);
}

std::string cuPlatform::version() const {
    auto result = 0;
    CheckError(cuDriverGetVersion(&result));
    return "CUDA driver " + std::to_string(result);
}

std::string cuDevice::version() const {
    auto result = 0;
    CheckError(cuDriverGetVersion(&result));
    return "CUDA driver " + std::to_string(result);
}

std::string cuDevice::vendor() const {
    return "NVIDIA";
}

std::string cuDevice::name() const {
    std::string result;
    result.resize(256);
    CheckError(cuDeviceGetName(&result[0], result.size(), m_device));
    return result;
}

DeviceType cuDevice::type() const {
    return DeviceType::GPU;
}

uint64_t cuDevice::memorySize() const {
    size_t result = 0;
    CheckError(cuDeviceTotalMem(&result, m_device));
    return result;
}

size_t cuDevice::getInfo(CUdevice_attribute info) const {
    int result = 0;
    CheckError(cuDeviceGetAttribute(&result, info, m_device));
    return result;
}

std::shared_ptr<raw::Context> cuDevice::createContext() const {
    CUcontext context = nullptr;
    CheckError(cuCtxCreate(&context, 0, m_device));
    return std::make_shared<cuContext>(context);
}

std::shared_ptr<raw::Queue> cuContext::createQueue() const {
    CUstream queue = nullptr;
    CheckError(cuStreamCreate(&queue, CU_STREAM_NON_BLOCKING));
    return std::make_shared<cuQueue>(queue);
}

std::shared_ptr<raw::Event> cuContext::createEvent() const {
    CUevent start = nullptr, end = nullptr;
    CheckError(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CheckError(cuEventCreate(&end, CU_EVENT_DEFAULT));
    return std::make_shared<cuEvent>(start, end);
}

std::shared_ptr<raw::Buffer> cuContext::createBuffer(size_t size, BufferAccess access) const {
    CUdeviceptr buffer = 0;
    CheckError(cuMemAlloc(&buffer, size));
    return std::make_shared<cuBuffer>(access, buffer);
}

static std::string getBuildInfo(nvrtcProgram program) {
    size_t bytes = 0;
    CheckNVRTC(nvrtcGetProgramLogSize(program, &bytes));

    std::string log;
    log.resize(bytes);
    CheckNVRTC(nvrtcGetProgramLog(program, &log[0]));
    return log;
}

std::shared_ptr<raw::Program> cuContext::compileProgram(
    const char* source, const std::vector<std::string>& options) const
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
    CheckNVRTC(nvrtcGetPTX(program, &ir[0]));

    CheckNVRTC(nvrtcDestroyProgram(&program));
    return loadProgram(ir);
}

std::shared_ptr<raw::Program> cuContext::loadProgram(const std::string& ir) const {
    CUmodule module;
    CheckError(cuModuleLoadDataEx(&module, ir.data(), 0, nullptr, nullptr));
    return std::make_shared<cuProgram>(module, ir);
}

cuContext::~cuContext() {
    if (m_context)
        CheckErrorDtor(cuCtxDestroy(m_context));
}

cuEvent::~cuEvent() {
    if (m_start)
        CheckErrorDtor(cuEventDestroy(m_start));
    if (m_end)
        CheckErrorDtor(cuEventDestroy(m_end));
}

cublasHandle_t cuQueue::getCublasHandle() const {
    if (m_cublas == nullptr) {
        // TODO: check error
        cublasCreate(&m_cublas);
        cublasSetStream(m_cublas, m_queue);
    }
    return m_cublas;
}

void cuQueue::finish(raw::Event& event) const {
    CheckError(cuEventSynchronize(cuEvent::end(event)));
    finish();
}

void cuQueue::finish() const {
    CheckError(cuStreamSynchronize(m_queue));
}

cuQueue::~cuQueue() {
    if (m_queue)
        CheckErrorDtor(cuStreamDestroy(m_queue));
    if (m_cublas)
        cublasDestroy(m_cublas);
}

void cuBuffer::read(const raw::Queue& queue, void* host, size_t size, size_t offset, raw::Event*) const {
    if (m_access == BufferAccess::kWriteOnly)
        throw LogicError("Buffer: reading from a write-only buffer");
    CheckError(cuMemcpyDtoHAsync(host, m_buffer+offset, size, *cuQueue::unwrap(queue)));
}

void cuBuffer::write(const raw::Queue& queue, const void* host, size_t size, size_t offset, raw::Event*) {
    if (m_access == BufferAccess::kReadOnly)
        throw LogicError("Buffer: writing to a read-only buffer");
    CheckError(cuMemcpyHtoDAsync(m_buffer+offset, host, size, *cuQueue::unwrap(queue)));
}

void cuBuffer::copyTo(const raw::Queue& queue, raw::Buffer& dest, size_t size, raw::Event*) const {
    CheckError(cuMemcpyDtoDAsync(*cuBuffer::unwrap(dest), m_buffer, size, *cuQueue::unwrap(queue)));
}

cuBuffer::~cuBuffer() {
    if (m_buffer)
        CheckErrorDtor(cuMemFree(m_buffer));
}

std::shared_ptr<raw::Kernel> cuProgram::getKernel(const char *name) const {
    CUfunction kernel = nullptr;
    CheckError(cuModuleGetFunction(&kernel, m_module, name));
    return std::make_shared<cuKernel>(kernel);
}

cuProgram::~cuProgram() {
    if (m_module)
        CheckErrorDtor(cuModuleUnload(m_module));
}

uint64_t cuKernel::localMemoryUsage(const raw::Device&) const {
    auto result = 0;
    CheckError(cuFuncGetAttribute(&result, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, m_kernel));
    return static_cast<uint64_t>(result);
}

/**
 * Sets a kernel argument at the indicated position. This stores both the
 * value of the argument (as raw bytes) and the index indicating where
 * this value can be found.
 */
void cuKernel::setArgument(size_t index, const void* value, size_t size) const {
    if (index >= m_arguments_indices.size())
        m_arguments_indices.resize(index+1);
    m_arguments_indices[index] = m_arguments_data.size();

    size_t end = m_arguments_data.size();
    m_arguments_data.resize(end + size);
    memcpy(&m_arguments_data[end], value, size);
}

void cuKernel::setArgument(size_t index, const raw::Buffer& buffer) const {
    setArgument(index, cuBuffer::unwrap(buffer), sizeof(CUdeviceptr));
}

void cuKernel::launch(const raw::Queue& queue,
                      const std::vector<size_t>& global,
                      const std::vector<size_t>& local,
                      raw::Event* event) const
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

}} // namespace gpgpu::cu
