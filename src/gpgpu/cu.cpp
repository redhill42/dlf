#include <iostream>
#include "cu.hpp"

namespace gpgpu::cu {

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

std::shared_ptr<raw::Device> cuPlatform::device() {
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
    const char* source, const std::vector<std::string>& options)
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

std::shared_ptr<raw::Program> cuContext::loadProgram(const std::string& ir) {
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

cublasHandle_t cuQueue::getCublasHandle() {
    if (m_cublas == nullptr) {
        // TODO: check error
        cublasCreate(&m_cublas);
        cublasSetStream_v2(m_cublas, m_queue);
    }
    return m_cublas;
}

void cuQueue::finish(raw::Event& event) {
    CheckError(cuEventSynchronize(cuEvent::end(event)));
    finish();
}

void cuQueue::finish() {
    CheckError(cuStreamSynchronize(m_queue));
}

cuQueue::~cuQueue() {
    if (m_queue)
        CheckErrorDtor(cuStreamDestroy(m_queue));
    if (m_cublas)
        cublasDestroy(m_cublas);
}

void cuBuffer::read(raw::Queue& queue, void* host, size_t size, size_t offset) {
    if (m_access == BufferAccess::kWriteOnly)
        throw LogicError("Buffer: reading from a write-only buffer");
    CheckError(cuMemcpyDtoHAsync(host, m_buffer+offset, size, *cuQueue::unwrap(queue)));
}

void cuBuffer::write(raw::Queue& queue, const void* host, size_t size, size_t offset) {
    if (m_access == BufferAccess::kReadOnly)
        throw LogicError("Buffer: writing to a read-only buffer");
    CheckError(cuMemcpyHtoDAsync(m_buffer+offset, host, size, *cuQueue::unwrap(queue)));
}

void cuBuffer::copy(raw::Queue& queue, raw::Buffer& dest, size_t size) {
    CheckError(cuMemcpyDtoDAsync(*cuBuffer::unwrap(dest), m_buffer, size, *cuQueue::unwrap(queue)));
}

cuBuffer::~cuBuffer() {
    if (m_buffer)
        CheckErrorDtor(cuMemFree(m_buffer));
}

std::shared_ptr<raw::Kernel> cuProgram::getKernel(const char *name) {
    CUfunction kernel = nullptr;
    CheckError(cuModuleGetFunction(&kernel, m_module, name));
    return std::make_shared<cuKernel>(kernel);
}

cuProgram::~cuProgram() {
    if (m_module)
        CheckErrorDtor(cuModuleUnload(m_module));
}

/**
 * Sets a kernel argument at the indicated position. This stores both the
 * value of the argument (as raw bytes) and the index indicating where
 * this value can be found.
 */
void cuKernel::setArgument(size_t index, const void* value, size_t size) {
    if (index >= m_arguments_indices.size())
        m_arguments_indices.resize(index+1);
    m_arguments_indices[index] = m_arguments_data.size();

    char* end = &m_arguments_data[m_arguments_data.size()];
    m_arguments_data.resize(m_arguments_data.size() + size);
    memcpy(end, value, size);
}

void cuKernel::setArgument(size_t index, const raw::Buffer& buffer) {
    setArgument(index, cuBuffer::unwrap(buffer), sizeof(CUdeviceptr));
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

} // namespace gpgpu::cu
