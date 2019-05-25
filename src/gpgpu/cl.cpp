#include <numeric>
#include <iostream>
#include "cl.hpp"

namespace gpgpu::cl {

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

std::shared_ptr<raw::Platform> probe() {
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

std::string clPlatform::getInfo(cl_device_info info) const {
    size_t bytes = 0;
    CheckError(clGetPlatformInfo(m_platform, info, 0, nullptr, &bytes));
    std::string result;
    result.resize(bytes);
    CheckError(clGetPlatformInfo(m_platform, info, bytes, &result[0], nullptr));
    result.resize(strlen(result.c_str())); // remove trailing '\0'
    return result;
}

std::shared_ptr<raw::Context> clDevice::createContext() {
    auto status = CL_SUCCESS;
    auto context = clCreateContext(nullptr, 1, &m_device, nullptr, nullptr, &status);
    check(status, "clCreateContext");
    return std::make_shared<clContext>(m_device, context);
}

template <typename T>
T clDevice::getInfo(cl_device_info info) const {
    T result{};
    CheckError(clGetDeviceInfo(m_device, info, sizeof(T), &result, nullptr));
    return result;
}

template <>
std::string clDevice::getInfo(cl_device_info info) const {
    size_t bytes = 0;
    CheckError(clGetDeviceInfo(m_device, info, 0, nullptr, &bytes));
    std::string result;
    result.resize(bytes);
    CheckError(clGetDeviceInfo(m_device, info, bytes, &result[0], nullptr));
    result.resize(strlen(result.c_str())); // removes any trailing '\0'
    return result;
}

template <typename T>
std::vector<T> clDevice::getInfoVec(cl_device_info info) const {
    size_t bytes = 0;
    CheckError(clGetDeviceInfo(m_device, info, 0, nullptr, &bytes));
    std::vector<T> result(bytes/sizeof(T));
    CheckError(clGetDeviceInfo(m_device, info, bytes, result.data(), nullptr));
    return result;
}

clDevice::~clDevice() {
    if (m_device)
        CheckErrorDtor(clReleaseDevice(m_device));
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

std::shared_ptr<raw::Program> clContext::compileProgram(
    const char *source, const std::vector<std::string> &options)
{
    auto status = CL_SUCCESS;
    auto program = clCreateProgramWithSource(m_context, 1, &source, nullptr, &status);
    check(status, "clCreateProgramWithSource");

    auto option_string = std::accumulate(options.begin(), options.end(), std::string{" "});
    status = clBuildProgram(program, 1, &m_device, option_string.c_str(), nullptr, nullptr);
    if (status == CL_BUILD_PROGRAM_FAILURE)
        std::cerr << getBuildInfo(program, m_device);
    if (status != CL_SUCCESS)
        clReleaseProgram(program);
    check(status, "clBuildProgram");

    return std::make_shared<clProgram>(program);
}

std::shared_ptr<raw::Program> clContext::loadProgram(const std::string &binary) {
    auto binary_ptr = reinterpret_cast<const unsigned char*>(binary.data());
    auto length = binary.length();
    auto status1 = CL_SUCCESS;
    auto status2 = CL_SUCCESS;

    auto program = clCreateProgramWithBinary(
        m_context, 1, &m_device, &length, &binary_ptr, &status1, &status2);
    check(status1, "clCreateProgramWithBinary (binary status)");
    check(status2, "clCreateProgramWithBinary");

    // we need to build the program even if it's loaded from binary
    status1 = clBuildProgram(program, 1, &m_device, nullptr, nullptr, nullptr);
    if (status1 == CL_BUILD_PROGRAM_FAILURE)
        std::cerr << getBuildInfo(program, m_device);
    if (status1 != CL_SUCCESS)
        clReleaseProgram(program);
    check(status1, "clBuildProgram");

    return std::make_shared<clProgram>(program);
}

clContext::~clContext() {
    if (m_context)
        CheckErrorDtor(clReleaseContext(m_context));
}

void clQueue::finish(raw::Event&) {
    finish();
}

void clQueue::finish() {
    CheckError(clFinish(m_queue));
}

clQueue::~clQueue() {
    if (m_queue)
        CheckErrorDtor(clReleaseCommandQueue(m_queue));
}

void clEvent::waitForCompletion() {
    CheckError(clWaitForEvents(1, &m_event));
}

float clEvent::getElapsedTime() {
    waitForCompletion();

    cl_ulong time_start = 0, time_end = 0;
    const auto bytes = sizeof(cl_ulong);
    CheckError(clGetEventProfilingInfo(m_event, CL_PROFILING_COMMAND_START, bytes, &time_start, nullptr));
    CheckError(clGetEventProfilingInfo(m_event, CL_PROFILING_COMMAND_END, bytes, &time_end, nullptr));
    return static_cast<float>(time_end - time_start) * 1.0e-6f;
}

clEvent::~clEvent() {
    if (m_event)
        CheckErrorDtor(clReleaseEvent(m_event));
}

void clBuffer::read(raw::Queue &queue, void *host, size_t size, size_t offset) {
    if (m_access == BufferAccess::kWriteOnly)
        throw LogicError("Buffer: reading from a write-only buffer");
    CheckError(clEnqueueReadBuffer(
        *clQueue::unwrap(queue), m_buffer, CL_FALSE,
        offset, size, host, 0, nullptr, nullptr));
}

void clBuffer::write(raw::Queue &queue, const void *host, size_t size, size_t offset) {
    if (m_access == BufferAccess::kReadOnly)
        throw LogicError("Buffer: writing to a read-only buffer");
    CheckError(clEnqueueWriteBuffer(
        *clQueue::unwrap(queue), m_buffer, CL_FALSE,
        offset, size, host, 0, nullptr, nullptr));
}

void clBuffer::copy(raw::Queue &queue, gpgpu::raw::Buffer &dest, size_t size) {
    CheckError(clEnqueueCopyBuffer(
        *clQueue::unwrap(queue),
        m_buffer, *clBuffer::unwrap(dest),
        0, 0, size, 0, nullptr, nullptr));
}

clBuffer::~clBuffer() {
    if (m_buffer)
        CheckErrorDtor(clReleaseMemObject(m_buffer));
}

std::string clProgram::getIR() {
    size_t bytes = 0;
    CheckError(clGetProgramInfo(m_program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bytes, nullptr));
    std::string binary;
    binary.resize(bytes);
    auto ptr = binary.data();
    CheckError(clGetProgramInfo(m_program, CL_PROGRAM_BINARIES, sizeof(char*), &ptr, nullptr));
    return binary;
}

std::shared_ptr<raw::Kernel> clProgram::getKernel(const char* name) {
    auto status = CL_SUCCESS;
    auto kernel = clCreateKernel(m_program, name, &status);
    check(status, "clCreateKernel");
    return std::make_shared<clKernel>(kernel);
}

clProgram::~clProgram() {
    if (m_program)
        CheckErrorDtor(clReleaseProgram(m_program));
}

void clKernel::setArgument(size_t index, const void *value, size_t size) {
    CheckError(clSetKernelArg(m_kernel, index, size, value));
}

void clKernel::setArgument(size_t index, const raw::Buffer &buffer) {
    setArgument(index, clBuffer::unwrap(buffer), sizeof(cl_mem));
}

void clKernel::launch(raw::Queue &queue,
                      const std::vector<size_t> &global,
                      const std::vector<size_t> &local,
                      raw::Event *event)
{
    CheckError(clEnqueueNDRangeKernel(
        *clQueue::unwrap(queue), m_kernel,
        global.size(), nullptr, global.data(),
        local.data(), 0, nullptr, clEvent::unwrap(event)));
}

clKernel::~clKernel() {
    if (m_kernel)
        CheckErrorDtor(clReleaseKernel(m_kernel));
}

} // namespace gpgpu::cl
