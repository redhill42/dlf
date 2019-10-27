#include <numeric>
#include <iostream>
#include "gpgpu_cl.hpp"

namespace gpgpu { namespace cl {

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

std::shared_ptr<rawPlatform> probe() {
    static std::shared_ptr<rawPlatform> platform;

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

std::vector<std::shared_ptr<rawDevice>> clPlatform::devices(DeviceType type) const {
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

    std::vector<std::shared_ptr<rawDevice>> devices;
    auto filter = parseDeviceFilter(num_devices);
    for (size_t i = 0; i < device_ids.size(); i++) {
        if (filter[i]) {
            auto id = device_ids[i];
            devices.push_back(std::make_shared<clDevice>(id));
        }
    }
    return devices;
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

std::shared_ptr<rawContext> clDevice::createContext() const {
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

std::string clDevice::getInfoStr(cl_device_info info) const {
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

std::shared_ptr<rawQueue> clContext::createQueue() const {
    auto status = CL_SUCCESS;
    auto queue = clCreateCommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &status);
    check(status, "clCreateCommandQueue");
    return std::make_shared<clQueue>(queue);
}

std::shared_ptr<rawEvent> clContext::createEvent() const {
    return std::make_shared<clEvent>();
}

std::shared_ptr<rawBuffer> clContext::createBuffer(size_t size, BufferAccess access) const {
    auto flags = CL_MEM_READ_WRITE;
    if (access == BufferAccess::ReadOnly)
        flags = CL_MEM_READ_ONLY;
    if (access == BufferAccess::WriteOnly)
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

std::shared_ptr<rawProgram> clContext::compileProgram(
    const char* source, const std::vector<std::string>& options) const
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

std::shared_ptr<rawProgram> clContext::loadProgram(const std::string& binary) const {
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

void clQueue::finish(rawEvent&) const {
    finish();
}

void clQueue::finish() const {
    CheckError(clFinish(m_queue));
}

clQueue::~clQueue() {
    if (m_queue)
        CheckErrorDtor(clReleaseCommandQueue(m_queue));
}

void clEvent::waitForCompletion() const {
    CheckError(clWaitForEvents(1, &m_event));
}

float clEvent::getElapsedTime() const {
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

void clBuffer::read(const rawQueue& queue, void* host, size_t size, size_t offset, rawEvent* event) const {
    if (m_access == BufferAccess::WriteOnly)
        throw LogicError("Buffer: reading from a write-only buffer");
    CheckError(clEnqueueReadBuffer(
        *clQueue::unwrap(queue), m_buffer, CL_FALSE,
        offset, size, host, 0, nullptr, clEvent::unwrap(event)));
}

void clBuffer::write(const rawQueue& queue, const void* host, size_t size, size_t offset, rawEvent* event) {
    if (m_access == BufferAccess::ReadOnly)
        throw LogicError("Buffer: writing to a read-only buffer");
    CheckError(clEnqueueWriteBuffer(
        *clQueue::unwrap(queue), m_buffer, CL_FALSE,
        offset, size, host, 0, nullptr, clEvent::unwrap(event)));
}

void clBuffer::copyTo(const rawQueue& queue, gpgpu::rawBuffer& dest, size_t size,
                      size_t src_offset, size_t dst_offset, rawEvent* event) const {
    CheckError(clEnqueueCopyBuffer(
        *clQueue::unwrap(queue),
        m_buffer, *clBuffer::unwrap(dest),
        src_offset, dst_offset, size,
        0, nullptr, clEvent::unwrap(event)));
}

clBuffer::~clBuffer() {
    if (m_buffer)
        CheckErrorDtor(clReleaseMemObject(m_buffer));
}

std::string clProgram::getIR() const {
    size_t bytes = 0;
    CheckError(clGetProgramInfo(m_program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bytes, nullptr));
    std::string binary;
    binary.resize(bytes);
    auto ptr = binary.data();
    CheckError(clGetProgramInfo(m_program, CL_PROGRAM_BINARIES, sizeof(char*), &ptr, nullptr));
    return binary;
}

std::shared_ptr<rawKernel> clProgram::getKernel(const char* name) const {
    auto status = CL_SUCCESS;
    auto kernel = clCreateKernel(m_program, name, &status);
    check(status, "clCreateKernel");
    return std::make_shared<clKernel>(kernel);
}

clProgram::~clProgram() {
    if (m_program)
        CheckErrorDtor(clReleaseProgram(m_program));
}

uint64_t clKernel::localMemoryUsage(const rawDevice& device) const {
    auto device_id = reinterpret_cast<cl_device_id>(device.id());
    cl_ulong result = 0;
    CheckError(clGetKernelWorkGroupInfo(
        m_kernel, device_id,
        CL_KERNEL_LOCAL_MEM_SIZE,
        sizeof(result), &result, nullptr));
    return static_cast<uint64_t>(result);
}

void clKernel::setArgument(size_t index, const void* value, size_t size) const {
    CheckError(clSetKernelArg(m_kernel, index, size, value));
}

void clKernel::setArgument(size_t index, const rawBuffer& buffer) const {
    setArgument(index, clBuffer::unwrap(buffer), sizeof(cl_mem));
}

void clKernel::setLocalMemorySize(size_t size) const {
    cl_uint num_args;
    CheckError(clGetKernelInfo(
        m_kernel, CL_KERNEL_NUM_ARGS,
        sizeof(num_args), &num_args, nullptr));
    CheckError(clSetKernelArg(m_kernel, num_args-1, size, nullptr));
}

void clKernel::launch(const rawQueue& queue,
                      const std::vector<size_t>& global,
                      const std::vector<size_t>& local,
                      rawEvent* event) const
{
    CheckError(clEnqueueNDRangeKernel(
        *clQueue::unwrap(queue), m_kernel,
        global.size(), nullptr, global.data(), local.data(),
        0, nullptr, clEvent::unwrap(event)));
}

clKernel::~clKernel() {
    if (m_kernel)
        CheckErrorDtor(clReleaseKernel(m_kernel));
}

}} // namespace gpgpu::cl
