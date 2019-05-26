#ifndef _GPGPU_H
#define _GPGPU_H

#include <string>
#include <cstring>
#include <vector>
#include <memory>
#include <stdexcept>
#include "os_blas.h"

namespace gpgpu {

//==-------------------------------------------------------------------------
// Exceptions
//==-------------------------------------------------------------------------

/**
 * Basic exception class: represents an error happened inside our
 * code (as opposed to an error in C++ runtime)
 */
template <typename Base>
class Error : public Base {
public:
    template <typename... Args>
    explicit Error(Args&&... args)
        : Base(std::forward<Args>(args)...)
    {}
};

/**
 * Represents a generic device-specific runtime error (returned by
 * an OpenCL or CUDA API function)
 */
class DeviceError : public Error<std::runtime_error> {
public:
    template <typename... Args>
    explicit DeviceError(Args&&... args)
        : Error<std::runtime_error>(std::forward<Args>(args)...)
    {}

    static std::string trimCallString(const char* where) {
        const char* paren = strchr(where, '(');
        if (paren) {
            return std::string(where, paren);
        } else {
            return std::string(where);
        }
    }
};

/**
 * Represents a generic runtime error (aka environmental problem)
 */
class RuntimeError : public Error<std::runtime_error> {
public:
    explicit RuntimeError(const std::string& reason)
        : Error("Run-time error: " + reason)
    {}
};

/**
 * Represents a generic logic error (aka failed assertion)
 */
class LogicError : public Error<std::logic_error> {
public:
    explicit LogicError(const std::string& reason)
        : Error("Internal logic error: " + reason)
    {}
};

/**
 * Internal exception base class with a status field and a sublcass-specific
 * "details" field which can be used to recreate an exception
 */
template <typename Base, typename Status>
class ErrorCode : public Base {
public:
    ErrorCode(Status status, const std::string& details, const std::string& reason)
        : Base(reason), status_(status), details_(details)
    {}

    Status status() const noexcept {
        return status_;
    }

    const std::string& details() const noexcept {
        return details_;
    }

private:
    const Status status_;
    const std::string details_;
};

/**
 * Represents a runtime error returned by an OpenCL/CUDA API function.
 */
class APIError : public ErrorCode<DeviceError, int> {
public:
    APIError(int status, const std::string& where, const std::string& detail)
        : ErrorCode(status, where, detail)
    {}
};

/**
 * Represents a runtime error returned by a runtime compilation.
 */
class BuildError : public ErrorCode<DeviceError, int> {
public:
    BuildError(int status, const std::string& where, const std::string& detail)
        : ErrorCode(status, where, detail)
    {}
};

//==-------------------------------------------------------------------------
// Definitions
//==-------------------------------------------------------------------------

// Enumeration of API types
enum class APITypes { OpenCL, CUDA };

// Enumeration of device types
enum class DeviceType { Default, CPU, GPU, Accelerator, All, Unknown };

// Enumeration of buffer access types
enum class BufferAccess { kReadOnly, kWriteOnly, kReadWrite };

using PlatformID = intptr_t;
using DeviceID = intptr_t;
using ContextID = intptr_t;

//==-------------------------------------------------------------------------
// Raw Interface
//==-------------------------------------------------------------------------

namespace raw {

// forward declaration
class Platform;
class Device;
class Context;
class Queue;
class Event;
class Buffer;
class Program;
class Kernel;

class Platform {
public:
    virtual ~Platform() = default;

    virtual PlatformID  id() const noexcept = 0;
    virtual APITypes    api() const noexcept = 0;
    virtual std::string name() const = 0;
    virtual std::string vendor() const = 0;
    virtual std::string version() const = 0;

    /**
     * Get all devices in this platform.
     */
    virtual std::vector<std::shared_ptr<Device>> devices(DeviceType type) const = 0;

    /**
     * Get the default device in this platform.
     */
    virtual std::shared_ptr<Device> device() = 0;
};

class Device {
public:
    virtual ~Device() = default;

    // Information
    virtual DeviceID    id() const noexcept = 0;
    virtual DeviceType  type() const = 0;
    virtual std::string version() const = 0;
    virtual std::string name() const = 0;
    virtual std::string vendor() const = 0;

    // Attributes
    virtual size_t maxWorkGroupSize() const = 0;
    virtual size_t maxWorkItemDimensions() const = 0;
    virtual std::vector<size_t> maxWorkItemSizes() const = 0;
    virtual uint64_t localMemSize() const = 0;

    // Capabilities
    virtual std::string capabilities() const = 0;
    virtual uint32_t coreClock() const = 0;
    virtual uint32_t computeUnits() const = 0;
    virtual uint64_t memorySize() const = 0;
    virtual uint64_t maxAllocSize() const = 0;

    /**
     * Create a new context.
     */
    virtual std::shared_ptr<Context> createContext() = 0;
};

class Context {
public:
    virtual ~Context() = default;

    virtual ContextID id() const noexcept = 0;

    virtual std::shared_ptr<Queue> createQueue() = 0;
    virtual std::shared_ptr<Event> createEvent() = 0;
    virtual std::shared_ptr<Buffer> createBuffer(BufferAccess access, size_t size) = 0;

    virtual std::shared_ptr<Program> compileProgram(const char* source, const std::vector<std::string>& options) = 0;
    virtual std::shared_ptr<Program> loadProgram(const std::string& binary) = 0;
};

class Queue {
public:
    virtual ~Queue() = default;

    virtual void finish(Event& event) = 0;
    virtual void finish() = 0;
};

class Event {
public:
    virtual ~Event() = default;

    virtual void waitForCompletion() = 0;
    virtual float getElapsedTime() = 0;
};

class Buffer {
public:
    virtual ~Buffer() = default;

    virtual void read(Queue& queue, void* host, size_t size, size_t offset) = 0;
    virtual void write(Queue& queue, const void* host, size_t size, size_t offset) = 0;
    virtual void copy(Queue& queue, Buffer& dest, size_t size) = 0;
};

class Program {
public:
    virtual ~Program() = default;
    virtual std::string getIR() = 0;
    virtual std::shared_ptr<Kernel> getKernel(const char* name) = 0;
};

class Kernel {
public:
    virtual ~Kernel() = default;

    virtual void setArgument(size_t index, const void* value, size_t size) = 0;
    virtual void setArgument(size_t index, const Buffer& buffer) = 0;
    virtual void launch(Queue& queue,
                        const std::vector<size_t>& global,
                        const std::vector<size_t>& local,
                        Event* event) = 0;
};

} // namespace raw

//==-------------------------------------------------------------------------
// Public Interface
//==-------------------------------------------------------------------------

// forward declaration
class Platform;
class Device;
class Context;
class Queue;
class Event;
template <typename T> class Buffer;
class Program;
class Kernel;

class Platform {
    std::shared_ptr<raw::Platform> m_raw;

    explicit Platform(std::shared_ptr<raw::Platform> raw)
        : m_raw(std::move(raw)) {}

    friend Platform probe();

public:
    Platform() = default;

    // Information
    PlatformID id() const noexcept {
        return m_raw->id();
    }

    APITypes api() const noexcept {
        return m_raw->api();
    }

    std::string name() const {
        return m_raw->name();
    }

    std::string vendor() const {
        return m_raw->vendor();
    }

    std::string version() const {
        return m_raw->version();
    };

    /**
     * Get all devices in this platform.
     */
    std::vector<Device> devices(DeviceType type) const;

    /**
     * Get the default device in this platform.
     */
    Device device();
};

class Device {
    Platform m_platform;
    std::shared_ptr<raw::Device> m_raw;

public:
    Device() = default;

    Device(Platform platform, std::shared_ptr<raw::Device> device)
        : m_platform(std::move(platform)), m_raw(std::move(device)) {}

    Platform& platform() {
        return m_platform;
    }

    DeviceID id() const noexcept {
        return m_raw->id();
    }

    std::string version() const {
        return m_raw->version();
    }

    std::string name() const {
        return m_raw->name();
    }

    std::string vendor() const {
        return m_raw->vendor();
    }

    DeviceType type() const {
        return m_raw->type();
    }

    std::string typeString() const {
        switch (type()) {
        case DeviceType::CPU:
            return "CPU";
        case DeviceType::GPU:
            return "GPU";
        case DeviceType::Accelerator:
            return "Accelerator";
        default:
            return "Unknown";
        }
    }

    size_t maxWorkGroupSize() const {
        return m_raw->maxWorkGroupSize();
    }

    size_t maxWorkItemDimensions() const {
        return m_raw->maxWorkItemDimensions();
    }

    std::vector<size_t> maxWorkItemSizes() const {
        return m_raw->maxWorkItemSizes();
    }

    uint64_t localMemSize() const {
        return m_raw->localMemSize();
    }

    std::string capabilities() const {
        return m_raw->capabilities();
    }

    bool hasExtension(const std::string& extension) const {
        return capabilities().find(extension) != std::string::npos;
    }

    uint32_t coreClock() const {
        return m_raw->coreClock();
    }

    uint32_t computeUnits() const {
        return m_raw->computeUnits();
    }

    uint64_t memorySize() const {
        return m_raw->memorySize();
    }

    uint64_t maxAllocSize() const {
        return m_raw->maxAllocSize();
    }

    /**
     * Create a new context.
     */
    Context createContext();
};

class Context {
    Device m_device;
    std::shared_ptr<raw::Context> m_raw;

    friend class Device;
    Context(Device device, std::shared_ptr<raw::Context> context)
        : m_device(std::move(device)), m_raw(std::move(context)) {}

public:
    Context() = default;

    /**
     * Returns the device that was created this context.
     */
    Device& device() noexcept {
        return m_device;
    }

    ContextID id() const noexcept {
        return m_raw->id();
    }

    /**
     * Compile program from source.
     */
    Program compileProgram(const char* source, const std::vector<std::string>& options);

    /**
     * Load program from binary.
     */
    Program loadProgram(const std::string& binary);

   /**
    * Create a queue that can be used to schedule commands such as
    * launching a kernel or performing a device-host memory copy.
    */
    Queue createQueue();

    /**
     * Create a event to be able to time kernels.
     */
    Event createEvent();

    /**
     * Create the device buffer.
     */
    template <typename T>
    Buffer<T> createBuffer(BufferAccess access, size_t size);
};

class Queue {
    Context m_context;
    std::shared_ptr<raw::Queue> m_raw;

    friend class Context;
    Queue(Context context, std::shared_ptr<raw::Queue> queue)
        : m_context(std::move(context)), m_raw(std::move(queue)) {}

public:
    Queue() = default;
    raw::Queue& raw() const { return *m_raw; }

    Context& context() noexcept {
        return m_context;
    }

    void finish(Event& event);
    void finish();
};

class Event {
    std::shared_ptr<raw::Event> m_raw;

    friend class Context;
    explicit Event(std::shared_ptr<raw::Event> event)
        : m_raw(std::move(event)) {}

public:
    Event() = default;
    raw::Event* raw() const { return m_raw.get(); }

    void waitForCompletion();
    float getElapsedTime() const;
};

template <typename T>
class Buffer {
    std::shared_ptr<raw::Buffer> m_raw;

    friend class Context;
    explicit Buffer(std::shared_ptr<raw::Buffer> buffer)
        : m_raw(std::move(buffer)) {}

public:
    Buffer() = default;
    raw::Buffer& raw() const { return *m_raw; }

    void readAsync(Queue& queue, T* host, size_t size, size_t offset = 0) const {
        m_raw->read(queue.raw(), host, size*sizeof(T), offset*sizeof(T));
    }

    void read(Queue& queue, T* host, const size_t size, size_t offset = 0) const {
        readAsync(queue, host, size, offset);
        queue.finish();
    }

    void writeAsync(Queue& queue, const T* host, size_t size, size_t offset = 0) {
        m_raw->write(queue.raw(), host, size*sizeof(T), offset*sizeof(T));
    }

    void write(Queue& queue, const T* host, size_t size, size_t offset = 0) {
        writeAsync(queue, host, size, offset);
        queue.finish();
    }

    void copyAsync(Queue& queue, Buffer& dest, size_t size) const {
        m_raw->copy(queue.raw(), dest.raw(), size*sizeof(T));
    }

    void copy(Queue& queue, Buffer& dest, size_t size) const {
        copyAsync(queue, dest, size);
        queue.finish();
    }
};

class Program {
    Context m_context;
    std::shared_ptr<raw::Program> m_raw;

    friend class Context;
    Program(Context context, std::shared_ptr<raw::Program> program)
        : m_context(std::move(context)), m_raw(std::move(program)) {}

public:
    Program() = default;

    Context& context() noexcept {
        return m_context;
    }

    std::string getIR();
    Kernel getKernel(const char* name);
};

class Kernel {
    std::shared_ptr<raw::Kernel> m_raw;

    friend class Program;
    explicit Kernel(std::shared_ptr<raw::Kernel> kernel)
        : m_raw(std::move(kernel)) {}

public:
    Kernel() = default;

    template <typename T>
    void setArgument(size_t index, const T& value) {
        m_raw->setArgument(index, &value, sizeof(T));
    }

    template <typename T>
    void setArgument(size_t index, Buffer<T>& buffer) {
        m_raw->setArgument(index, buffer.raw());
    }

    template <typename T, typename... Ts>
    void setArguments(T&& first, Ts&&... rest) {
        setArgumentsRec(0, std::forward<T>(first), std::forward<Ts>(rest)...);
    }

    void launch(const Queue& queue,
                const std::vector<size_t>& global,
                const std::vector<size_t>& local,
                Event* event = nullptr)
    {
        auto ev = event==nullptr ? nullptr : event->raw();
        m_raw->launch(queue.raw(), global, local, ev);
    }

private:
    template <typename T>
    void setArgumentsRec(size_t index, T&& last) {
        setArgument(index, std::forward<T>(last));
    }

    template <typename T, typename... Ts>
    void setArgumentsRec(size_t index, T&& first, Ts&&... rest) {
        setArgument(index, std::forward<T>(first));
        setArgumentsRec(index+1, std::forward<Ts>(rest)...);
    }
};

//==-------------------------------------------------------------------------
// Global functions
//==-------------------------------------------------------------------------

extern Platform probe();

inline bool isOpenCL() {
    return probe().api() == APITypes::OpenCL;
}

inline bool isCUDA() {
    return probe().api() == APITypes::CUDA;
}

//==-------------------------------------------------------------------------
// Implementation
//==-------------------------------------------------------------------------

inline Device Platform::device() {
    return Device{*this, m_raw->device()};
}

inline Context Device::createContext() {
    return Context{*this, m_raw->createContext()};
}

inline Queue Context::createQueue() {
    return Queue(*this, m_raw->createQueue());
}

inline Event Context::createEvent() {
    return Event(m_raw->createEvent());
}

template <typename T>
inline Buffer<T> Context::createBuffer(BufferAccess access, size_t size) {
    return Buffer<T>(m_raw->createBuffer(access, size*sizeof(T)));
}

inline void Queue::finish(Event& event) {
    m_raw->finish(*event.raw());
}

inline void Queue::finish() {
    m_raw->finish();
}

inline void Event::waitForCompletion() {
    m_raw->waitForCompletion();
}

inline float Event::getElapsedTime() const {
    return m_raw->getElapsedTime();
}

inline Program Context::compileProgram(const char* source, const std::vector<std::string>& options) {
    return Program(*this, m_raw->compileProgram(source, options));
}

inline Program Context::loadProgram(const std::string& binary) {
    return Program(*this, m_raw->loadProgram(binary));
}

inline std::string Program::getIR() {
    return m_raw->getIR();
}

inline Kernel Program::getKernel(const char* name) {
    return Kernel(m_raw->getKernel(name));
}

//==-------------------------------------------------------------------------
// BLAS (Basic Linear Algebra Subprograms) interface
//==-------------------------------------------------------------------------

namespace blas {

using ::blas::Layout;
using ::blas::Transpose;

//==-------------------------------------------------------------------------
// BLAS level-1 (vector-vector) routines
//==-------------------------------------------------------------------------

template <typename T>
void asum(size_t N, const Buffer<T>& X, size_t incX, Buffer<T>& result, Queue& queue, Event* event = nullptr);

template <typename T>
void axpy(size_t N, const T alpha, const Buffer<T>& X, size_t incX, Buffer<T>& Y, size_t incY,
          Queue& queue, Event* event = nullptr);

template <typename T>
void copy(size_t N, const Buffer<T>& X, size_t incX, Buffer<T>& Y, size_t incY, Queue& queue, Event* event = nullptr);

template <typename T>
void dot(size_t N, const Buffer<T>& X, size_t incX, const Buffer<T>& Y, size_t incY,
        Buffer<T>& result, Queue& queue, Event* event = nullptr);

template <typename T>
void nrm2(size_t N, Buffer<T>& X, size_t incX, Buffer<T>& result, Queue& queue, Event* event = nullptr);

template <typename T>
void scal(const size_t N, const T alpha, Buffer<T>& X, const size_t incX,
          Queue& queue, Event* event = nullptr);

template <typename T>
void swap(size_t N, Buffer<T>& X, size_t incX, Buffer<T>& Y, size_t incY, Queue& queue, Event* event = nullptr);

//==-------------------------------------------------------------------------
// BLAS level-2 (matrix-vector) routines
//==-------------------------------------------------------------------------

template <typename T>
void gemv(Layout layout, Transpose trans, size_t M, size_t N, const T alpha,
          const Buffer<T>& A, size_t lda, const Buffer<T>X, size_t incX,
          const T beta, Buffer<T>& Y, size_t incY,
          Queue& queue, Event* event = nullptr);

//==-------------------------------------------------------------------------
// BLAS level-3 (matrix-matrix) routines
//==-------------------------------------------------------------------------

template <typename T>
void gemm(const Layout layout, const Transpose transA, const Transpose transB,
          const size_t M, const size_t N, const size_t K,
          const T alpha,
          const Buffer<T>& A, const size_t lda,
          const Buffer<T>& B, const size_t ldb,
          const T beta,
          Buffer<T>& C, const size_t ldc,
          Queue& queue, Event* event = nullptr);

} // namespace blas
} // namespace gpgpu

#endif //_GPGPU_H
