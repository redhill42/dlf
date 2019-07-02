#ifndef GPGPU_H_
#define GPGPU_H_

#include <string>
#include <cstring>
#include <vector>
#include <memory>
#include <stdexcept>

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

/**
 * Represents no device found in the system.
 */
class NoDeviceFound : public RuntimeError {
public:
    NoDeviceFound() : RuntimeError("No device found") {}
};

//==-------------------------------------------------------------------------
// Definitions
//==-------------------------------------------------------------------------

// Enumeration of API types
enum class APIType { OpenCL, CUDA };

// Enumeration of device types
enum class DeviceType { Default, CPU, GPU, Accelerator, All, Unknown };

// Enumeration of buffer access types
enum class BufferAccess { ReadOnly, WriteOnly, ReadWrite };

using PlatformID = intptr_t;
using DeviceID = intptr_t;
using ContextID = intptr_t;
using QueueID = intptr_t;

// forward declaration
class Platform;
class Device;
class Context;
class Queue;
class Event;
template <typename T> class Buffer;
class Program;
class Kernel;

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
    Platform() = default;
    virtual ~Platform() = default;

    Platform(const Platform&) = delete;
    Platform(Platform&&) = delete;
    Platform& operator=(const Platform&) = delete;
    Platform& operator=(Platform&&) = delete;

    virtual PlatformID  id() const noexcept = 0;
    virtual APIType     api() const noexcept = 0;
    virtual std::string name() const = 0;
    virtual std::string vendor() const = 0;
    virtual std::string version() const = 0;

    /**
     * Get all devices in this platform.
     */
    virtual std::vector<std::shared_ptr<Device>> devices(DeviceType type) const = 0;
};

class Device {
public:
    Device() = default;
    virtual ~Device() = default;

    Device(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(const Device&) = delete;
    Device& operator=(Device&&) = delete;

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
    virtual bool supportsFP64() const = 0;
    virtual bool supportsFP16() const = 0;

    // Capabilities
    virtual std::string capabilities() const = 0;
    virtual uint32_t coreClock() const = 0;
    virtual uint32_t computeUnits() const = 0;
    virtual uint64_t memorySize() const = 0;
    virtual uint64_t maxAllocSize() const = 0;

    /**
     * Create a new context.
     */
    virtual std::shared_ptr<Context> createContext() const = 0;
};

class Context {
public:
    Context() = default;
    virtual ~Context() = default;

    Context(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(const Context&) = delete;
    Context& operator=(Context&&) = delete;

    virtual ContextID id() const noexcept = 0;

    virtual void activate() const = 0;
    virtual void deactivate() const = 0;

    virtual std::shared_ptr<Queue> createQueue() const = 0;
    virtual std::shared_ptr<Event> createEvent() const = 0;
    virtual std::shared_ptr<Buffer> createBuffer(size_t size, BufferAccess access) const = 0;

    virtual std::shared_ptr<Program> compileProgram(
        const char* source, const std::vector<std::string>& options) const = 0;
    virtual std::shared_ptr<Program> loadProgram(const std::string& binary) const = 0;
};

class Queue {
public:
    Queue() = default;
    virtual ~Queue() = default;

    Queue(const Queue&) = delete;
    Queue(Queue&&) = delete;
    Queue& operator=(const Queue&) = delete;
    Queue& operator=(Queue&&) = delete;

    virtual QueueID id() const noexcept = 0;
    virtual void finish(Event& event) const = 0;
    virtual void finish() const = 0;
};

class Event {
public:
    Event() = default;
    virtual ~Event() = default;

    Event(const Event&) = delete;
    Event(Event&&) = delete;
    Event& operator=(const Event&) = delete;
    Event& operator=(Event&&) = delete;

    virtual void waitForCompletion() const = 0;
    virtual float getElapsedTime() const = 0;
};

class Buffer {
public:
    Buffer() = default;
    virtual ~Buffer() = default;

    Buffer(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer& operator=(Buffer&&) = delete;

    virtual void read(const Queue& queue, void* host, size_t size, size_t offset, Event* event) const = 0;
    virtual void write(const Queue& queue, const void* host, size_t size, size_t offset, Event* event) = 0;
    virtual void copyTo(const Queue& queue, Buffer& dest, size_t size, Event* event) const = 0;
};

class Program {
public:
    Program() = default;
    virtual ~Program() = default;

    Program(const Program&) = delete;
    Program(Program&&) = delete;
    Program& operator=(const Program&) = delete;
    Program& operator=(Program&&) = delete;

    virtual std::string getIR() const = 0;
    virtual std::shared_ptr<Kernel> getKernel(const char* name) const = 0;
};

class Kernel {
public:
    Kernel() = default;
    virtual ~Kernel() = default;

    Kernel(const Kernel&) = delete;
    Kernel(Kernel&&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    Kernel& operator=(Kernel&&) = delete;

    virtual uint64_t localMemoryUsage(const Device& device) const = 0;

    virtual void setArgument(size_t index, const void* value, size_t size) const = 0;
    virtual void setArgument(size_t index, const Buffer& buffer) const = 0;

    virtual void launch(const Queue& queue,
                        const std::vector<size_t>& global,
                        const std::vector<size_t>& local,
                        Event* event) const = 0;
};

} // namespace raw

//==-------------------------------------------------------------------------
// Public Interface
//==-------------------------------------------------------------------------

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

    APIType api() const noexcept {
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
    std::vector<Device> devices(DeviceType type = DeviceType::GPU) const;

    bool operator==(const Platform& other) const noexcept {
        return m_raw->id() == other.m_raw->id();
    }

    bool operator!=(const Platform& other) const noexcept {
        return !(*this == other);
    }
};

class Device {
    Platform m_platform;
    std::shared_ptr<raw::Device> m_raw;

public:
    Device() = default;
    const raw::Device& raw() const noexcept { return *m_raw; }

    Device(Platform platform, std::shared_ptr<raw::Device> device)
        : m_platform(std::move(platform)), m_raw(std::move(device)) {}

    const Platform& platform() const noexcept {
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

    bool supportsFP64() const {
        return m_raw->supportsFP64();
    }

    bool supportsFP16() const {
        return m_raw->supportsFP16();
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
    Context createContext() const;


    bool operator==(const Device& other) const noexcept {
        return m_raw->id() == other.m_raw->id();
    }

    bool operator!=(const Device& other) const noexcept {
        return !(*this == other);
    }
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
    const Device& device() const noexcept {
        return m_device;
    }

    ContextID id() const noexcept {
        return m_raw == nullptr ? 0 : m_raw->id();
    }

    /**
     * Activate thi context and associate it to current CPU thread.
     */
    const Context& activate() const {
        m_raw->activate();
        return *this;
    }

    /**
     * Deactivate this context and disassociate it from current CPU thread.
     */
    void deactivate() const {
        m_raw->deactivate();
    }

    /**
     * Compile program from source.
     */
    Program compileProgram(const char* source, const std::vector<std::string>& options) const;

    /**
     * Load program from binary.
     */
    Program loadProgram(const std::string& binary)const;

   /**
    * Create a queue that can be used to schedule commands such as
    * launching a kernel or performing a device-host memory copy.
    */
    Queue createQueue() const;

    /**
     * Create a event to be able to time kernels.
     */
    Event createEvent() const;

    /**
     * Create the device buffer.
     */
    template <typename T>
    Buffer<T> createBuffer(size_t size, BufferAccess access = BufferAccess::ReadWrite) const;

    bool operator==(const Context& other) const noexcept {
        return m_raw->id() == other.m_raw->id();
    }

    bool operator!=(const Context& other) const noexcept {
        return !(*this == other);
    }
};

/**
 * The context activation guard.
 */
class ContextActivation {
    const Context& m_context;

public:
    explicit ContextActivation(const Context& context) : m_context(context) {
        m_context.activate();
    }

    ~ContextActivation() {
        m_context.deactivate();
    }

    ContextActivation(const ContextActivation&) = delete;
    ContextActivation& operator=(const ContextActivation&) = delete;
};

class Queue {
    Context m_context;
    std::shared_ptr<raw::Queue> m_raw;

    friend class Context;
    Queue(Context context, std::shared_ptr<raw::Queue> queue)
        : m_context(std::move(context)), m_raw(std::move(queue)) {}

public:
    Queue() = default;
    const raw::Queue& raw() const noexcept { return *m_raw; }

    QueueID id() const noexcept {
        return m_raw == nullptr ? 0 : m_raw->id();
    }

    const Context& context() const noexcept {
        return m_context;
    }

    void finish(Event& event) const;
    void finish() const;

    bool operator==(const Queue& other) const noexcept {
        return m_raw->id() == other.m_raw->id();
    }

    bool operator!=(const Queue& other) const noexcept {
        return !(*this == other);
    }
};

class Event {
    std::shared_ptr<raw::Event> m_raw;

    friend class Context;
    explicit Event(std::shared_ptr<raw::Event> event)
        : m_raw(std::move(event)) {}

public:
    raw::Event* raw() const noexcept { return m_raw.get(); }

    void waitForCompletion() const;
    float getElapsedTime() const;
};

template <typename T>
class Buffer {
    std::shared_ptr<raw::Buffer> m_raw;
    size_t m_size;

    friend class Context;

public:
    Buffer() = default;
    explicit Buffer(std::shared_ptr<raw::Buffer> buffer, size_t size)
        : m_raw(std::move(buffer)), m_size(size) {}

    raw::Buffer& raw() const noexcept { return *m_raw; }
    std::shared_ptr<raw::Buffer> handle() const noexcept { return m_raw; }

    size_t size() const noexcept { return m_size; }
    size_t data_size() const noexcept { return m_size * sizeof(T); }

    void readAsync(const Queue& queue, T* host, size_t size, size_t offset = 0, Event* event = nullptr) const {
        auto ev = event== nullptr ? nullptr : event->raw();
        m_raw->read(queue.raw(), host, size*sizeof(T), offset*sizeof(T), ev);
    }

    void read(const Queue& queue, T* host, const size_t size, size_t offset = 0) const {
        readAsync(queue, host, size, offset);
        queue.finish();
    }

    void writeAsync(const Queue& queue, const T* host, size_t size, size_t offset = 0, Event* event = nullptr) {
        auto ev = event== nullptr ? nullptr : event->raw();
        m_raw->write(queue.raw(), host, size*sizeof(T), offset*sizeof(T), ev);
    }

    void write(const Queue& queue, const T* host, size_t size, size_t offset = 0) {
        writeAsync(queue, host, size, offset);
        queue.finish();
    }

    void copyToAsync(const Queue &queue, Buffer &dest, size_t size, Event* event = nullptr) const {
        auto ev = event== nullptr ? nullptr : event->raw();
        m_raw->copyTo(queue.raw(), dest.raw(), size * sizeof(T), ev);
    }

    void copyTo(const Queue &queue, Buffer &dest, size_t size) const {
        copyToAsync(queue, dest, size);
        queue.finish();
    }
};

class Program {
    std::shared_ptr<raw::Program> m_raw;

    friend class Context;
    Program(std::shared_ptr<raw::Program> program)
        : m_raw(std::move(program)) {}

public:
    Program() = default;

    std::string getIR() const;
    Kernel getKernel(const char* name) const;
    Kernel getKernel(const std::string& name) const;
};

class Kernel {
    Program m_program;
    std::shared_ptr<raw::Kernel> m_raw;

    friend class Program;
    explicit Kernel(Program program, std::shared_ptr<raw::Kernel> kernel)
        : m_program(std::move(program)), m_raw(std::move(kernel)) {}

public:
    Kernel() = default;

    const Program& program() const noexcept {
        return m_program;
    }

    uint64_t localMemoryUsage(const Device& device) const {
        return m_raw->localMemoryUsage(device.raw());
    }

    template <typename T>
    void setArgument(size_t index, const T& value) const {
        m_raw->setArgument(index, &value, sizeof(T));
    }

    template <typename T>
    void setArgument(size_t index, const Buffer<T>& buffer) const {
        m_raw->setArgument(index, buffer.raw());
    }

    template <typename T, typename... Ts>
    void setArguments(T&& first, Ts&&... rest) const {
        setArgumentsRec(0, std::forward<T>(first), std::forward<Ts>(rest)...);
    }

    void launch(const Queue& queue,
                const std::vector<size_t>& global,
                const std::vector<size_t>& local,
                Event* event = nullptr) const
    {
        auto ev = event==nullptr ? nullptr : event->raw();
        m_raw->launch(queue.raw(), global, local, ev);
    }

private:
    template <typename T>
    void setArgumentsRec(size_t index, T&& last) const {
        setArgument(index, std::forward<T>(last));
    }

    template <typename T, typename... Ts>
    void setArgumentsRec(size_t index, T&& first, Ts&&... rest) const {
        setArgument(index, std::forward<T>(first));
        setArgumentsRec(index+1, std::forward<Ts>(rest)...);
    }
};

class current {
public:
    current() = delete; // disable construction

    /**
     * Returns the context associated with current thread.
     */
    static const Context& context();

    /**
     * Returns the queue associated with current thread.
     */
    static const Queue& queue();
};

//==-------------------------------------------------------------------------
// Global functions
//==-------------------------------------------------------------------------

extern Platform probe();

extern std::vector<bool> parseDeviceFilter(int num_devices, const char* env);
extern std::vector<bool> parseDeviceFilter(int num_devices);

//==-------------------------------------------------------------------------
// Implementation
//==-------------------------------------------------------------------------

inline Context Device::createContext() const {
    return Context{*this, m_raw->createContext()};
}

inline Queue Context::createQueue() const {
    return Queue(*this, m_raw->createQueue());
}

inline Event Context::createEvent() const {
    return Event(m_raw->createEvent());
}

template <typename T>
inline Buffer<T> Context::createBuffer(size_t size, BufferAccess access) const {
    return Buffer<T>(m_raw->createBuffer(size*sizeof(T), access), size);
}

inline void Queue::finish(Event& event) const {
    m_raw->finish(*event.raw());
}

inline void Queue::finish() const {
    m_raw->finish();
}

inline void Event::waitForCompletion() const {
    m_raw->waitForCompletion();
}

inline float Event::getElapsedTime() const {
    return m_raw->getElapsedTime();
}

inline Program Context::compileProgram(
    const char* source, const std::vector<std::string>& options) const
{
    return Program(m_raw->compileProgram(source, options));
}

inline Program Context::loadProgram(const std::string& binary) const {
    return Program(m_raw->loadProgram(binary));
}

inline std::string Program::getIR() const {
    return m_raw->getIR();
}

inline Kernel Program::getKernel(const char* name) const {
    return Kernel(*this, m_raw->getKernel(name));
}

inline Kernel Program::getKernel(const std::string& name) const {
    return getKernel(name.c_str());
}

} // namespace gpgpu

#endif //GPGPU_H_
