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

//==-------------------------------------------------------------------------
// Raw Interface
//==-------------------------------------------------------------------------

// forward declaration
class rawPlatform;
class rawDevice;
class rawContext;
class rawQueue;
class rawEvent;
class rawBuffer;
class rawProgram;
class rawKernel;

class rawPlatform {
public:
    rawPlatform() = default;
    virtual ~rawPlatform() = default;

    rawPlatform(const rawPlatform&) = delete;
    rawPlatform(rawPlatform&&) = delete;
    rawPlatform& operator=(const rawPlatform&) = delete;
    rawPlatform& operator=(rawPlatform&&) = delete;

    virtual PlatformID  id() const noexcept = 0;
    virtual APIType     api() const noexcept = 0;
    virtual std::string name() const = 0;
    virtual std::string vendor() const = 0;
    virtual std::string version() const = 0;

    /**
     * Get all devices in this platform.
     */
    virtual std::vector<std::shared_ptr<rawDevice>> devices(DeviceType type) const = 0;
};

class rawDevice {
public:
    rawDevice() = default;
    virtual ~rawDevice() = default;

    rawDevice(const rawDevice&) = delete;
    rawDevice(rawDevice&&) = delete;
    rawDevice& operator=(const rawDevice&) = delete;
    rawDevice& operator=(rawDevice&&) = delete;

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
    virtual std::shared_ptr<rawContext> createContext() const = 0;
};

class rawContext {
public:
    rawContext() = default;
    virtual ~rawContext() = default;

    rawContext(const rawContext&) = delete;
    rawContext(rawContext&&) = delete;
    rawContext& operator=(const rawContext&) = delete;
    rawContext& operator=(rawContext&&) = delete;

    virtual ContextID id() const noexcept = 0;

    virtual void activate() const = 0;
    virtual void deactivate() const = 0;

    virtual std::shared_ptr<rawQueue> createQueue() const = 0;
    virtual std::shared_ptr<rawEvent> createEvent() const = 0;
    virtual std::shared_ptr<rawBuffer> createBuffer(size_t size, BufferAccess access) const = 0;

    virtual std::shared_ptr<rawProgram> compileProgram(
        const char* source, const std::vector<std::string>& options) const = 0;
    virtual std::shared_ptr<rawProgram> loadProgram(const std::string& binary) const = 0;
};

class rawQueue {
public:
    rawQueue() = default;
    virtual ~rawQueue() = default;

    rawQueue(const rawQueue&) = delete;
    rawQueue(rawQueue&&) = delete;
    rawQueue& operator=(const rawQueue&) = delete;
    rawQueue& operator=(rawQueue&&) = delete;

    virtual QueueID id() const noexcept = 0;
    virtual void finish(rawEvent& event) const = 0;
    virtual void finish() const = 0;
};

class rawEvent {
public:
    rawEvent() = default;
    virtual ~rawEvent() = default;

    rawEvent(const rawEvent&) = delete;
    rawEvent(rawEvent&&) = delete;
    rawEvent& operator=(const rawEvent&) = delete;
    rawEvent& operator=(rawEvent&&) = delete;

    virtual void waitForCompletion() const = 0;
    virtual float getElapsedTime() const = 0;
};

class rawBuffer {
public:
    rawBuffer() = default;
    virtual ~rawBuffer() = default;

    rawBuffer(const rawBuffer&) = delete;
    rawBuffer(rawBuffer&&) = delete;
    rawBuffer& operator=(const rawBuffer&) = delete;
    rawBuffer& operator=(rawBuffer&&) = delete;

    virtual void read(const rawQueue& queue, void* host, size_t size, size_t offset, rawEvent* event) const = 0;
    virtual void write(const rawQueue& queue, const void* host, size_t size, size_t offset, rawEvent* event) = 0;
    virtual void copyTo(const rawQueue& queue, rawBuffer& dest, size_t size, rawEvent* event) const = 0;
};

class rawProgram {
public:
    rawProgram() = default;
    virtual ~rawProgram() = default;

    rawProgram(const rawProgram&) = delete;
    rawProgram(rawProgram&&) = delete;
    rawProgram& operator=(const rawProgram&) = delete;
    rawProgram& operator=(rawProgram&&) = delete;

    virtual std::string getIR() const = 0;
    virtual std::shared_ptr<rawKernel> getKernel(const char* name) const = 0;
};

class rawKernel {
public:
    rawKernel() = default;
    virtual ~rawKernel() = default;

    rawKernel(const rawKernel&) = delete;
    rawKernel(rawKernel&&) = delete;
    rawKernel& operator=(const rawKernel&) = delete;
    rawKernel& operator=(rawKernel&&) = delete;

    virtual uint64_t localMemoryUsage(const rawDevice& device) const = 0;

    virtual void setArgument(size_t index, const void* value, size_t size) const = 0;
    virtual void setArgument(size_t index, const rawBuffer& buffer) const = 0;

    virtual void launch(const rawQueue& queue,
                        const std::vector<size_t>& global,
                        const std::vector<size_t>& local,
                        rawEvent* event) const = 0;
};

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
    std::shared_ptr<rawPlatform> m_raw;

    explicit Platform(std::shared_ptr<rawPlatform> raw)
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
    std::shared_ptr<rawDevice> m_raw;

public:
    Device() = default;
    const rawDevice& raw() const noexcept { return *m_raw; }

    Device(Platform platform, std::shared_ptr<rawDevice> device)
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
    std::shared_ptr<rawContext> m_raw;

    friend class Device;
    Context(Device device, std::shared_ptr<rawContext> context)
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

class Queue {
    Context m_context;
    std::shared_ptr<rawQueue> m_raw;

    friend class Context;
    Queue(Context context, std::shared_ptr<rawQueue> queue)
        : m_context(std::move(context)), m_raw(std::move(queue)) {}

public:
    Queue() = default;
    const rawQueue& raw() const noexcept { return *m_raw; }

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
    std::shared_ptr<rawEvent> m_raw;

    friend class Context;
    explicit Event(std::shared_ptr<rawEvent> event)
        : m_raw(std::move(event)) {}

public:
    rawEvent* raw() const noexcept { return m_raw.get(); }

    void waitForCompletion() const;
    float getElapsedTime() const;
};

template <typename T>
class Buffer {
    std::shared_ptr<rawBuffer> m_raw;
    size_t m_size;

    friend class Context;

public:
    Buffer() = default;
    explicit Buffer(std::shared_ptr<rawBuffer> buffer, size_t size)
        : m_raw(std::move(buffer)), m_size(size) {}

    rawBuffer& raw() const noexcept { return *m_raw; }
    std::shared_ptr<rawBuffer> handle() const noexcept { return m_raw; }

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

template <typename T>
bool operator==(const Buffer<T>& lhs, const Buffer<T>& rhs) {
    return lhs.handle() == rhs.handle();
}

template <typename T>
bool operator!=(const Buffer<T>& lhs, const Buffer<T>& rhs) {
    return lhs.handle() != rhs.handle();
}

class Program {
    std::shared_ptr<rawProgram> m_raw;

    friend class Context;
    Program(std::shared_ptr<rawProgram> program)
        : m_raw(std::move(program)) {}

public:
    Program() = default;

    std::string getIR() const;
    Kernel getKernel(const char* name) const;
    Kernel getKernel(const std::string& name) const;
};

class Kernel {
    Program m_program;
    std::shared_ptr<rawKernel> m_raw;

    friend class Program;
    explicit Kernel(Program program, std::shared_ptr<rawKernel> kernel)
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
    /**
     * Create a new context and associate it with current thread.
     */
    current(const Queue& queue);

    /**
     * Deassociate context from current thread.
     */
     ~current();

     current(const current&) = delete;
     current(current&&) = delete;
     current& operator=(const current&) = delete;
     current& operator=(current&&) = delete;

    /**
     * Returns the context associated with current thread.
     */
    static const Context& context();

    /**
     * Returns the queue associated with current thread.
     */
    static const Queue& queue();

private:
    Queue previous_queue;
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
