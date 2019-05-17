#ifndef KNERON_CONCURRENT_H
#define KNERON_CONCURRENT_H

#include <queue>
#include <vector>
#include <thread>
#include <future>
#include <mutex>

#include <tbb/tbb.h>

namespace kneron::concurrent {

#ifndef GRAINSIZE
#define GRAINSIZE 1000
#endif

template <class InputIterator, class OutputIterator>
inline void
parallel_copy(size_t grainsize, InputIterator first, InputIterator last, OutputIterator result) {
    tbb::parallel_for(tbb::blocked_range(first, last, grainsize), [&](const auto& r) {
        auto offset = std::distance(first, r.begin());
        std::copy(r.begin(), r.end(), std::next(result, offset));
    });
}

template <class InputIterator, class OutputIterator>
inline void
parallel_copy(InputIterator first, InputIterator last, OutputIterator result) {
    parallel_copy(GRAINSIZE, first, last, result);
}

template <class InputIterator, class OutputIterator, class UnaryOperation>
inline void
parallel_transform(size_t grainsize, InputIterator first, InputIterator last,
                   OutputIterator result, UnaryOperation op)
{
    tbb::parallel_for(tbb::blocked_range(first, last, grainsize), [&](const auto& r) {
        auto offset = std::distance(first, r.begin());
        std::transform(r.begin(), r.end(), std::next(result, offset), op);
    });
}

template <class InputIterator, class OutputIterator, class UnaryOperation>
inline void
parallel_transform(InputIterator first, InputIterator last,
                   OutputIterator result, UnaryOperation op)
{
    parallel_transform(GRAINSIZE, first, last, result, op);
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryOperation>
inline void
parallel_transform(size_t grainsize, InputIterator1 first1, InputIterator1 last1,
                   InputIterator2 first2, OutputIterator result, BinaryOperation op)
{
    tbb::parallel_for(tbb::blocked_range(first1, last1, grainsize), [&](const auto& r) {
        auto offset = std::distance(first1, r.begin());
        std::transform(r.begin(), r.end(), std::next(first2, offset), std::next(result, offset), op);
    });
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryOperation>
inline void
parallel_transform(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                   OutputIterator result, BinaryOperation op)
{
    parallel_transform(GRAINSIZE, first1, last1, first2, result, op);
}

//==-------------------------------------------------------------------------

/**
 * A blocking queue that supports operations that wait for the queue to become
 * non-empty when retrieving an element, and wait for space to become available
 * in the queue when storing an element.
 *
 * @tparam E the type of elements held in this queue
 */
template <typename E>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t capacity = std::numeric_limits<size_t>::max());

    ~BlockingQueue() { invalidate(); clear(); }

    /**
     * Inserts the specified element into this queue, waiting if necessary
     * for space to become available.
     *
     * @param e the element to add
     * @return true if element is enqueued, false if the queue is invalid
      */
    bool put(E e);
    bool put(E* e);

    /**
     * Inserts the specified element into this queue if it is possible to do
     * so immediately without violating capacity restrictions, returning
     * true upon success and false if no space is currently available.
     *
     * @param e the element to add
     * @return true if the element was added to this queue, else false
     */
    bool offer(E e);
    bool offer(E* e);

    /**
     * Inserts the specified element into this queue, waiting up to the
     * specified wait time if necessary for space to become available.
     *
     * @param e the element to add
     * @param timeout how long to wait before giving up
     */
    template <typename Rep, typename Duration>
    bool offer(E e, std::chrono::duration<Rep, Duration> timeout);
    template <typename Rep, typename Duration>
    bool offer(E* e, std::chrono::duration<Rep, Duration> timeout);

    /**
     * Retrieves and removes the head of this queue, waiting if necessary
     * until an element becomes available.
     *
     * @param p the address of element to take, must not be null
     * @return true if an element is dequeued, false if the queue is invalid
     */
    bool take(E *p);

    /**
     * Retrieves and removes the head of this queue if an element is available.
     *
     * @param p the address of element to poll, must not be null
     * @return true if element is available, false otherwise
     */
    bool poll(E *p);

    /**
     * Retrieves and removes the head of this queue, waiting up to the
     * specified wait time if necessary for an element to become available.
     *
     * @param p the address of element to poll, must not be null
     * @param timeout how long to wait before giving up.
     * @return true if element is available, false if the specified
     * waiting time elapses before an element is available.
     */
    template <typename Rep, typename Duration>
    bool poll(E *p, std::chrono::duration<Rep, Duration> timeout);

    /**
     * Returns true if this queue is empty, false otherwise.
     */
    bool empty() const noexcept;

    /**
     * Returns number elements available in the queue.
     */
    size_t size() const noexcept;

    /**
     * Returns the number of additional elements that this queue can ideally
     * (in the absence of memory or resource constraints) accept without
     * blocking.
     */
    size_t remaining() const noexcept;

    /**
     * Remove all elements from queue.
     */
    void clear();

    /**
     * Invalidate the queue. No more elements could be enqueued and all
     * waiting thread will be notified.
     */
    void invalidate();

    /**
     * Returns validate state of this queue.
     */
    bool valid() const noexcept;

private:
    /** The underlying queue */
    std::queue<E> m_queue;

    /** The capacity bound, or maximum size_t value if none */
    std::atomic<size_t> m_capacity;

    /** Main lock guarding all access */
    mutable std::mutex m_lock;

    /** Wait queue for waiting takes */
    std::condition_variable m_notEmpty;

    /** Wait queue for waiting puts */
    std::condition_variable m_notFull;

    bool enqueue(std::unique_lock<std::mutex>& lock, E* e);
    bool dequeue(std::unique_lock<std::mutex>& lock, E* p);
};

template <typename E>
BlockingQueue<E>::BlockingQueue(size_t capacity) {
    if (capacity == 0)
        capacity = std::numeric_limits<size_t>::max();
    m_capacity = capacity;
}

template <typename E>
inline bool BlockingQueue<E>::empty() const noexcept {
    std::unique_lock lock(m_lock);
    return m_queue.empty();
}

template <typename E>
inline size_t BlockingQueue<E>::size() const noexcept {
    std::unique_lock lock(m_lock);
    return m_queue.size();
}

template <typename E>
inline size_t BlockingQueue<E>::remaining() const noexcept {
    std::unique_lock lock(m_lock);
    return m_capacity > m_queue.size() ? m_capacity - m_queue.size() : 0;
}

template <typename E>
bool BlockingQueue<E>::enqueue(std::unique_lock<std::mutex>& lock, E* e) {
    if (m_queue.size() < m_capacity) {
        m_queue.push(std::move(*e));
        lock.unlock();
        m_notEmpty.notify_one();
        return true;
    } else {
        return false;
    }
}

template <typename E>
bool BlockingQueue<E>::dequeue(std::unique_lock<std::mutex>& lock, E* p) {
    if (!m_queue.empty()) {
        *p = std::move(m_queue.front());
        m_queue.pop();
        lock.unlock();
        m_notFull.notify_one();
        return true;
    } else {
        return false;
    }
}

template <typename E>
bool BlockingQueue<E>::put(E e) {
    std::unique_lock lock{m_lock};
    m_notFull.wait(lock, [this]() {
        return m_capacity == 0 || m_queue.size() < m_capacity;
    });
    return enqueue(lock, &e);
}

template <typename E>
bool BlockingQueue<E>::put(E* e) {
    std::unique_lock lock{m_lock};
    m_notFull.wait(lock, [this]() {
        return m_capacity == 0 || m_queue.size() < m_capacity;
    });
    return enqueue(lock, e);
}

template <typename E>
bool BlockingQueue<E>::offer(E e) {
    std::unique_lock lock{m_lock};
    return enqueue(lock, &e);
}

template <typename E>
bool BlockingQueue<E>::offer(E* e) {
    std::unique_lock lock{m_lock};
    return enqueue(lock, e);
}

template <typename E>
template <typename Rep, typename Period>
bool BlockingQueue<E>::offer(E e, std::chrono::duration<Rep,Period> timeout) {
    std::unique_lock lock{m_lock};
    m_notFull.wait_for(lock, timeout, [this]() {
        return m_capacity == 0 || m_queue.size() < m_capacity;
    });
    return enqueue(lock, &e);
}

template <typename E>
template <typename Rep, typename Period>
bool BlockingQueue<E>::offer(E* e, std::chrono::duration<Rep,Period> timeout) {
    std::unique_lock lock{m_lock};
    m_notFull.wait_for(lock, timeout, [this]() {
        return m_capacity == 0 || m_queue.size() < m_capacity;
    });
    return enqueue(lock, e);
}

template <typename E>
bool BlockingQueue<E>::take(E* p) {
    std::unique_lock lock{m_lock};
    m_notEmpty.wait(lock, [this]() {
        return m_capacity == 0 || !m_queue.empty();
    });
    return dequeue(lock, p);
}

template <typename E>
bool BlockingQueue<E>::poll(E* p) {
    std::unique_lock lock{m_lock};
    return dequeue(lock, p);
}

template <typename E>
template <typename Rep, typename Period>
bool BlockingQueue<E>::poll(E* p, std::chrono::duration<Rep, Period> timeout) {
    std::unique_lock lock{m_lock};
    m_notEmpty.wait_for(lock, timeout, [this]() {
        return m_capacity == 0 || !m_queue.empty();
    });
    return dequeue(lock, p);
}

template <typename E>
void BlockingQueue<E>::clear() {
    std::unique_lock lock(m_lock);
    while (!m_queue.empty())
        m_queue.pop();
    lock.unlock();
    m_notFull.notify_all();
}

template <typename E>
void BlockingQueue<E>::invalidate() {
    size_t cap = m_capacity;
    if (cap != 0 && m_capacity.compare_exchange_strong(cap, 0)) {
        m_notEmpty.notify_all();
        m_notFull.notify_all();
    }
}

template <typename E>
bool BlockingQueue<E>::valid() const noexcept {
    return m_capacity != 0;
}

//==-------------------------------------------------------------------------

// /*
//  * The Executor executes submitted tasks. This interface provides a way
//  * of decoupling task submission from the mechanics of how each task will
//  * be run, including details of thread use, scheduling, etc. An Executor
//  * is normally used instead of explicitly creating threads.
//  *
//  * However, the Executor interface does not strictly require that execution
//  * be asynchronous. In the simplest case, an executor can run the submitted
//  * task immediately in the caller's thread.
//  */
// class Executor {
// public:
//     /**
//      * Executes the given command at some time in the future. The command
//      * may execute in a new thread, in a pooled thread, or in the calling
//      * thread, at the discretion of the Executor implementation.
//      *
//      * @param command the runnable task
//      */
//     void execute(std::unique_ptr<Runnable> command);
//
//     /**
//      * Submits a value-returning task for execution and returns a future
//      * representing the pending results of the task. The future's get method
//      * will return the task's result upon successful completion.
//      *
//      * @param task the task function to submit
//      * @param args the task arguments to submit
//      * @return a future representing pending completion of the task
//      */
//     template <typename F, typename... Args>
//     auto submit(F&& task, Args&&... args) -> std::future<std::invoke_result_t<F,Args...)>>;
// };

/**
 * The Runnable interface should be implemented by any class where instances
 * are intended to be executed by a thread.
 */
struct Runnable {
    Runnable() = default;
    virtual ~Runnable() = default;
    Runnable(const Runnable&) = delete;
    Runnable& operator=(const Runnable&) = delete;
    Runnable(Runnable&&) = default;
    Runnable& operator=(Runnable&&) = default;

    virtual void run() = 0;

    void operator()() {
        run();
    }
};

/**
 * The PackagedTask class encapsulate any callable object and implements the
 * Runnable interface.
 */
template <typename Func, typename... Args>
class PackagedTask : public Runnable {
public:
    using ResultType = std::invoke_result_t<Func, Args...>;

    explicit PackagedTask(Func&& func, Args&&... args)
        : func(std::packaged_task<ResultType(Args...)>(std::move(func))),
          args(std::move(args)...) {}

    PackagedTask(PackagedTask&& t) noexcept
        : func(std::move(t.func)), args(std::move(t.args)) {}

    std::future<ResultType> get_future() {
        return func.get_future();
    }

    void run() override {
        execute(std::index_sequence_for<Args...>());
    }

private:
    std::packaged_task<ResultType(Args...)> func;
    std::tuple<Args...> args;

    template <size_t... I>
    void execute(std::index_sequence<I...>) {
        std::invoke(std::move(func), std::move(std::get<I>(args))...);
    }
};

/**
 * The thread pool executor executes each submitted task using one of possibly
 * several threads.
 */
class ThreadPoolExecutor {
    template <typename Rep, typename Period>
    std::chrono::nanoseconds toNano(std::chrono::duration<Rep, Period> d) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(d);
    }

    template <typename T>
    typename std::decay_t<T> decay_copy(T&& t) {
        return std::forward<T>(t);
    }

public:
    /**
     * Creates a new ThreadPoolExecutor with the given initial parameters.
     *
     * @param corePoolSize the number of threads to keep in the pool, even
     *        if they are idle, unless allowCoreThreadTimeOut is set
     * @param maximumPoolSize the maximum number of threads to allow in the
     *        pool
     * @param queueSize the number of tasks holding in a queue before they
     *        are executed.
     * @param keepAliveTime when the number of threads is greater than
     *        the core, this is the maximum time that excess idle threads
     *        will wait for new tasks before terminating.
     */
    template <typename Rep, typename Period>
    ThreadPoolExecutor(size_t corePoolSize, size_t maximumPoolSize, size_t queueSize,
                       std::chrono::duration<Rep, Period> keepAliveTime);

    /**
     * The destructor shutdown the thread pool.
     */
    ~ThreadPoolExecutor();

    /**
     * Executes the given task sometime in the future. The task may
     * execute in a new thread or in an existing pooled thread.
     *
     * @param command the task to execute
     */
    void execute(std::unique_ptr<Runnable> command);

    /**
     * Submits a value-returning task for execution and returns a std::future
     * representing the pending results of the task. The std::future's get
     * method will return the task's result upon successful completion.
     *
     * @param func callable object to call
     * @param args parameters to pass to func
     * @return std::future referring to the shared state
     */
    template <typename Func, typename... Args>
    auto submit(Func&& func, Args&&... args) {
        auto task = new PackagedTask(decay_copy(std::forward<Func>(func)),
                                     decay_copy(std::forward<Args>(args))...);
        auto result = task->get_future();
        execute(std::unique_ptr<Runnable>(task));
        return result;
    }

    /**
     * Initiates an orderly shutdown in which previously submitted
     * tasks are executed, but no new tasks will be accepted.
     * Invocation has no additional effect if already shutdown.
     *
     * This method does not wait for previously submitted tasks to
     * complete execution, Use awaitTermination to do that.
     */
    void shutdown();

    /**
     * Attempts to stop all actively executing tasks, halts the
     * processing of waiting tasks, and returns a list of the tasks
     * that were awaiting execution. These tasks are drained (removed)
     * from the task queue upon return from this method.
     *
     * This method does not wait for actively executing tasks to
     * terminate. Use awaitTermination to do that.
     */
    std::vector<std::unique_ptr<Runnable>> shutdownNow();

    /**
     * Returns true if this executor has been shut down.
     */
    bool isShutdown() const noexcept {
        return !isRunning(m_ctl);
    }

    /**
     * Returns true if all tasks have completed following shut down.
     */
    bool isTerminated() const noexcept {
        return m_ctl >= TERMINATED;
    }

    /**
     * Blocks until all tasks have completed execution after a shutdown
     * request, or the times occurs, whichever happens first
     *
     * @param timeout the maximum time to wait
     * @return true if this executor terminated and false if
     *         the timeout elapsed before termination
     */
    template <typename Rep, typename Period>
    bool awaitTermination(std::chrono::duration<Rep, Period> timeout) {
        return awaitTerminationNano(toNano(timeout));
    }

private:
    using ctl_t = uint32_t;

    static constexpr ctl_t COUNT_BITS = std::numeric_limits<ctl_t>::digits - 3;
    static constexpr ctl_t CAPACITY   = (1 << COUNT_BITS) - 1;

    /*
     * The runState provides the main lifecycle control, taking on values:
     *
     *   RUNNING:  Accept new tasks and process queued tasks
     *   SHUTDOWN: Don't accept new tasks, but process queued tasks
     *   STOP:     Don't accept new tasks, don't process queued tasks,
     *             and interrupt in-progress tasks
     *   TIDYING:  All tasks have terminated, workerCount is zero
     *   TERMINATED: terminated() has completed
     *
     * The numerical order among these values matters, to allow
     * ordered comparisons. The runState monotonically increases over
     * time, but need not hit each state. The transitions are:
     *
     * RUNNING -> SHUTDOWN
     *   On invocation of shutdown(), perhaps implicitly in destructor
     * (RUNNING or SHUTDOWN) -> STOP
     *   On invocation of shutdownNow()
     * SHUTDOWN -> TIDYING
     *   When both queue and pool are empty
     * STOP -> TIDYING
     *   When pool is empty
     * TIDYING -> TERMINATED
     *   When the terminated hook method has completed
     */
    static constexpr ctl_t RUNNING    = 0 << COUNT_BITS;
    static constexpr ctl_t SHUTDOWN   = 1 << COUNT_BITS;
    static constexpr ctl_t STOP       = 2 << COUNT_BITS;
    static constexpr ctl_t TIDYING    = 3 << COUNT_BITS;
    static constexpr ctl_t TERMINATED = 4 << COUNT_BITS;

    // Packing and unpacking ctl
    static constexpr ctl_t runStateOf(ctl_t c)       { return c & ~CAPACITY; }
    static constexpr ctl_t workerCountOf(ctl_t c)    { return c & CAPACITY; }
    static constexpr ctl_t ctlOf(ctl_t rs, ctl_t wc) { return rs | wc; }

    std::atomic<ctl_t> m_ctl{ctlOf(RUNNING, 0)};

    /*
     * Bit field accessors that don't require unpacking ctl.
     * These depend on the bit layout and on workerCount being never negative.
     */

    static constexpr bool isRunning(ctl_t c) {
        return c < SHUTDOWN;
    }

    /**
     * Attempts to CAS-increment the workerCount field of ctl.
     */
    bool compareAndIncrementWorkerCount(ctl_t expect) {
        return m_ctl.compare_exchange_strong(expect, expect + 1);
    }

    /**
     * Attempts to CAS-decrement the workerCount field of ctl.
     */
    bool compareAndDecrementWorkerCount(ctl_t expect) {
        assert(workerCountOf(expect) != 0);
        return m_ctl.compare_exchange_strong(expect, expect - 1);
    }

    /**
     * Decrements the workerCount field of ctl. This is called only on
     * abrupt termination of a thread (see onWorkerExit). Other
     * decrements are performed within getTask.
     */
    void decrementWorkerCount() {
        ctl_t c = m_ctl--;
        assert(workerCountOf(c) != 0);
    }

    void advanceRunState(ctl_t targetState);

private:
    BlockingQueue<std::unique_ptr<Runnable>> m_workQueue;
    std::mutex m_mainLock;
    std::condition_variable m_termination;

    size_t m_corePoolSize{0};
    size_t m_maximumPoolSize{0};
    std::chrono::nanoseconds m_keepAliveTime{0};
    bool m_allowCoreThreadTimeOut{false};

private:
    void tryTerminate();
    bool addWorker(std::unique_ptr<Runnable>* task, bool core);
    std::unique_ptr<Runnable> getTask();
    void runWorker(std::unique_ptr<Runnable> task);
    void onWorkerExit(bool completedAbruptly);
    bool awaitTerminationNano(std::chrono::nanoseconds timeout);
    void reject(std::unique_ptr<Runnable> task);
};

template <typename Rep, typename Period>
ThreadPoolExecutor::ThreadPoolExecutor(
        size_t corePoolSize, size_t maximumPoolSize, size_t queueSize,
        std::chrono::duration<Rep, Period> keepAliveTime)
    : m_corePoolSize{corePoolSize},
      m_maximumPoolSize{maximumPoolSize},
      m_workQueue{queueSize},
      m_keepAliveTime{toNano(keepAliveTime)}
{
    assert(maximumPoolSize > 0 && maximumPoolSize >= corePoolSize);
}

/**
 * Factory methods for Executor.
 */
namespace Executors {

/**
 * Creates a thread pool that reuses a fixed number of threads
 * operating off a shared unbounded queue. At any point, at most
 * nThreads threads will be active processing tasks. If additional
 * tasks are submitted when all threads are active, they will wait
 * in the queue until a thread is available. If any thread terminates
 * due to a failure during execution prior to shutdown, a new one
 * will take its place if needed to execute subsequent tasks. The
 * threads in the pool will exist until it is explicitly shutdown.
 *
 * @param nThreads the number of threads in the pool
 * @return the newly created thread pool
 */
inline ThreadPoolExecutor newFixedThreadPool(size_t nThreads) {
    return ThreadPoolExecutor(nThreads, nThreads, 0, std::chrono::nanoseconds(0));
}

/**
 * Creates a thread pool that creates new threads as needed, but
 * will reuse previously constructed threads when they are
 * available. These pools will typically improve the performance
 * of programs that execute may short-lived asynchronous tasks.
 * Calls to execute will reuse previously constructed threads
 * if available. If no existing thread is available, a new
 * thread will be created and added to the pool. Threads that
 * have not been used for sixty seconds are terminated and
 * removed from the cache. Thus a pool that remains idle for
 * long enough will not consume any resources.
 *
 * @return the newly created thread pool
 */
inline ThreadPoolExecutor newCachedThreadPool() {
    return ThreadPoolExecutor(0, std::numeric_limits<size_t>::max(), 1,
                              std::chrono::seconds(60));
}

/**
 * Returns a singleton thread pool.
 */
inline ThreadPoolExecutor& defaultThreadPool() {
    static ThreadPoolExecutor theDefault =
            newFixedThreadPool(std::max(std::thread::hardware_concurrency(), 2U) - 1);
    return theDefault;
}

} // namespace Executors

} // namespace kneron::concurrent

#endif //KNERON_CONCURRENT_H
