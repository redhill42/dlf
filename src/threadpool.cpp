#include <thread>
#include <mutex>
#include <vector>
#include <stdexcept>
#include <utility>

#include "concurrent.h"

using namespace kneron::concurrent;

template <typename Func>
struct AtEnd {
    const Func& block;
    explicit AtEnd(const Func& func) : block(func) {}
    ~AtEnd() { block(); }
};

void ThreadPoolExecutor::advanceRunState(ctl_t targetState) {
    ctl_t c = m_ctl.load(std::memory_order_relaxed);
    for (;;) {
        if (c >= targetState)
            break;
        if (m_ctl.compare_exchange_weak(
                c, ctlOf(targetState, workerCountOf(c)),
                std::memory_order_release, std::memory_order_relaxed))
            break;
    }
}

/**
 * Transitions to TERMINATED state if either (SHUTDOWN and pool
 * and queue empty) or (STOP and pool empty). This method must
 * be called following any action that might make termination
 * possible -- reducing worker count or removing tasks from
 * the queue during shutdown.
 */
void ThreadPoolExecutor::tryTerminate() {
    ctl_t c = m_ctl.load(std::memory_order_relaxed);
    for (;;) {
        if (isRunning(c) || (c >= TIDYING) ||
            (runStateOf(c) == SHUTDOWN && !m_workQueue.empty()))
            return;

        if (workerCountOf(c) != 0) {
            m_workQueue.invalidate();
            return;
        }

        if (m_ctl.compare_exchange_weak(c, ctlOf(TIDYING, 0),
                                        std::memory_order_release,
                                        std::memory_order_relaxed)) {
            std::unique_lock lock(m_mainLock);
            m_ctl = ctlOf(TERMINATED, 0);
            m_termination.notify_all();
            return;
        }
    }
}

bool ThreadPoolExecutor::addWorker(std::unique_ptr<Runnable>* task, bool core) {
    bool retry = true;
    while (retry) {
        ctl_t c = m_ctl;
        ctl_t rs = runStateOf(c);

        // don't add worker if shutting down unless work queue is not empty
        if (rs >= SHUTDOWN && !(rs == SHUTDOWN && task == nullptr && !m_workQueue.empty()))
            return false;

        for (;;) {
            ctl_t wc = workerCountOf(c);
            if (wc >= CAPACITY || wc >= (core ? m_corePoolSize : m_maximumPoolSize))
                return false;
            if (compareAndIncrementWorkerCount(c)) {
                retry = false;
                break;
            }
            c = m_ctl; // re-read ctl
            if (runStateOf(c) != rs)
                break;
            // else CAS failed due to workerCount change; retry inner loop
        }
    }

    try {
        std::thread(
            &ThreadPoolExecutor::runWorker, this,
            task ? std::move(*task) : nullptr
        ).detach();
    } catch (...) {
        decrementWorkerCount();
        tryTerminate();
        throw;
    }

    return true;
}

std::unique_ptr<Runnable> ThreadPoolExecutor::getTask() {
    bool timedOut = false;

    for (;;) {
        ctl_t c = m_ctl;
        ctl_t rs = runStateOf(c);

        // check if queue empty only if necessary
        if (rs >= STOP || (rs >= SHUTDOWN && m_workQueue.empty())) {
            decrementWorkerCount();
            return nullptr;
        }

        ctl_t wc = workerCountOf(c);

        // Are workers subject to culling?
        bool timed = m_allowCoreThreadTimeOut || wc > m_corePoolSize;

        if ((wc > m_maximumPoolSize || (timed && timedOut))
            && (wc > 1 || m_workQueue.empty())) {
            if (compareAndDecrementWorkerCount(c))
                return nullptr;
            continue;
        }

        std::unique_ptr<Runnable> task;
        if (timed ? m_workQueue.poll(&task, std::chrono::nanoseconds(m_keepAliveTime))
                  : m_workQueue.take(&task))
            return task;
        timedOut = m_workQueue.valid();
    }
}

void ThreadPoolExecutor::runWorker(std::unique_ptr<Runnable> task) {
    bool completedAbruptly = true;

    AtEnd atThreadExit([&]() {
        onWorkerExit(completedAbruptly);
    });

    while (task != nullptr || (task = getTask()) != nullptr) {
        task->run();
        task = nullptr;
    }

    completedAbruptly = false;
}

void ThreadPoolExecutor::onWorkerExit(bool completedAbruptly) {
    if (completedAbruptly) // If abrupt, then workerCount wasn't adjusted
        decrementWorkerCount();

    tryTerminate();

    ctl_t c = m_ctl;
    if (c < STOP) {
        if (!completedAbruptly) {
            size_t min = m_allowCoreThreadTimeOut ? 0 : m_corePoolSize;
            if (min == 0 && !m_workQueue.empty())
                min = 1;
            if (workerCountOf(c) >= min)
                return;
        }
        addWorker(nullptr, false);
    }
}

void ThreadPoolExecutor::execute(std::unique_ptr<Runnable> command) {
    assert(command != nullptr);
    if (workerCountOf(m_ctl) < m_corePoolSize) {
        if (addWorker(&command, true))
            return;
    }
    if (isRunning(m_ctl) && m_workQueue.offer(&command)) {
        if (workerCountOf(m_ctl) == 0)
            addWorker(nullptr, false);
    } else if (!addWorker(&command, false)) {
        reject(std::move(command));
    }
}

void ThreadPoolExecutor::shutdown() {
    advanceRunState(SHUTDOWN);
    tryTerminate();
}

std::vector<std::unique_ptr<Runnable>> ThreadPoolExecutor::shutdownNow() {
    std::vector<std::unique_ptr<Runnable>> result;
    std::unique_ptr<Runnable> task;

    advanceRunState(STOP);
    result.reserve(m_workQueue.size());
    while (m_workQueue.poll(&task))
        result.push_back(std::move(task));
    tryTerminate();
    return result;
}

bool ThreadPoolExecutor::awaitTerminationNano(std::chrono::nanoseconds timeout) {
    std::unique_lock lock(m_mainLock);
    for (;;) {
        if (m_ctl >= TERMINATED) {
            // FIXME: wait for worker threads fully terminated
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return true;
        }
        if (timeout.count() == 0)
            return false;
        if (m_termination.wait_for(lock, timeout) == std::cv_status::timeout)
            return false;
    }
}

void ThreadPoolExecutor::reject(std::unique_ptr<Runnable> task) {
    // Default implementation is run task on caller's thread
    task->run();
}

ThreadPoolExecutor::~ThreadPoolExecutor() {
    shutdown();
    awaitTermination(std::chrono::seconds(30));
}
