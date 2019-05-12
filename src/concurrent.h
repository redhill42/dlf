#ifndef KNERON_CONCURRENT_H
#define KNERON_CONCURRENT_H

#include <queue>
#include <thread>
#include <future>
#include <mutex>

namespace kneron::concurrent {

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

    /**
     * Inserts the specified element into this queue if it is possible to do
     * so immediately without violating capacity restrictions, returning
     * true upon success and false if no space is currently available.
     *
     * @param e the element to add
     * @return true if the element was added to this queue, else false
     */
    bool offer(E e);

    /**
     * Inserts the specified element into this queue, waiting up to the
     * specified wait time if necessary for space to become available.
     *
     * @param e the element to add
     * @param timeout how long to wait before giving up
     */
    template<typename Rep, typename Duration>
    bool offer(E e, std::chrono::duration<Rep, Duration> timeout);

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
    template<typename Rep, typename Duration>
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

private:
    /** The underlying queue */
    std::queue<E> m_queue;

    /** The capacity bound, or maximum size_t value if none */
    size_t m_capacity;

    /** Main lock guarding all access */
    mutable std::mutex m_lock;

    /** Wait queue for waiting takes */
    std::condition_variable m_notEmpty;

    /** Wait queue for waiting puts */
    std::condition_variable m_notFull;

    bool enqueue(std::unique_lock<std::mutex>& lock, E&& e);
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
bool BlockingQueue<E>::enqueue(std::unique_lock<std::mutex>& lock, E&& e) {
    if (m_queue.size() < m_capacity) {
        m_queue.push(std::move(e));
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
    return enqueue(lock, std::move(e));
}

template <typename E>
bool BlockingQueue<E>::offer(E e) {
    std::unique_lock lock{m_lock};
    return enqueue(lock, std::move(e));
}

template <typename E>
template <typename Rep, typename Period>
bool BlockingQueue<E>::offer(E e, std::chrono::duration<Rep,Period> timeout) {
    std::unique_lock lock{m_lock};
    m_notFull.wait_for(lock, timeout, [this]() {
        return m_capacity == 0 || m_queue.size() < m_capacity;
    });
    return enqueue(lock, std::move(e));
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
    std::unique_lock lock(m_lock);
    if (m_capacity != 0) {
        m_capacity = 0;
        lock.unlock();
        m_notEmpty.notify_all();
        m_notFull.notify_all();
    }
}

} // namespace kneron::concurrent

#endif //KNERON_CONCURRENT_H
