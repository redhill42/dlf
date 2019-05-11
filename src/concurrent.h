#ifndef KNERON_CONCURRENT_H
#define KNERON_CONCURRENT_H

#include <vector>
#include <queue>
#include <thread>
#include <future>
#include <mutex>

namespace kneron::concurrent {

// /**
//  * A blocking queue that supports operations that wait for the queue to become
//  * non-empty when retrieving an element, and wait for space to become available
//  * in the queue when storing an element.
//  *
//  * @tparam E the type of elements held in this queue
//  */
// template <typename E>
// class BlockingQueue {
// public:
//     /**
//      * Inserts the specified element into this queue, waiting if necessary
//      * for space to become available.
//      *
//      * @param e the element to add
//      * @return true if element is enqueued, false if the queue is invalid
//      */
//     bool put(E e);
//
//     /**
//      * Inserts the specified element into this queue if it is possible to do
//      * so immediately without violating capacity restrictions, returning
//      * true upon success and false if no space is currently available.
//      *
//      * @param e the element to add
//      * @return true if the element was added to this queue, else false
//      */
//     bool offer(E e);
//
//     /**
//      * Inserts the specified element into this queue, waiting up to the
//      * specified wait time if necessary for space to become available.
//      *
//      * @param e the element to add
//      * @param timeout how long to wait before giving up
//      */
//     template<typename Rep, typename Duration>
//     bool offer(E e, std::chrono::duration<Rep, Duration> timeout);
//
//     /**
//      * Retrieves and removes the head of this queue, waiting if necessary
//      * until an element becomes available.
//      *
//      * @param p the address to element to take, must not be null
//      * @return true if an element is dequeued, false if the queue is invalid
//      */
//     bool take(E *p);
//
//     /**
//      * Retrieves and removes the head of this queue if an element is available.
//      *
//      * @param p the address to element to poll, must not be null
//      * @return true if element is available, false otherwise
//      */
//     bool poll(E *p);
//
//     /**
//      * Retrieves and removes the head of this queue, waiting up to the
//      * specified wait time if necessary for an element to become available.
//      *
//      * @param p the address to element to poll, must not be null
//      * @param timeout how long to wait before giving up.
//      * @return true if element is available, false if the specified
//      * waiting time elapses before an element is available.
//      */
//     template<typename Rep, typename Duration>
//     bool poll(E *p, std::chrono::duration<Rep, Duration> timeout);
//
//     /**
//      * Returns true if this queue is empty, false otherwise.
//      */
//     bool empty() const noexcept;
//
//     /**
//      * Returns number elements available in the queue.
//      */
//     size_t size() const noexcept;
//
//     /**
//      * Returns the number of additional elements that this queue can ideally
//      * (in the absence of memory or resource constraints) accept without
//      * blocking.
//      */
//     size_t remaining() const noexcept;
//
//     /**
//      * Remove all elements from queue.
//      */
//     void clear();
//
//     /**
//      * Invalidate the queue. No more elements could be enqueued and all
//      * waiting thread will be notified.
//      */
//     void invalidate();
//
//     /**
//      * Returns true if this queue is valid, false othersie.
//      */
//     bool valid() const noexcept;
// };

//==-------------------------------------------------------------------------

/**
 * A blocking queue that supports operations that wait for the queue to become
 * non-empty when retrieving an element, and wait for space to become available
 * in the queue when storing an element.
 *
 * @tparam E the type of elements held in this queue
 */
template <typename E>
class BoundedBlockingQueue {
public:
    /**
     * Construct a blocking queue with given capacity.
     */
    explicit BoundedBlockingQueue(size_t capacity)
        : m_items(capacity) {}

    ~BoundedBlockingQueue() { invalidate(); clear(); }

    bool put(E e);
    bool offer(E e);
    template <typename Rep, typename Duration>
    bool offer(E e, std::chrono::duration<Rep, Duration> timeout);

    bool take(E* p);
    bool poll(E* p);
    template <typename Rep, typename Duration>
    bool poll(E* p, std::chrono::duration<Rep, Duration> timeout);

    bool empty() const noexcept;
    size_t size() const noexcept;
    size_t remaining() const noexcept;

    void clear();
    void invalidate();
    bool valid() const noexcept;

private:
    /** The queued items */
    std::vector<E> m_items;

    /** items index for next take, poll, peek or remove */
    size_t m_takeIndex = 0;

    /** items index for next put, offer, or add */
    size_t m_putIndex = 0;

    /** Number of elements in the queue */
    size_t m_count = 0;

    /** Main lock guarding all access */
    mutable std::mutex m_lock;

    /** Condition for waiting takes */
    std::condition_variable m_notEmpty;

    /** Condition for waiting puts */
    std::condition_variable m_notFull;

    /** Validation state. */
    bool m_valid = true;

    bool enqueue(std::unique_lock<std::mutex>& lock, E& x);
    bool dequeue(std::unique_lock<std::mutex>& lock, E* p);

    bool isEmpty() const noexcept { return m_count == 0; }
    bool isFull()  const noexcept { return m_count == m_items.size(); }
    bool isValid() const noexcept { return m_valid; }
};

template <typename E>
bool BoundedBlockingQueue<E>::enqueue(std::unique_lock<std::mutex>& lock, E& x) {
    m_items[m_putIndex] = std::move(x);
    if (++m_putIndex == m_items.size())
        m_putIndex = 0;
    m_count++;

    lock.unlock();
    m_notEmpty.notify_one();
    return true;
}

template <typename E>
bool BoundedBlockingQueue<E>::dequeue(std::unique_lock<std::mutex>& lock, E* p) {
    *p = std::move(m_items[m_takeIndex]);
    if (++m_takeIndex == m_items.size())
        m_takeIndex = 0;
    m_count--;

    lock.unlock();
    m_notFull.notify_one();
    return true;
}

template <typename E>
bool BoundedBlockingQueue<E>::put(E e) {
    std::unique_lock lock(m_lock);
    m_notFull.wait(lock, [this]() {
        return !isValid() || !isFull();
    });
    return isValid() && enqueue(lock, e);
}

template <typename E>
bool BoundedBlockingQueue<E>::offer(E e) {
    std::unique_lock lock(m_lock);
    return isValid() && !isFull() && enqueue(lock, e);
}

template <typename E>
template <typename Rep, typename Duration>
bool BoundedBlockingQueue<E>::offer(E e, std::chrono::duration<Rep,Duration> timeout) {
    std::unique_lock lock(m_lock);
    if (!m_notFull.wait_for(lock, timeout, [this]() {
        return !isValid() || !isFull();
    })) return false;
    return isValid() && enqueue(lock, e);
}

template <typename E>
bool BoundedBlockingQueue<E>::take(E* p) {
    std::unique_lock lock(m_lock);
    m_notEmpty.wait(lock, [this]() {
        return !isValid() || !isEmpty();
    });
    return !isEmpty() && dequeue(lock, p);
}

template <typename E>
bool BoundedBlockingQueue<E>::poll(E* p) {
    std::unique_lock lock(m_lock);
    return !isEmpty() && dequeue(lock, p);
}

template <typename E>
template <typename Rep, typename Duration>
bool BoundedBlockingQueue<E>::poll(E* p, std::chrono::duration<Rep, Duration> timeout) {
    std::unique_lock lock(m_lock);
    if (!m_notEmpty.wait_for(lock, timeout, [this]() {
        return !isValid() || !isEmpty();
    })) return false;
    return !isEmpty() && dequeue(lock, p);
}

template <typename E>
inline bool BoundedBlockingQueue<E>::empty() const noexcept {
    std::unique_lock lock(m_lock);
    return isEmpty();
}

template <typename E>
inline size_t BoundedBlockingQueue<E>::size() const noexcept {
    std::unique_lock lock(m_lock);
    return m_count;
}

template <typename E>
inline size_t BoundedBlockingQueue<E>::remaining() const noexcept {
    std::unique_lock lock(m_lock);
    return m_items.size() - m_count;
}

template <typename E>
void BoundedBlockingQueue<E>::clear() {
    std::unique_lock lock(m_lock);
    m_items.clear();
    lock.unlock();
    m_notFull.notify_all();
}

template <typename E>
void BoundedBlockingQueue<E>::invalidate() {
    std::unique_lock lock(m_lock);
    if (m_valid) {
        m_valid = false;
        m_notEmpty.notify_all();
        m_notFull.notify_all();
    }
}

template <typename E>
inline bool BoundedBlockingQueue<E>::valid() const noexcept {
    std::unique_lock lock(m_lock);
    return m_valid;
}

//==-------------------------------------------------------------------------

/**
 * A unbounded blocking queue. Unbounded queues typically have higher throughput
 * than bounded queues but less predictable performance in most concurrent
 * applications.
 *
 * @tparam E the type of elements held in this queue
 */
template <typename E>
class UnboundedBlockingQueue {
public:
    ~UnboundedBlockingQueue() { invalidate(); clear(); }

    bool put(E e);
    bool offer(E e);
    template <typename Rep, typename Duration>
    bool offer(E e, std::chrono::duration<Rep, Duration> timeout);

    bool take(E* p);
    bool poll(E* p);
    template <typename Rep, typename Duration>
    bool poll(E* p, std::chrono::duration<Rep, Duration> timeout);

    bool empty() const noexcept;
    size_t size() const noexcept;
    size_t remaining() const noexcept;

    void clear();
    void invalidate();
    bool valid() const noexcept;

private:
    /** The underlying queue */
    std::queue<E> m_queue;

    /** Main lock guarding all access */
    mutable std::mutex m_lock;

    /** Condition for waiting takes */
    std::condition_variable m_notEmpty;

    /** Validation state */
    bool m_valid = true;

    bool isValid() { return m_valid; }

    bool enqueue(E& e);
    bool dequeue(E* p);
};

template <typename E>
bool UnboundedBlockingQueue<E>::enqueue(E& e) {
    std::unique_lock lock(m_lock);
    if (isValid()) {
        m_queue.push(std::move(e));
        lock.unlock();
        m_notEmpty.notify_one();
        return true;
    } else {
        return false;
    }
}

template <typename E>
bool UnboundedBlockingQueue<E>::dequeue(E* p) {
    if (!m_queue.empty()) {
        *p = std::move(m_queue.front());
        m_queue.pop();
        return true;
    } else {
        return false;
    }
}

template <typename E>
bool UnboundedBlockingQueue<E>::put(E e) {
    return enqueue(e);
}

template <typename E>
inline bool UnboundedBlockingQueue<E>::offer(E e) {
    return enqueue(e);
}

template <typename E>
template <typename Rep, typename Duration>
inline bool UnboundedBlockingQueue<E>::offer(E e, std::chrono::duration<Rep, Duration>) {
    return enqueue(e);
}

template <typename E>
bool UnboundedBlockingQueue<E>::take(E* p) {
    std::unique_lock lock(m_lock);
    m_notEmpty.wait(lock, [this]() {
        return !isValid() || !m_queue.empty();
    });
    return dequeue(p);
}

template <typename E>
bool UnboundedBlockingQueue<E>::poll(E* p) {
    std::unique_lock lock(m_lock);
    return dequeue(p);
}

template <typename E>
template <typename Rep, typename Duration>
bool UnboundedBlockingQueue<E>::poll(E* p, std::chrono::duration<Rep, Duration> timeout) {
    std::unique_lock lock(m_lock);
    if (!m_notEmpty.wait_for(lock, timeout, [this]() {
        return !isValid() || !m_queue.empty();
    })) return false;
    return dequeue(p);
}

template <typename E>
inline bool UnboundedBlockingQueue<E>::empty() const noexcept {
    std::unique_lock lock(m_lock);
    return m_queue.empty();
}

template <typename E>
inline size_t UnboundedBlockingQueue<E>::size() const noexcept {
    std::unique_lock lock(m_lock);
    return m_queue.size();
}

template <typename E>
inline size_t UnboundedBlockingQueue<E>::remaining() const noexcept {
    std::unique_lock lock(m_lock);
    return std::numeric_limits<size_t>::max();
}

template <typename E>
void UnboundedBlockingQueue<E>::clear() {
    std::unique_lock lock(m_lock);
    while (!m_queue.empty()) {
        m_queue.pop();
    }
}

template <typename E>
void UnboundedBlockingQueue<E>::invalidate() {
    std::unique_lock lock(m_lock);
    if (m_valid) {
        m_valid = false;
        m_notEmpty.notify_all();
    }
}

template <typename E>
inline bool UnboundedBlockingQueue<E>::valid() const noexcept {
    std::unique_lock lock(m_lock);
    return m_valid;
}

//==-------------------------------------------------------------------------

} // namespace kneron::concurrent

#endif //KNERON_CONCURRENT_H
