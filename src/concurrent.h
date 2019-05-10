#ifndef KNERON_CONCURRENT_H
#define KNERON_CONCURRENT_H

#include <vector>
#include <thread>
#include <future>
#include <optional>

namespace kneron::concurrent {

//==-------------------------------------------------------------------------

/**
 * A blocking queue that supports operations that wait for the queue to become
 * non-empty when retrieving an element, and wait for space to become available
 * in the queue when storing an element.
 */
template <typename E>
class BoundedBlockingQueue {
public:
    /**
     * Construct a blocking queue with given capacity.
     */
    explicit BoundedBlockingQueue(size_t capacity)
        : m_items(capacity) {}

    /**
     * Inserts the specified element into this queue, waiting if necessary
     * for space to become available.
     *
     * @param e the element to add
     */
    void put(E e);

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
    template <typename Rep, typename Duration>
    bool offer(E e, std::chrono::duration<Rep, Duration> timeout);

    /**
     * Retrieves and removes the head of this queue, waiting if necessary
     * until an element becomes available.
     *
     * @param p the address to element to take, must not be null
     */
    void take(E* p);

    /**
     * Retrieves and removes the head of this queue if an element is available.
     *
     * @param p the address to element to poll, must not be null
     * @return true if element is available, false otherwise
     */
    bool poll(E* p);

    /**
     * Retrieves and removes the head of this queue, waiting up to the
     * specified wait time if necessary for an element to become available.
     *
     * @param p the address to element to poll, must not be null
     * @param timeout how long to wait before giving up.
     * @return true if element is available, false if the specified
     * waiting time elapses before an element is available.
     */
    template <typename Rep, typename Duration>
    bool poll(E* p, std::chrono::duration<Rep, Duration> timeout);

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

    void enqueue(E& x);
    void dequeue(E* p);

    bool isEmpty() { return m_count == 0; }
    bool isFull()  { return m_count == m_items.size(); }
};

template <typename E>
void BoundedBlockingQueue<E>::enqueue(E& x) {
    m_items[m_putIndex] = std::move(x);
    if (++m_putIndex == m_items.size())
        m_putIndex = 0;
    m_count++;
}

template <typename E>
void BoundedBlockingQueue<E>::dequeue(E* p) {
    *p = std::move(m_items[m_takeIndex]);
    if (++m_takeIndex == m_items.size())
        m_takeIndex = 0;
    m_count--;
}

template <typename E>
void BoundedBlockingQueue<E>::put(E e) {
    {
        std::unique_lock lock(m_lock);
        while (isFull())
            m_notFull.wait(lock);
        enqueue(e);
    }
    m_notEmpty.notify_one();
}

template <typename E>
bool BoundedBlockingQueue<E>::offer(E e) {
    {
        std::unique_lock lock(m_lock);
        if (isFull())
            return false;
        enqueue(e);
    }
    m_notEmpty.notify_one();
    return true;
}

template <typename E>
template <typename Rep, typename Duration>
bool BoundedBlockingQueue<E>::offer(E e, std::chrono::duration<Rep,Duration> timeout) {
    {
        std::unique_lock lock(m_lock);
        if (isFull() && !m_notFull.wait_for(lock, timeout, [this](){ return !isFull(); }))
            return false;
        enqueue(e);
    }
    m_notEmpty.notify_one();
    return true;
}

template <typename E>
void BoundedBlockingQueue<E>::take(E* p) {
    {
        std::unique_lock lock(m_lock);
        while (isEmpty())
            m_notEmpty.wait(lock);
        dequeue(p);
    }
    m_notFull.notify_one();
}

template <typename E>
bool BoundedBlockingQueue<E>::poll(E* p) {
    {
        std::unique_lock lock(m_lock);
        if (isEmpty())
            return false;
        dequeue(p);
    }
    m_notFull.notify_one();
    return true;
}

template <typename E>
template <typename Rep, typename Duration>
bool BoundedBlockingQueue<E>::poll(E* p, std::chrono::duration<Rep, Duration> timeout) {
    {
        std::unique_lock lock(m_lock);
        if (isEmpty() && !m_notEmpty.wait_for(lock, timeout, [this](){ return !isEmpty(); }))
            return false;
        dequeue(p);
    }
    m_notFull.notify_one();
    return true;
}

template <typename E>
bool BoundedBlockingQueue<E>::empty() const noexcept {
    std::unique_lock lock(m_lock);
    return isEmpty();
}

template <typename E>
size_t BoundedBlockingQueue<E>::size() const noexcept {
    std::unique_lock lock(m_lock);
    return m_count;
}

template <typename E>
size_t BoundedBlockingQueue<E>::remaining() const noexcept {
    std::unique_lock lock(m_lock);
    return m_items.size() - m_count;
}

//==-------------------------------------------------------------------------

} // namespace kneron::concurrent

#endif //KNERON_CONCURRENT_H
