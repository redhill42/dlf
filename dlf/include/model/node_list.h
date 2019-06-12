#pragma once

#include <cstddef>
#include <cassert>

namespace dlf { namespace model {
/*
 * Intrusive doubly linked lists with sane reverse iterators.
 *
 * The templated type T must support a few operations:
 *
 *  - It must have a field: T* next_in_graph[2] = { nullptr, nullptr };
 *    which are used for the intrusive linked list pointers.
 *
 *  - It must have a method 'destroy()', which removes T from the
 *    list and frees a T.
 *
 * In practice, we are only using it with Node and const Node. 'destroy()'
 * needs to be renegotiated if you want to use this somewhere else.
 *
 * Besides the benefits of being intrusive, unlink std::list, these lists
 * handle forward and backward iteration uniformly because we require a
 * "before-first-element" sentinel. This means that reverse iterators
 * physically point to the element they logically point to, rather than
 * the off-by-one behavior for all standard library reverse iterators.
 */

static constexpr size_t kNextDirection = 0;
static constexpr size_t kPrevDirection = 1;

template <typename T>
class GenericNodeListIterator {
    T* cur;
    size_t dir; // direction, 0 is forward, 1 is reverse

    size_t reverseDir() {
        return dir == kNextDirection ? kPrevDirection : kNextDirection;
    }

public:
    GenericNodeListIterator() : cur(nullptr), dir(kNextDirection) {}
    GenericNodeListIterator(T* cur, size_t dir) : cur(cur), dir(dir) {}
    GenericNodeListIterator(const GenericNodeListIterator& rhs)
        : cur(rhs.cur), dir(rhs.dir) {}

    T* operator*() const noexcept { return cur; }
    T* operator->() const noexcept { return cur; }

    GenericNodeListIterator& operator++() {
        assert(cur);
        cur = cur->next_in_graph[dir];
        return *this;
    }

    GenericNodeListIterator operator++(int) {
        auto old = *this;
        ++(*this);
        return old;
    }

    GenericNodeListIterator& operator--() {
        assert(cur);
        cur = cur->next_in_graph[reverseDir()];
        return *this;
    }

    GenericNodeListIterator operator--(int) {
        auto old = *this;
        --(*this);
        return old;
    }

    /**
     * Erase cur without invalidating this iterator. Named differently
     * from destroy so that ->/. bugs do not silently cause the wrong
     * one to be called. Iterator will point to the previous entry after
     * call.
     */
    void destroyCurrent() {
        T* n = cur;
        cur = cur->next_in_graph[reverseDir()];
        n->destroy();
    }

    GenericNodeListIterator reverse() {
        return GenericNodeListIterator(cur, reverseDir());
    }
};

template <typename T>
class GenericNodeList {
    T* head;
    size_t dir;

public:
    using iterator = GenericNodeListIterator<T>;
    using const_iterator = GenericNodeListIterator<const T>;

    GenericNodeList(T* head, size_t dir)
        : head(head), dir(dir) {}

    iterator begin() noexcept {
        return {head->next_in_graph[dir], dir};
    }

    const_iterator begin() const noexcept {
        return {head->next_in_graph[dir], dir};
    }

    iterator end() noexcept {
        return {head, dir};
    }

    const_iterator end() const noexcept {
        return {head, dir};
    }

    iterator rbegin() noexcept {
        return reverse().begin();
    }

    const_iterator rbegin() const noexcept {
        return reverse().begin();
    }

    iterator rend() noexcept {
        return reverse().end();
    }

    const_iterator rend() const noexcept {
        return reverse().end();
    }

    GenericNodeList reverse() noexcept {
        return {head, dir == kNextDirection ? kPrevDirection : kNextDirection};
    }

    GenericNodeList reverse() const noexcept {
        return {head, dir == kNextDirection ? kPrevDirection : kNextDirection};
    }
};

template <typename T>
static inline bool operator==(GenericNodeListIterator<T> a, GenericNodeListIterator<T> b) {
    return *a == *b;
}

template <typename T>
static inline bool operator!=(GenericNodeListIterator<T> a, GenericNodeListIterator<T> b) {
    return *a != *b;
}

}} // namespace dlf::model

namespace std {

template <typename T>
struct iterator_traits<dlf::model::GenericNodeListIterator<T>> {
    using difference_type = size_t;
    using value_type = T*;
    using pointer = T**;
    using reference = T*&;
    using iterator_category = bidirectional_iterator_tag;
};

} // namespace std
