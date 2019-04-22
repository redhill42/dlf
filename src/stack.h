#ifndef _STACK_H
#define _STACK_H

#include <stdexcept>

class StackOverflow : public std::runtime_error
{
public:
    StackOverflow() : std::runtime_error("stack overflow") {}
};

class StackUnderflow : public std::runtime_error
{
public:
    StackUnderflow() : std::runtime_error("stack underflow") {}
};

template <typename T, std::size_t N = 10>
class Stack {
public:
    Stack() : top(N) {}

    void push(T t) {
        if (top > 0) {
            data[--top] = t;
        } else {
            throw StackOverflow();
        }
    }

    T pop() {
        if (top == N) {
            throw StackUnderflow();
        } else {
            return data[top++];
        }
    }

    T peek() const {
        if (top == N) {
            throw StackUnderflow();
        } else {
            return data[top];
        }
    }

    std::size_t size() const noexcept {
        return N - top;
    }

private:
    T data[N];
    std::size_t top;
};

#endif //_STACK_H
