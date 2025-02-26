#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <array>
#include <iterator>
#include <type_traits>
#include <cassert>
#include <sstream>

#ifndef CPP_VER
#  if __cplusplus <= 201103L
#    define CPP_VER 11
#  elif __cplusplus <= 201402L
#    define CPP_VER 14
#  elif __cplusplus <= 201703L
#    define CPP_VER 17
#  else
#    define CPP_VER 20
#  endif
#endif // !CPP_VER

#if CPP_VER < 14
#  error C++14 is the minimal requirement
#endif

#ifndef GRAINSIZE
#  ifdef NDEBUG
#    define GRAINSIZE 1024
#  else
#    define GRAINSIZE 4
#  endif
#endif

// C++17 back ports
namespace cxx {

#if CPP_VER >= 17
using std::conjunction;
using std::disjunction;
using std::negation;
#else
template <class...> struct conjunction;
template <> struct conjunction<> : std::true_type {};
template <class B0> struct conjunction<B0> : B0 {};
template <class B0, class B1>
struct conjunction<B0, B1> : std::conditional<B0::value, B1, B0>::type {};
template <class B0, class B1, class B2, class... Bn>
struct conjunction<B0, B1, B2, Bn...>
    : std::conditional<B0::value, conjunction<B1, B2, Bn...>, B0>::type {};

template <class...> struct disjunction;
template <> struct disjunction<> : std::false_type {};
template <class B0> struct disjunction<B0> : B0 {};
template <class B0, class B1>
struct disjunction<B0, B1> : std::conditional<B0::value, B0, B1>::type {};
template <class B0, class B1, class B2, class... Bn>
struct disjunction<B0, B1, B2, Bn...>
    : std::conditional<B0::value, B0, disjunction<B1, B2, Bn...>>::type {};

template <class T>
struct negation : std::conditional<T::value, std::false_type, std::true_type>::type {};
#endif

#if CPP_VER >= 17
using std::invoke_result;
using std::invoke_result_t;
#else
template <typename Fn, typename... Args>
using invoke_result = std::result_of<Fn(Args...)>;
template <typename Fn, typename... Args>
using invoke_result_t = std::result_of_t<Fn(Args...)>;
#endif

#if CPP_VER >= 17
using std::clamp;
#else
template <class T, class Compare>
inline constexpr const T& clamp(const T& v, const T& lo, const T& hi, Compare comp) {
    return comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}

template <class T>
inline constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
    return cxx::clamp(v, lo, hi, std::less<T>());
}
#endif

#if CPP_VER >= 17
using std::size;
#else
template <typename Cont>
inline constexpr auto size(const Cont& c)
noexcept(noexcept(c.size()))
-> decltype      (c.size())
{ return          c.size(); }

template <typename T, size_t Sz>
inline constexpr size_t size(const T (&)[Sz]) noexcept { return Sz; }
#endif

template <typename Predicate>
inline constexpr bool all_of(Predicate f) {
    return true;
}

template <typename Predicate, typename T, typename... Rest>
inline constexpr bool all_of(Predicate f, T&& x, Rest&&... xs) {
    return f(std::forward<T>(x)) && all_of(f, std::forward<Rest>(xs)...);
}

template <typename Predicate>
inline constexpr bool none_of(Predicate f) {
    return true;
}

template <typename Predicate, typename T, typename... Rest>
inline constexpr bool none_of(Predicate f, T&& x, Rest&&... xs) {
    return !f(std::forward<T>(x)) && none_of(f, std::forward<Rest>(xs)...);
}

template <typename Predicate>
inline constexpr bool any_of(Predicate f) {
    return false;
}

template <typename Predicate, typename T, typename... Rest>
inline constexpr bool any_of(Predicate f, T&& x, Rest&&... xs) {
    return f(std::forward<T>(x)) || any_of(f, std::forward<Rest>(xs)...);
}

template <class T, class Compare>
inline constexpr T max(std::initializer_list<T> t, Compare comp) {
    return *std::max_element(t.begin(), t.end(), comp);
}

template <class T>
inline constexpr T max(std::initializer_list<T> t) {
    return *std::max_element(t.begin(), t.end(), std::less<T>());
}

template <typename InputIterator, typename T, typename BinaryOp, typename UnaryOp>
inline T transform_reduce(InputIterator first, InputIterator last,
                          T init, BinaryOp b, UnaryOp u)
{
    for (; first != last; ++first)
        init = b(init, u(*first));
    return init;
}

template <typename InputIterator1, typename InputIterator2,
          typename T, typename BinaryOp1, typename BinaryOp2>
inline T transform_reduce(InputIterator1 first1, InputIterator1 last1,
                          InputIterator2 first2, T init, BinaryOp1 b1, BinaryOp2 b2)
{
    for (; first1 != last1; ++first1, (void) ++first2)
        init = b1(init, b2(*first1, *first2));
    return init;
}

template <typename InputIterator1, typename InputIterator2, typename T>
inline T transform_reduce(InputIterator1 first1, InputIterator1 last1,
                          InputIterator2 first2, T init)
{
    return transform_reduce(first1, last1, first2, std::move(init),
                            std::plus<>(), std::multiplies<>());
}

#if CPP_VER >= 17
using std::string_view;
#else
using string_view = std::string;
#endif

template <typename T>
void copy(int n, const T* px, const int x_inc, T* py, const int y_inc) {
    if (x_inc == 1 && y_inc == 1) {
        std::copy(px, px + n, py);
    } else {
        for (; n > 0; --n, px += x_inc, py += y_inc)
            *py = *px;
    }
}

template <typename T>
void move(int n, T* px, const int x_inc, T* py, const int y_inc) {
    if (x_inc == 1 && y_inc == 1) {
        std::move(px, px + n, py);
    } else {
        for (; n > 0; --n, px += x_inc, py += y_inc)
            *py = std::move(*px);
    }
}

template <typename T, typename Compare>
constexpr int lower_bound(int n, const T* data, const int stride, const T& value, Compare comp) {
    int i = 0;
    while (n != 0) {
        int m = n / 2;
        if (comp(data[(m + i) * stride], value)) {
            i += m + 1;
            n -= m + 1;
        } else {
            n = m;
        }
    }
    return i;
}

template <typename T, typename Compare>
constexpr int upper_bound(int n, const T* data, const int stride, const T& value, Compare comp) {
    int i = 0;
    while (n != 0) {
        int m = n / 2;
        if (comp(value, data[(m + i) * stride]))
            n = m;
        else {
            i += m + 1;
            n -= m + 1;
        }
    }
    return i;
}

} // namespace cxx

namespace cxx {

/**
 * array_ref - Represent a constant reference to an array (0 or more elements
 * consecutively in memory), i.e. a start pointer and a length. It allows
 * various APIs to take consecutive elements easily and conveniently.
 *
 * This class does not own the underlying data, it is expected to be used in
 * situations where the data resides in some other buffer, whose lifetime
 * extends past that of the array_ref. For this reason, it is not in general
 * safe to store an array_ref.
 *
 * This is intended to be trivially copyable, so it should be passed by
 * value.
 */
template <typename T>
class array_ref {
public:
    using iterator = const T*;
    using const_iterator = const T*;
    using size_type = size_t;
    using reverse_iterator = std::reverse_iterator<iterator>;

private:
    // The start of the array, in an external buffer.
    const T* m_data;

    // The number of elements.
    size_type m_length;

public:
    /**
     * Construct an empty array_ref.
     */
    array_ref() : m_data(nullptr), m_length(0) {}

    /**
     * Construct an array_ref from a single element
     */
    explicit array_ref(const T& elt) : m_data(&elt), m_length(1) {}

    /**
     * Construct an array_ref from a pointer and length.
     */
    explicit array_ref(const T* data, size_t length)
        : m_data(data), m_length(length) {}

    /**
     * Construct an array_ref from a range.
     */
    explicit array_ref(const T* begin, const T* end)
        : m_data(begin), m_length(end - begin) {}

    /**
     * Construct an array_ref from a std::vector.
     */
    template <typename A>
    /*implicit*/ array_ref(const std::vector<T, A>& vec)
        : m_data(vec.data()), m_length(vec.size()) {}

    /**
     * Construct an array_ref from a C array.
     */
    template <size_t N>
    /*implicit*/ constexpr array_ref(const std::array<T, N>& arr)
        : m_data(arr.data()), m_length(N) {}

    /**
     * Construct an array_ref from a C array.
     */
    template <size_t N>
    /*implicit*/ constexpr array_ref(const T (&arr)[N])
        : m_data(arr), m_length(N) {}

    /**
     * Construct an array_ref from a std::initializer_list.
     */
    /*implicit*/ array_ref(const std::initializer_list<T>& vec)
        : m_data(vec.begin() == vec.end() ? nullptr : vec.begin()),
          m_length(vec.size()) {}

    // Simple Operations

    iterator begin() const noexcept { return m_data; }
    iterator end() const noexcept { return m_data + m_length; }

    reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
    reverse_iterator rend() const noexcept  { return reverse_iterator(begin()); }

    size_t size() const noexcept { return m_length; }
    bool empty() const noexcept { return m_length == 0; }
    const T* data() const noexcept { return m_data; }

    const T& front() const noexcept {
        assert(!empty());
        return m_data[0];
    }

    const T& back() const noexcept {
        assert(!empty());
        return m_data[m_length-1];
    }

    bool operator==(array_ref rhs) const noexcept {
        return std::equal(begin(), end(), rhs.begin(), rhs.end());
    }

    /**
     * Chop off the first N elements of the array, and keep M
     * elements in the array.
     */
    array_ref slice(size_t n, size_t m) const noexcept {
        assert(n+m <= size() && "Invalid specifier");
        return array_ref(data()+n, m);
    }

    /**
     * Chop off the first N elements of the array.
     */
    array_ref slice(size_t n) const noexcept {
        return slice(n, size()-n);
    }

    const T& operator[](size_t index) const noexcept {
        assert(index < m_length && "Invalid index!");
        return m_data[index];
    }

    const T& at(size_t index) const noexcept {
        assert(index < m_length && "Invalid index!");
        return m_data[index];
    }

    // Expensive operations

    std::vector<T> vec() const noexcept {
        return std::vector<T>(m_data, m_data + m_length);
    }

    operator std::vector<T>() const noexcept {
        return std::vector<T>(m_data, m_data + m_length);
    }
};

namespace detail {
#if CPP_VER >= 17
template <typename... Args>
inline void string_concat(std::stringstream& ss, Args&&... args) {
    (void)(ss << ... << std::forward<Args>(args));
}
#else
inline void string_concat(std::stringstream&) {}

template <typename T, typename... Args>
inline void string_concat(std::stringstream& ss, T&& t, Args&&... args) {
    ss << std::forward<T>(t);
    string_concat(ss, std::forward<Args>(args)...);
}
#endif
}

template <typename... Args>
std::string string_concat(Args&&... args) {
    std::stringstream ss;
    detail::string_concat(ss, std::forward<Args>(args)...);
    return ss.str();
}

} // namespace cxx
