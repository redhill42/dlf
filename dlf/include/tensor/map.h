#pragma once

namespace dlf {

template <typename Body, typename First>
std::enable_if_t<is_cpu_tensor<First>::value && !is_tensor_view<First>::value>
inline unravel(Body body, First&& first) {
    body(first.data());
}

template <typename Body, typename First>
std::enable_if_t<is_cpu_tensor<First>::value && is_tensor_view<First>::value>
inline unravel(Body body, First&& first) {
    if (first.shape().is_contiguous()) {
        body(first.data() + first.shape().offset());
    } else {
        body(first.begin());
    }
}

template <typename Body, typename First, typename... Rest>
std::enable_if_t<is_cpu_tensor<First>::value && !is_tensor_view<First>::value>
inline unravel(Body body, First&& first, Rest&&... rest) {
    unravel([begin = first.data(), body](auto... args) {
        body(begin, args...);
    }, std::forward<Rest>(rest)...);
}

template <typename Body, typename First, typename... Rest>
std::enable_if_t<is_cpu_tensor<First>::value && is_tensor_view<First>::value>
inline unravel(Body body, First&& first, Rest&&... rest) {
    if (first.shape().is_contiguous()) {
        unravel([begin = first.data() + first.shape().offset(), body](auto... args) {
            body(begin, args...);
        }, std::forward<Rest>(rest)...);
    } else {
        unravel([begin = first.begin(), body](auto... args) {
            body(begin, args...);
        }, std::forward<Rest>(rest)...);
    }
}

/*-------------------------------------------------------------------------*/

struct map_id {};

namespace map_detail {

template <typename T>
std::enable_if_t<!is_tensor<T>::value, Shape>
inline shape_of(T&&) { return Shape(); }

template <typename T>
std::enable_if_t<is_cpu_tensor<T>::value, Shape>
inline shape_of(T&& t) { return t.shape(); }

template <typename T>
std::enable_if_t<!is_tensor<T>::value, T&&>
inline broadcast_to(T&& t, const Shape&) {
    return std::forward<T>(t);
}

template <typename T>
std::enable_if_t<is_cpu_tensor<T>::value,
    std::conditional_t<std::is_const<std::remove_reference_t<T>>::value,
        const TensorView<tensor_value_type<T>>,
              TensorView<tensor_value_type<T>>>>
inline broadcast_to(T&& t, const Shape& shape) {
    return t.broadcast_to(shape);
}

template <typename T>
struct scalar_iter {
    using value_type = T;
    using difference_type = ptrdiff_t;
    using reference = value_type&;
    using pointer = value_type*;
    using iterator_category = std::random_access_iterator_tag;

    T* p; difference_type i = 0;
    scalar_iter(T& v, difference_type i = 0) : p(&v), i(i) {}

    reference    operator*()  noexcept { return *p; }
    scalar_iter& operator++() noexcept { ++i; return *this; }
    scalar_iter& operator--() noexcept { --i; return *this; }
    scalar_iter& operator+=(difference_type n) noexcept { i += n; return *this; }
    scalar_iter& operator-=(difference_type n) noexcept { i -= n; return *this; }

    scalar_iter operator+(difference_type n) const noexcept
        { return scalar_iter(*p, i + n); }
    scalar_iter operator-(difference_type n) const noexcept
        { return scalar_iter(*p, i - n); }
    difference_type operator-(const scalar_iter& rhs) const noexcept
        { return i - rhs.i; }

    bool operator==(const scalar_iter& rhs) const noexcept
        { return i == rhs.i; }
    bool operator!=(const scalar_iter& rhs) const noexcept
        { return i != rhs.i; }
    bool operator<(const scalar_iter& rhs) const noexcept
        { return i < rhs.i; }
    bool operator>(const scalar_iter& rhs) const noexcept
        { return i > rhs.i; }
    bool operator<=(const scalar_iter& rhs) const noexcept
        { return i <= rhs.i; }
    bool operator>=(const scalar_iter& rhs) const noexcept
        { return i >= rhs.i; }
};

struct map_id_iter {
    using value_type = size_t;
    using difference_type = ptrdiff_t;
    using reference = const value_type&;
    using pointer = const value_type*;
    using iterator_category = std::random_access_iterator_tag;

    size_t i;
    map_id_iter(size_t i) : i(i) {}

    reference operator*() noexcept { return i; }
    map_id_iter& operator++() noexcept { ++i; return *this; }
    map_id_iter& operator--() noexcept { --i; return *this; }
    map_id_iter& operator+=(difference_type n) noexcept { i += n; return *this; }
    map_id_iter& operator-=(difference_type n) noexcept { i -= n; return *this; }

    map_id_iter operator+(difference_type n) const noexcept
        { return map_id_iter(i + n); }
    map_id_iter operator-(difference_type n) const noexcept
        { return map_id_iter(i - n); }
    difference_type operator-(const map_id_iter& rhs) const noexcept
        { return i - rhs.i; }

    bool operator==(const map_id_iter& rhs) const noexcept
        { return i == rhs.i; }
    bool operator!=(const map_id_iter& rhs) const noexcept
        { return i != rhs.i; }
    bool operator<(const map_id_iter& rhs) const noexcept
        { return i < rhs.i; }
    bool operator>(const map_id_iter& rhs) const noexcept
        { return i > rhs.i; }
    bool operator<=(const map_id_iter& rhs) const noexcept
        { return i <= rhs.i; }
    bool operator>=(const map_id_iter& rhs) const noexcept
        { return i >= rhs.i; }
};

struct map_impl_base {
    template <typename... Args>
    static constexpr bool is_prefer_serial(Args&&...) { return false; }
};

template <typename Function>
struct map_impl : map_impl_base {
    template <typename... Args>
    void operator()(Function f, size_t n, Args... its) {
        while (n--) {
            f((*its)...);
            (++its, ...);
        }
    }
};

template <>
struct map_impl<xfn::transfer<void>> : map_impl_base {
    template <typename Iterator1, typename T>
    void operator()(xfn::transfer<void>, size_t n, Iterator1 q, scalar_iter<T> p) {
        std::fill(q, q + n, *p);
    }

    template <typename Iterator1, typename Iterator2>
    void operator()(xfn::transfer<void>, size_t n, Iterator1 q, Iterator2 p) {
        std::copy(p, p + n, q);
    }
};

#include "vml.h"

/*-------------------------------------------------------------------------*/

template <typename Function, typename First, typename... Rest>
std::enable_if_t<!is_tensor<First>::value>
do_serial_map(Function f, size_t begin, First&& first, Rest&&... rest);

template <typename Function, typename... Rest>
void do_serial_map(Function f, size_t begin, map_id, Rest&&... rest);

template <typename Function, typename First, typename... Rest>
std::enable_if_t<is_cpu_tensor<First>::value && !is_tensor_view<First>::value>
do_serial_map(Function f, size_t begin, First&& first, Rest&&... rest);

template <typename Function, typename First, typename... Rest>
std::enable_if_t<is_cpu_tensor<First>::value && is_tensor_view<First>::value>
do_serial_map(Function f, size_t begin, First&& first, Rest&&... rest);

template <typename Function>
inline void do_serial_map(Function f, size_t) {
    f();
}

template <typename Function, typename First, typename... Rest>
std::enable_if_t<!is_tensor<First>::value>
do_serial_map(Function f, size_t begin, First&& scalar, Rest&&... rest) {
    do_serial_map([&](auto... args) {
        f(scalar_iter<std::remove_reference_t<First>>(scalar), args...);
    }, begin, std::forward<Rest>(rest)...);
}

template <typename Function, typename... Rest>
void do_serial_map(Function f, size_t begin, map_id, Rest&&... rest) {
    do_serial_map([=](auto... args) {
        f(map_id_iter(begin), args...);
    }, begin, std::forward<Rest>(rest)...);
}

template <typename Function, typename First, typename... Rest>
std::enable_if_t<is_cpu_tensor<First>::value && !is_tensor_view<First>::value>
do_serial_map(Function f, size_t begin, First&& tensor, Rest&&... rest) {
    do_serial_map([&](auto... args) {
        f(tensor.data() + begin, args...);
    }, begin, std::forward<Rest>(rest)...);
}

template <typename Function, typename First, typename... Rest>
std::enable_if_t<is_cpu_tensor<First>::value && is_tensor_view<First>::value>
do_serial_map(Function f, size_t begin, First&& tensor, Rest&&... rest) {
    do_serial_map([&](auto... args) {
        if (tensor.shape().is_contiguous())
            f(tensor.data() + tensor.shape().offset() + begin, args...);
        else
            f(tensor.begin() + begin, args...);
    }, begin, std::forward<Rest>(rest)...);
}

template <typename Function, typename... Args>
void serial_map(Function f, size_t begin, size_t n, Args&&... args) {
    do_serial_map([f, n](auto... its) {
        map_impl<Function>()(f, n, its...);
    }, begin, std::forward<Args>(args)...);
}

template <typename Function, typename... Args>
void parallel_map(Function f, size_t n, Args&&... args) {
    if (n < GRAINSIZE || map_impl<Function>::is_prefer_serial(args...)) {
        serial_map(f, 0, n, std::forward<Args>(args)...);
    } else {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, GRAINSIZE), [&](const auto& r) {
            serial_map(f, r.begin(), r.size(), args...);
        });
    }
}

} // namespace map_detail

/*-------------------------------------------------------------------------*/

template <typename Function, typename... Args>
std::enable_if_t<cxx::conjunction<cxx::negation<is_gpu_tensor<Args>>...>::value>
map(Function f, Args&&... args) {
    auto shape = Shape::broadcast(map_detail::shape_of(args)...);
    map_detail::parallel_map(f, shape.size(), map_detail::broadcast_to(args, shape)...);
}

template <typename Function, typename TensorX, typename TensorY>
std::enable_if_t<
    is_cpu_tensor<TensorX>::value && is_cpu_tensor<TensorY>::value &&
    !std::is_const<std::remove_reference_t<TensorY>>::value>
map(xfn::transfer<Function> f, TensorY&& Y, TensorX&& X) {
    Y.resize(X.shape());
    map_detail::parallel_map(f, X.size(), std::forward<TensorY>(Y), std::forward<TensorX>(X));
}

template <typename Function, typename TensorY, typename... Args>
std::enable_if_t<
    is_cpu_tensor<TensorY>::value &&
    !std::is_const<std::remove_reference_t<TensorY>>::value &&
    cxx::conjunction<cxx::negation<is_gpu_tensor<Args>>...>::value>
mapTo(TensorY&& Y, Function f, Args&&... args) {
    auto shape = Shape::broadcast(map_detail::shape_of(args)...);
    Y.resize(shape);
    map_detail::parallel_map(xfn::transfer<Function>(f),
        shape.size(), std::forward<TensorY>(Y),
        map_detail::broadcast_to(args, shape)...);
}

template <typename Function, typename TensorX, typename TensorY>
std::enable_if_t<
    is_cpu_tensor<TensorX>::value && is_cpu_tensor<TensorY>::value &&
    !std::is_const<std::remove_reference_t<TensorY>>::value>
inline mapTo(TensorY&& Y, Function f, TensorX&& X) {
    map(xfn::transfer<Function>(f), std::forward<TensorY>(Y), std::forward<TensorX>(X));
}

} // namespace dlf
