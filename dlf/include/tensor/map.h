#pragma once

namespace dlf {

struct map_id {};

namespace map_detail {

template <typename T>
std::enable_if_t<!is_tensor<T>::value, Shape>
inline shape_of(T&&) { return Shape({}); }

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

template <typename Function>
struct map_impl {
    template <typename... Args>
    void operator()(Function f, size_t n, Args... its) {
        while (n--) {
            f((*its)...);
            (++its, ...);
        }
    }
};

template <>
struct map_impl<xfn::transfer<void>> {
    template <typename Iterator1, typename T>
    void operator()(xfn::transfer<void>, size_t n, Iterator1 q, scalar_iter<T> p) {
        std::fill(q, q + n, *p);
    }

    template <typename Iterator1, typename Iterator2>
    void operator()(xfn::transfer<void>, size_t n, Iterator1 q, Iterator2 p) {
        std::copy(p, p + n, q);
    }
};

template <typename Function, typename First, typename... Rest>
std::enable_if_t<!is_tensor<First>::value>
do_serial_map(Function f, size_t begin, First&& first, Rest&&... rest);

template <typename Function, typename... Rest>
void do_serial_map(Function f, size_t begin, map_id, Rest&&... rest);

template <typename Function, typename First, typename... Rest>
std::enable_if_t<is_cpu_tensor<First>::value>
do_serial_map(Function f, size_t begin, First&& first, Rest&&... rest);

template <typename Function>
inline void do_serial_map(Function f, size_t) {
    f();
}

template <typename Function, typename First, typename... Rest>
std::enable_if_t<!is_tensor<First>::value>
do_serial_map(Function f, size_t begin, First&& first, Rest&&... rest) {
    do_serial_map([&](auto... args) {
        f(scalar_iter<std::remove_reference_t<First>>(std::forward<First>(first)), args...);
    }, begin, std::forward<Rest>(rest)...);
}

template <typename Function, typename... Rest>
void do_serial_map(Function f, size_t begin, map_id, Rest&&... rest) {
    do_serial_map([=](auto... args) {
        f(map_id_iter(begin), args...);
    }, begin, std::forward<Rest>(rest)...);
}

template <typename Function, typename First, typename... Rest>
std::enable_if_t<is_cpu_tensor<First>::value>
do_serial_map(Function f, size_t begin, First&& first, Rest&&... rest) {
    if (first.original_shape().size() == 1) {
        do_serial_map([&](auto... args) {
            f(scalar_iter<std::remove_reference_t<decltype(*first)>>(*first), args...);
        }, begin, std::forward<Rest>(rest)...);
    } else if (first.shape().is_contiguous()) {
        do_serial_map([&](auto... args) {
            f(first.data() + first.shape().offset() + begin, args...);
        }, begin, std::forward<Rest>(rest)...);
    } else {
        do_serial_map([&](auto... args) {
            f(first.begin() + begin, args...);
        }, begin, std::forward<Rest>(rest)...);
    }
}

template <typename Function, typename... Args>
void serial_map(Function f, size_t begin, size_t n, Args&&... args) {
    do_serial_map([f, n](auto... its) {
        map_impl<Function>()(f, n, its...);
    }, begin, std::forward<Args>(args)...);
}

template <typename Function, typename... Args>
void parallel_map(Function f, size_t n, Args&&... args) {
    if (n < GRAINSIZE) {
        serial_map(f, 0, n, std::forward<Args>(args)...);
    } else {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, GRAINSIZE), [&](const auto& r) {
            serial_map(f, r.begin(), r.size(), std::forward<Args>(args)...);
        });
    }
}

} // namespace map_detail

template <typename Function>
auto map(Function f) {
    return [=](auto&&... args) {
        static_assert(
            cxx::conjunction<cxx::negation<is_gpu_tensor<decltype(args)>>...>::value,
            "This operation only supports CPU tensors");
        auto shape = Shape::broadcast(map_detail::shape_of(args)...);
        map_detail::parallel_map(f, shape.size(), map_detail::broadcast_to(args, shape)...);
    };
}

template <typename Function>
auto serial_map(Function f) {
    return [=](auto&&... args) {
        static_assert(
            cxx::conjunction<cxx::negation<is_gpu_tensor<decltype(args)>>...>::value,
            "This operation only supports CPU tensors");
        auto shape = Shape::broadcast(map_detail::shape_of(args)...);
        map_detail::serial_map(f, 0, shape.size(), map_detail::broadcast_to(args, shape)...);
    };
}

template <typename TensorT, typename Function>
auto map(TensorT&& Y, Function f) {
    static_assert(is_cpu_tensor<TensorT>::value, "");
    static_assert(!std::is_const<std::remove_reference_t<TensorT>>::value, "");
    return [&Y, f](auto&&... args) {
        static_assert(
            cxx::conjunction<cxx::negation<is_gpu_tensor<decltype(args)>>...>::value,
            "This operation only supports CPU tensors");
        auto shape = Shape::broadcast(map_detail::shape_of(args)...);
        Y.resize(shape);
        map_detail::parallel_map(xfn::transfer<Function>(f),
            shape.size(), std::forward<TensorT>(Y),
            map_detail::broadcast_to(args, shape)...);
    };
}

template <typename TensorT, typename Function>
auto serial_map(TensorT&& Y, Function f) {
    static_assert(is_cpu_tensor<TensorT>::value, "");
    static_assert(!std::is_const<std::remove_reference_t<TensorT>>::value, "");
    return [&Y, f](auto&&... args) {
        static_assert(
            cxx::conjunction<cxx::negation<is_gpu_tensor<decltype(args)>>...>::value,
            "This operation only supports CPU tensors");
        auto shape = Shape::broadcast(map_detail::shape_of(args)...);
        Y.resize(shape);
        map_detail::serial_map(xfn::transfer<Function>(f),
            0, shape.size(), std::forward<TensorT>(Y),
            map_detail::broadcast_to(args, shape)...);
    };
}

} // namespace dlf
