#pragma once

namespace dlf {

//==-------------------------------------------------------------------------
// Tensor aggregate operations
//==-------------------------------------------------------------------------

namespace detail {
template <typename Fn, typename... Tensors>
struct aggregate_type {};

template <typename Fn, typename TensorT>
struct aggregate_type<Fn, TensorT> {
    using type = tensor_type<TensorT>;
};

template <typename Fn, typename First, typename... Tensors>
struct aggregate_type<Fn, First, Tensors...> {
    using type = tensor_invoke_result<Fn, First, typename aggregate_type<Fn, Tensors...>::type>;
};

template <typename Fn, typename TensorY, typename TensorA>
std::enable_if_t<!is_tensor_view<TensorA>::value>
inline aggregate(Fn, TensorY& Y, TensorA&& A) {
    assert(A.shape() == Y.shape());
    flat_copy(std::forward<TensorA>(A), Y);
}

template <typename Fn, typename TensorY, typename TensorA>
std::enable_if_t<is_tensor_view<TensorA>::value>
inline aggregate(Fn, TensorY& Y, TensorA&& A) {
    assert(A.shape() == Y.shape());
    reorder(std::forward<TensorA>(A), Y);
}

template <typename Fn, typename TensorY, typename First, typename Second, typename... Rest>
inline void aggregate(Fn fn, TensorY& Y, First&& first, Second&& second, Rest... rest) {
    transformTo(std::forward<First>(first), std::forward<Second>(second), Y, fn);
    aggregate(fn, Y, Y, rest...);
}
} // namespace detail

template <typename Fn, typename First, typename... Rest,
    typename = std::enable_if_t<
        is_tensor<First>::value &&
        cxx::conjunction<is_same_tensor<First, Rest>...>::value>>
auto aggregate(Fn fn, First&& first, Rest&&... rest)
    -> typename detail::aggregate_type<Fn, First, Rest...>::type
{
    using TensorR = typename detail::aggregate_type<Fn, First, Rest...>::type;
    TensorR result(Shape::broadcast(first, rest...));
    detail::aggregate(fn, result, std::forward<First>(first), std::forward<Rest>(rest)...);
    return result;
}

template <typename First, typename... Rest>
inline auto max(First&& first, Rest&&... rest) {
    return aggregate(xfn::max<tensor_value_type<First>>(),
                     std::forward<First>(first),
                     std::forward<Rest>(rest)...);
}

template <typename First, typename... Rest>
inline auto min(First&& first, Rest&&... rest) {
    return aggregate(xfn::min<tensor_value_type<First>>(),
                     std::forward<First>(first),
                     std::forward<Rest>(rest)...);
}

template <typename First, typename... Rest>
inline auto sum(First&& first, Rest&&... rest) {
    return aggregate(xfn::plus<>(), std::forward<First>(first), std::forward<Rest>(rest)...);
}

template <typename First, typename... Rest>
inline auto mean(First&& first, Rest&&... rest) {
    auto result = sum(first, rest...);
    using T = tensor_value_type<decltype(result)>;
    result /= static_cast<T>(1 + sizeof...(rest));
    return result;
}

template <typename First, typename... Rest>
inline auto product(First&& first, Rest&&... rest) {
    return aggregate(xfn::multiplies<>(), std::forward<First>(first), std::forward<Rest>(rest)...);
}

} // namespace dlf
