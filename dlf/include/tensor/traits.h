#pragma once

namespace dlf {

//==-------------------------------------------------------------------------
// Tensor traits
//==-------------------------------------------------------------------------

template <typename T> struct is_relop                        : public std::false_type {};
template <typename T> struct is_relop<xfn::equal_to<T>>      : public std::true_type {};
template <typename T> struct is_relop<xfn::not_equal_to<T>>  : public std::true_type {};
template <typename T> struct is_relop<xfn::less<T>>          : public std::true_type {};
template <typename T> struct is_relop<xfn::less_equal<T>>    : public std::true_type {};
template <typename T> struct is_relop<xfn::greater<T>>       : public std::true_type {};
template <typename T> struct is_relop<xfn::greater_equal<T>> : public std::true_type {};

namespace detail {
template <typename T> struct cpu {};
template <typename T> struct gpu {};

template <typename T>
struct tensor_traits_impl {
    using is_tensor = std::false_type;
    using is_view = std::false_type;
    using tag = void;
    using value_type = std::remove_cv_t<std::decay_t<T>>;

    template <typename U>
    using tensor_type = void;

    template <typename U>
    using tensor_view_type = void;
};

template <typename T>
struct tensor_traits_impl<Tensor<T>> {
    using is_tensor = std::true_type;
    using is_view = std::false_type;
    using tag = cpu<void>;
    using value_type = T;

    template <typename U>
    using tensor_type = Tensor<std::remove_cv_t<std::decay_t<U>>>;

    template <typename U>
    using tensor_view_type = TensorView<std::decay_t<U>>;
};

template <typename T>
struct tensor_traits_impl<TensorView<T>> {
    using is_tensor = std::true_type;
    using is_view = std::true_type;
    using tag = cpu<void>;
    using value_type = T;

    template <typename U>
    using tensor_type = Tensor<std::remove_cv_t<std::decay_t<U>>>;

    template <typename U>
    using tensor_view_type = TensorView<std::decay_t<U>>;
};

template <typename T>
struct tensor_traits_impl<DevTensor<T>> {
    using is_tensor = std::true_type;
    using is_view = std::false_type;
    using tag = gpu<T>;
    using value_type = T;

    template <typename U>
    using tensor_type = DevTensor<std::remove_cv_t<std::decay_t<U>>>;

    template <typename U>
    using tensor_view_type = DevTensorView<std::decay_t<U>>;
};

template <typename T>
struct tensor_traits_impl<DevTensorView<T>> {
    using is_tensor = std::true_type;
    using is_view = std::true_type;
    using tag = gpu<T>;
    using value_type = T;

    template <typename U>
    using tensor_type = DevTensor<std::remove_cv_t<std::decay_t<U>>>;

    template <typename U>
    using tensor_view_type = DevTensorView<std::decay_t<U>>;
};
} // namespace detail

template <typename TensorT>
struct tensor_traits : detail::tensor_traits_impl<std::decay_t<TensorT>> {};

template <typename TensorT>
using is_tensor = typename tensor_traits<TensorT>::is_tensor;

template <typename TensorT>
using is_cpu_tensor = std::is_same<typename tensor_traits<TensorT>::tag, detail::cpu<void>>;

template <typename TensorT>
using is_gpu_tensor = cxx::conjunction<is_tensor<TensorT>, cxx::negation<is_cpu_tensor<TensorT>>>;

template <typename TensorT>
using is_tensor_view = typename tensor_traits<TensorT>::is_view;

template <typename X, typename Y>
using is_same_tensor = cxx::conjunction<
    is_tensor<X>, is_tensor<Y>,
    std::is_same<typename tensor_traits<X>::tag, typename tensor_traits<Y>::tag>>;

template <typename TensorT>
using tensor_value_type = typename tensor_traits<TensorT>::value_type;

template <typename TensorT, typename U = tensor_value_type<TensorT>>
using tensor_type = typename tensor_traits<TensorT>::template tensor_type<U>;

template <typename TensorT, typename U = tensor_value_type<TensorT>>
using tensor_view_type = typename tensor_traits<TensorT>::template tensor_view_type<U>;

template <typename X, typename Y>
using is_exactly_same_tensor = cxx::conjunction<
    is_same_tensor<X, Y>,
    std::is_same<std::remove_cv_t<tensor_value_type<X>>, std::remove_cv_t<tensor_value_type<Y>>>>;

template <typename TensorT, typename R = tensor_type<TensorT>>
using enable_if_tensor = std::enable_if_t<is_tensor<TensorT>::value, R>;

template <typename TensorT, typename R = tensor_type<TensorT>>
using enable_if_non_view_tensor =
    std::enable_if_t<is_tensor<TensorT>::value && !is_tensor_view<TensorT>::value, R>;

template <typename Fn, typename LHS, typename RHS>
using tensor_invoke_result = tensor_type<
    std::conditional_t<is_tensor<LHS>::value, LHS, RHS>,
    std::conditional_t<is_relop<Fn>::value, bool,
        std::conditional_t<is_cpu_tensor<LHS>::value || is_cpu_tensor<RHS>::value,
            cxx::invoke_result_t<Fn, tensor_value_type<LHS>, tensor_value_type<RHS>>,
            tensor_value_type<LHS>>>>;

template <typename LHS, typename RHS, typename Fn, typename R = tensor_invoke_result<Fn, LHS, RHS>>
using enable_if_tensors =
    std::enable_if_t<
        is_same_tensor<LHS, RHS>::value ||
        (is_tensor<LHS>::value ^ is_tensor<RHS>::value),
    R>;

template <typename LHS, typename RHS, typename Fn, typename R = tensor_invoke_result<Fn, LHS, RHS>>
using enable_if_non_view_tensors =
    std::enable_if_t<
        !is_tensor_view<LHS>::value && !is_tensor_view<RHS>::value &&
        (is_same_tensor<LHS, RHS>::value || (is_tensor<LHS>::value ^ is_tensor<RHS>::value)),
    R>;

template <typename TensorT, typename U>
inline tensor_type<TensorT, U> tensor_scalar(U&& value) {
    return tensor_type<TensorT, U>::scalar(std::forward<U>(value));
}

} // namespace dlf
