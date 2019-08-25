#pragma once

namespace dlf {

//==-------------------------------------------------------------------------
// Tensor unary operations
//==-------------------------------------------------------------------------

/**
 * Transform tensor A's elements to tensor B by applying the given function.
 * The two tensor must have the same shape.
 */
template <typename TensorT, typename U, typename F>
std::enable_if_t<is_cpu_tensor<TensorT>::value, Tensor<U>&>
inline transformTo(const TensorT& X, Tensor<U>& Y, F f) {
    Y.resize(X.shape());
    par::transform(X.begin(), X.end(), Y.begin(), f);
    return Y;
}

template <typename TensorT, typename U, typename F>
std::enable_if_t<is_cpu_tensor<TensorT>::value, TensorView<U>&>
inline transformTo(const TensorT& X, TensorView<U>& Y, F f) {
    assert(Y.shape() == X.shape());
    par::transform(X.begin(), X.end(), Y.begin(), f);
    return Y;
}

template <typename TensorT, typename U, typename F>
std::enable_if_t<is_cpu_tensor<TensorT>::value, TensorView<U>&>
inline transformTo(const TensorT& X, TensorView<U>&& Y, F f) {
    return transformTo(X, Y, f);
}

/**
 * Transform a tensor to a new tensor by applying the given unary function
 * on tensor's elements.
 */
template <typename TensorT, typename F, typename U = cxx::invoke_result_t<F,tensor_value_type<TensorT>>>
std::enable_if_t<is_cpu_tensor<TensorT>::value, Tensor<U>>
inline transform(const TensorT& X, F f) {
    Tensor<U> Y{};
    transformTo(X, Y, f);
    return Y;
}

template <typename T, typename F>
std::enable_if_t<std::is_same<cxx::invoke_result_t<F,T>,T>::value, Tensor<T>>
inline transform(Tensor<T>&& A, F f) {
    return std::move(transformTo(A, A, f));
}

//==-------------------------------------------------------------------------
// DevTensor unary operations
//==-------------------------------------------------------------------------

namespace detail {
template <typename TensorX, typename TensorY>
void transform(const std::string& name, const TensorX& X, TensorY& Y) {
    assert(X.shape() == Y.shape());
    if (X.shape().is_contiguous() && Y.shape().is_contiguous()) {
        gpgpu::dnn::transform(name, X.size(),
                              X.data(), X.shape().offset(),
                              Y.data(), Y.shape().offset());
    } else {
        gpgpu::dnn::transform(name, X.size(), X.shape().extents(),
                              X.data(), X.shape().offset(), X.shape().strides(),
                              Y.data(), Y.shape().offset(), Y.shape().strides());
    }
}

template <typename T, typename TensorX, typename TensorY>
void transform(const std::string name, const T alpha, const T beta,
               const TensorX& X, TensorY& Y)
{
    assert(X.shape() == Y.shape());
    if (X.shape().is_contiguous() && Y.shape().is_contiguous()) {
        gpgpu::dnn::transform(name, alpha, beta, X.size(),
                              X.data(), X.shape().offset(),
                              Y.data(), Y.shape().offset());
    } else {
        gpgpu::dnn::transform(name, alpha, beta, X.size(), X.shape().extents(),
                              X.data(), X.shape().offset(), X.shape().strides(),
                              Y.data(), Y.shape().offset(), Y.shape().strides());
    }
}
} // namespace detail

template <typename T, typename DevTensorT, typename Fn>
std::enable_if_t<
    is_same_tensor<DevTensorT, DevTensor<T>>::value &&
    !std::is_base_of<xfn::parameterized_function<T>, Fn>::value,
    DevTensor<T>&>
inline transformTo(const DevTensorT& X, DevTensor<T>& Y, Fn) {
    Y.resize(X.shape());
    detail::transform(Fn::name, X, Y);
    return Y;
}

template <typename T, typename DevTensorT, typename Fn>
std::enable_if_t<
    is_same_tensor<DevTensorT, DevTensorView<T>>::value &&
    !std::is_base_of<xfn::parameterized_function<T>, Fn>::value,
    DevTensorView<T>&>
inline transformTo(const DevTensorT& X, DevTensorView<T>& Y, Fn) {
    assert(Y.shape() == X.shape());
    detail::transform(Fn::name, X, Y);
    return Y;
}

template <typename T, typename DevTensorT, typename Fn>
std::enable_if_t<
    is_same_tensor<DevTensorT, DevTensorView<T>>::value &&
    !std::is_base_of<xfn::parameterized_function<T>, Fn>::value,
    DevTensorView<T>&>
inline transformTo(const DevTensorT& X, DevTensorView<T>&& Y, Fn fn) {
    return transformTo(X, Y, fn);
}

template <typename T, typename DevTensorT, typename Fn>
std::enable_if_t<
    is_same_tensor<DevTensorT, DevTensor<T>>::value &&
    std::is_base_of<xfn::parameterized_function<T>, Fn>::value,
    DevTensor<T>&>
inline transformTo(const DevTensorT& X, DevTensor<T>& Y, Fn fn) {
    Y.resize(X.shape());
    detail::transform(Fn::name, fn.alpha, fn.beta, X, Y);
    return Y;
}

template <typename T, typename DevTensorT, typename Fn>
std::enable_if_t<
    is_same_tensor<DevTensorT, DevTensorView<T>>::value &&
    std::is_base_of<xfn::parameterized_function<T>, Fn>::value,
    DevTensorView<T>&>
inline transformTo(const DevTensorT& X, DevTensorView<T>& Y, Fn fn) {
    assert(Y.shape() == X.shape());
    detail::transform(Fn::name, fn.alpha, fn.beta, X, Y);
    return Y;
}

template <typename T, typename DevTensorT, typename Fn>
std::enable_if_t<
    is_same_tensor<DevTensorT, DevTensorView<T>>::value &&
    std::is_base_of<xfn::parameterized_function<T>, Fn>::value,
    DevTensorView<T>&>
inline transformTo(const DevTensorT& X, DevTensorView<T>&& Y, Fn fn) {
    return transformTo(X, Y, fn);
}

template <typename DevTensorT, typename Fn>
std::enable_if_t<is_gpu_tensor<DevTensorT>::value, tensor_type<DevTensorT>>
inline transform(const DevTensorT& X, Fn fn) {
    tensor_type<DevTensorT> Y{};
    transformTo(X, Y, fn);
    return Y;
}

template <typename T, typename Fn>
inline DevTensor<T> transform(DevTensor<T>&& X, Fn fn) {
    return std::move(transformTo(X, X, fn));
}

//==-------------------------------------------------------------------------
// Uniform tensor unary operators
//==-------------------------------------------------------------------------

#define DEFINE_UNARY_OPERATOR(name, op) \
template <typename TensorT> \
inline enable_if_tensor<TensorT> name(TensorT&& X) { \
    using T = tensor_value_type<TensorT>; \
    return transform(std::forward<TensorT>(X), xfn::op<T>()); \
}

DEFINE_UNARY_OPERATOR(abs, abs)
DEFINE_UNARY_OPERATOR(operator-, negate)
DEFINE_UNARY_OPERATOR(sign, sign)
DEFINE_UNARY_OPERATOR(reciprocal, reciprocal)
DEFINE_UNARY_OPERATOR(floor, floor)
DEFINE_UNARY_OPERATOR(ceil, ceil)
DEFINE_UNARY_OPERATOR(round, round)
DEFINE_UNARY_OPERATOR(sqrt, sqrt)
DEFINE_UNARY_OPERATOR(square, square)
DEFINE_UNARY_OPERATOR(exp, exp)
DEFINE_UNARY_OPERATOR(log, log)
DEFINE_UNARY_OPERATOR(sin, sin)
DEFINE_UNARY_OPERATOR(cos, cos)
DEFINE_UNARY_OPERATOR(tan, tan)
DEFINE_UNARY_OPERATOR(asin, asin)
DEFINE_UNARY_OPERATOR(acos, acos)
DEFINE_UNARY_OPERATOR(atan, atan)
DEFINE_UNARY_OPERATOR(sinh, sinh)
DEFINE_UNARY_OPERATOR(cosh, cosh)
DEFINE_UNARY_OPERATOR(tanh, tanh)
DEFINE_UNARY_OPERATOR(asinh, asinh)
DEFINE_UNARY_OPERATOR(acosh, acosh)
DEFINE_UNARY_OPERATOR(atanh, atanh)
DEFINE_UNARY_OPERATOR(erf, erf)
DEFINE_UNARY_OPERATOR(sigmoid, sigmoid)
#undef DEFINE_UNARY_OPERATOR

template <typename TensorT>
inline enable_if_tensor<TensorT>
clip(TensorT&& X, const tensor_value_type<TensorT> low, const tensor_value_type<TensorT> high) {
    return transform(std::forward<TensorT>(X), xfn::clip<tensor_value_type<TensorT>>(low, high));
}


//==-------------------------------------------------------------------------
// Tensor binary transformation implementation
//==-------------------------------------------------------------------------

namespace detail {
template <typename T, typename U, typename IteratorC, typename F>
void transformChannel(const Shape& shape_A, const T* data_A,
                      const Shape& shape_B, const U* data_B,
                      const Shape& shape_C, IteratorC begin_C,
                      int axis, F f)
{
    assert(shape_B.rank() == 1 || shape_A.find_channel_axis(shape_B) == axis);
    assert(axis < shape_A.rank());
    assert(shape_A.extent(axis) == shape_B.size());
    assert(shape_C == shape_A);

    size_t m = 1;
    for (int i = 0; i <= axis; i++)
        m *= shape_A.extent(i);
    size_t n = shape_A.size() / m;

    tbb::parallel_for(tbb::blocked_range2d<int>(0, m, 32, 0, n, 32), [&](auto r) {
        auto offset = r.rows().begin()*n + r.cols().begin();
        auto px = data_A + shape_A.offset() + offset;
        auto py = data_B + shape_B.offset();
        auto pz = begin_C + offset;
        for (int id = r.rows().begin(); id < r.rows().end(); id++) {
            auto y = py[id % shape_B.size()];
            std::transform(px, px+r.cols().size(), pz, [&](auto x){ return f(x, y); });
            px += n, pz += n;
        }
    });
}

template <typename T, typename U, typename W, typename F>
inline void transformChannel(const Tensor<T>& A, const Tensor<U>& B, Tensor<W>& C, size_t axis, F f) {
    transformChannel(A.shape(), A.data(), B.shape(), B.data(), C.shape(), C.begin(), axis, f);
}

template <typename T, typename U, typename IteratorC, typename F>
void transform(const Shape& shape_A, const T* data_A, const size_t size_A,
               const Shape& shape_B, const U* data_B, const size_t size_B,
               const Shape& shape_C, IteratorC begin_C, F f)
{
    Shape sA = shape_A.broadcast(shape_C);
    Shape sB = shape_B.broadcast(shape_C);
    int   axis;

    if (size_A == 1) {
        if (sB.is_contiguous()) {
            par::transform(data_B + sB.offset(), data_B + sB.offset() + sB.size(),
                           begin_C,
                           [x = *data_A, f](auto& y){ return f(x, y); });
        } else {
            par::transform(const_shaped_iterator<U>(sB, data_B, 0),
                           const_shaped_iterator<U>(sB, data_B, sB.size()),
                           begin_C,
                           [x = *data_A, f](auto& y){ return f(x, y); });
        }
        return;
    }

    if (size_B == 1) {
        if (sA.is_contiguous()) {
            par::transform(data_A + sA.offset(), data_A + sA.offset() + sA.size(),
                           begin_C,
                           [y = *data_B, f](auto& x){ return f(x, y); });
        } else {
            par::transform(const_shaped_iterator<T>(sA, data_A, 0),
                           const_shaped_iterator<T>(sA, data_A, sA.size()),
                           begin_C,
                           [y = *data_B, f](auto& x){ return f(x, y); });
        }
        return;
    }

    if (shape_A.is_contiguous() && shape_B.is_contiguous()) {
        if (shape_A == shape_B) {
            assert(shape_A == sA && shape_B == sB);
            par::transform(data_A + sA.offset(), data_A + sA.offset() + sA.size(),
                           data_B + sB.offset(),
                           begin_C, f);
            return;
        }

        if ((axis = shape_A.find_channel_axis(shape_B)) != -1) {
            transformChannel(shape_A, data_A, shape_B, data_B, shape_C, begin_C, axis, f);
            return;
        }
    }

    if (sA.is_contiguous()) {
        par::transform(data_A + sA.offset(), data_A + sA.offset() + sA.size(),
                       const_shaped_iterator<U>(sB, data_B, 0),
                       begin_C, f);
    } else if (sB.is_contiguous()) {
        par::transform(const_shaped_iterator<T>(sA, data_A, 0),
                       const_shaped_iterator<T>(sA, data_A, sA.size()),
                       data_B + sB.offset(),
                       begin_C, f);
    } else {
        par::transform(const_shaped_iterator<T>(sA, data_A, 0),
                       const_shaped_iterator<T>(sA, data_A, sA.size()),
                       const_shaped_iterator<U>(sB, data_B, 0),
                       begin_C, f);
    }
}

template <typename T, typename U, typename W, typename F>
Tensor<W>& transform(const Shape& shape_A, const T* data_A, const size_t size_A,
                     const Shape& shape_B, const U* data_B, const size_t size_B,
                     Tensor<W>& C, F f)
{
    C.resize(Shape::broadcast(shape_A, shape_B));
    transform(shape_A, data_A, size_A, shape_B, data_B, size_B, C.shape(), C.begin(), f);
    return C;
}

template <typename T, typename U, typename W, typename F>
TensorView<W>& transform(const Shape& shape_A, const T* data_A, const size_t size_A,
                         const Shape& shape_B, const U* data_B, const size_t size_B,
                         TensorView<W>& C, F f)
{
    if (C.shape() != Shape::broadcast(shape_A, shape_B))
        throw shape_error("incompatible shape");
    if (C.shape().is_contiguous()) {
        transform(shape_A, data_A, size_A, shape_B, data_B, size_B, C.shape(), C.data() + C.shape().offset(), f);
    } else {
        transform(shape_A, data_A, size_A, shape_B, data_B, size_B, C.shape(), C.begin(), f);
    }
    return C;
}
} // namespace detail

//==-------------------------------------------------------------------------
// DevTensor binary transformation implementation
//==-------------------------------------------------------------------------

namespace detail {
template <typename T, typename R, typename F>
void transformChannel(const DevTensor<T>& A, const DevTensor<T>& B, DevTensor<R>& C, size_t axis, F) {
    assert(B.is_vector() || A.shape().find_channel_axis(B.shape()) == axis);
    assert(axis < A.rank());
    assert(A.extent(axis) == B.size());
    assert(C.shape() == A.shape());

    size_t m = A.shape().partial_size(0, axis+1);
    size_t n = A.size() / m;
    gpgpu::dnn::transform(F::name, m, n, B.size(), A.data(), 0, B.data(), 0, C.data(), 0);
}

template <typename T, typename R>
void transform(const std::string& name,
               const Shape& shape_A, const gpgpu::Buffer<T>& data_A,
               const Shape& shape_B, const gpgpu::Buffer<T>& data_B,
               const Shape& shape_C, gpgpu::Buffer<R>& data_C)
{
    if (shape_A.is_contiguous() && shape_B.is_contiguous() && shape_C.is_contiguous()) {
        if (shape_A.is_tail(shape_B) || shape_B.is_tail(shape_A)) {
            gpgpu::dnn::transform(name,
                                  shape_A.size(), data_A, shape_A.offset(),
                                  shape_B.size(), data_B, shape_B.offset(),
                                  data_C, shape_C.offset());
            return;
        }

        int axis = shape_A.find_channel_axis(shape_B);
        if (axis != -1) {
            size_t m = shape_A.partial_size(0, axis+1);
            size_t n = shape_A.size() / m;
            gpgpu::dnn::transform(name, m, n, shape_B.size(),
                                  data_A, shape_A.offset(),
                                  data_B, shape_B.offset(),
                                  data_C, shape_C.offset());
            return;
        }
    }

    auto sA = shape_A.broadcast(shape_C);
    auto sB = shape_B.broadcast(shape_C);
    gpgpu::dnn::transform(name, shape_C.size(), shape_C.extents(),
                          data_A, sA.offset(), sA.strides(),
                          data_B, sB.offset(), sB.strides(),
                          data_C, shape_C.offset(), shape_C.strides());
}

template <typename T, typename R, typename F>
DevTensor<R>& transform(const Shape& shape_A, const gpgpu::Buffer<T>& data_A, const size_t,
                        const Shape& shape_B, const gpgpu::Buffer<T>& data_B, const size_t,
                        DevTensor<R>& C, F) {
    C.resize(Shape::broadcast(shape_A, shape_B));
    transform(F::name, shape_A, data_A, shape_B, data_B, C.shape(), C.data());
    return C;
}

template <typename T, typename R, typename F>
DevTensorView<R>& transform(const Shape& shape_A, const gpgpu::Buffer<T>& data_A, const size_t,
                            const Shape& shape_B, const gpgpu::Buffer<T>& data_B, const size_t,
                            DevTensorView<R>& C, F) {
    if (C.shape() != Shape::broadcast(shape_A, shape_B))
        throw shape_error("incompatible shape");
    transform(F::name, shape_A, data_A, shape_B, data_B, C.shape(), C.data());
    return C;
}
} // namespace detail

//==-------------------------------------------------------------------------
// Uniform tensor binary transformation
//==-------------------------------------------------------------------------

template <typename LHS, typename RHS, typename RET, typename F>
std::enable_if_t<
    is_same_tensor<LHS, RHS>::value &&
    is_same_tensor<RET,
        std::conditional_t<
            is_gpu_tensor<LHS>::value && is_relop<F>::value,
            tensor_type<LHS, bool>,
            LHS>>::value,
    RET&>
inline transformTo(const LHS& A, const RHS& B, RET&& C, F f) {
    return detail::transform(A.shape(), A.data(), A.original_shape().size(),
                             B.shape(), B.data(), B.original_shape().size(), C, f);
}

template <typename LHS, typename RHS, typename RET, typename F>
std::enable_if_t<
    is_same_tensor<LHS, RHS>::value &&
    !is_tensor_view<LHS>::value &&
    !is_tensor_view<RHS>::value &&
    !is_tensor_view<RET>::value &&
    is_same_tensor<RET,
        std::conditional_t<
            is_gpu_tensor<LHS>::value && is_relop<F>::value,
            tensor_type<LHS, bool>,
            LHS>>::value>
inline transformChannel(const LHS& A, const RHS& B, RET& C, size_t axis, F f) {
    detail::transformChannel(A, B, C, axis, f);
}

#if __cplusplus >= 201703L
template <typename LHS, typename RHS, typename F>
std::enable_if_t<is_same_tensor<LHS, RHS>::value, tensor_invoke_result<F, LHS, RHS>>
transform(LHS&& A, RHS&& B, F f) {
    using RET = tensor_invoke_result<F, LHS, RHS>;

    if constexpr (std::is_rvalue_reference<decltype(A)>::value &&
                  !is_tensor_view<LHS>::value &&
                  std::is_same<tensor_value_type<LHS>, tensor_value_type<RET>>::value) {
        if (A.shape() == Shape::broadcast(A, B))
            return std::move(transformTo(A, B, A, f));
    }

    if constexpr (std::is_rvalue_reference<decltype(B)>::value &&
                  !is_tensor_view<RHS>::value &&
                  std::is_same<tensor_value_type<RHS>, tensor_value_type<RET>>::value) {
        if (B.shape() == Shape::broadcast(A, B))
            return std::move(transformTo(A, B, B, f));
    }

    RET C;
    transformTo(A, B, C, f);
    return C;
}
#else
namespace detail {
template <typename LHS, typename RHS, typename F>
std::enable_if_t<is_same_tensor<LHS, RHS>::value, tensor_invoke_result<F, LHS, RHS>>
basic_transform(const LHS& A, const RHS& B, F f) {
    tensor_invoke_result<F, LHS, RHS> C{};
    transformTo(A, B, C, f);
    return C;
}

template <typename LHS, typename RHS, typename F>
std::enable_if_t<is_same_tensor<LHS, RHS>::value, tensor_invoke_result<F, LHS, RHS>>
inline transform(const LHS& A, const RHS& B, F f) {
    return basic_transform(A, B, f);
}

template <typename LHS, typename RHS, typename F>
std::enable_if_t<
    is_same_tensor<LHS, RHS>::value &&
    !is_tensor_view<LHS>::value && !std::is_lvalue_reference<LHS>::value &&
    std::is_same<tensor_type<LHS>, tensor_invoke_result<F,LHS,RHS>>::value &&
    !is_tensor_view<RHS>::value && !std::is_lvalue_reference<RHS>::value &&
    std::is_same<tensor_type<RHS>, tensor_invoke_result<F,LHS,RHS>>::value,
    tensor_invoke_result<F, LHS, RHS>>
inline transform(LHS&& A, RHS&& B, F f) {
    if (A.shape() == Shape::broadcast(A, B))
        return std::move(transformTo(A, B, A, f));
    if (B.shape() == Shape::broadcast(A, B))
        return std::move(transformTo(A, B, B, f));
    return basic_transform(A, B, f);
}

template <typename LHS, typename RHS, typename F>
std::enable_if_t<
    is_same_tensor<LHS, RHS>::value &&
    !is_tensor_view<LHS>::value && !std::is_lvalue_reference<LHS>::value &&
    std::is_same<tensor_type<LHS>, tensor_invoke_result<F,LHS,RHS>>::value,
    tensor_invoke_result<F, LHS, RHS>>
inline transform(LHS&& A, const RHS& B, F f) {
    if (A.shape() == Shape::broadcast(A, B))
        return std::move(transformTo(A, B, A, f));
    return basic_transform(A, B, f);
}

template <typename LHS, typename RHS, typename F>
std::enable_if_t<
    is_same_tensor<LHS, RHS>::value &&
    !is_tensor_view<RHS>::value && !std::is_lvalue_reference<RHS>::value &&
    std::is_same<tensor_type<RHS>, tensor_invoke_result<F,LHS,RHS>>::value,
    tensor_invoke_result<F, LHS, RHS>>
inline transform(const LHS& A, RHS&& B, F f) {
    if (B.shape() == Shape::broadcast(A, B))
        return std::move(transformTo(A, B, B, f));
    return basic_transform(A, B, f);
}
} // namespace detail

template <typename LHS, typename RHS, typename F>
std::enable_if_t<is_same_tensor<LHS, RHS>::value, tensor_invoke_result<F, LHS, RHS>>
inline transform(LHS&& lhs, RHS&& rhs, F f) {
    return detail::transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), f);
}
#endif

template <typename LHS, typename RHS, typename F>
std::enable_if_t<is_tensor<LHS>::value && !is_tensor<RHS>::value, tensor_invoke_result<F, LHS, RHS>>
inline transform(LHS&& A, RHS&& B, F f) {
    return transform(std::forward<LHS>(A), tensor_scalar<LHS>(std::forward<RHS>(B)), f);
}

template <typename LHS, typename RHS, typename F>
std::enable_if_t<!is_tensor<LHS>::value && is_tensor<RHS>::value, tensor_invoke_result<F, LHS, RHS>>
inline transform(LHS&& A, RHS&& B, F f) {
    return transform(tensor_scalar<RHS>(std::forward<LHS>(A)), std::forward<RHS>(B), f);
}

//==-------------------------------------------------------------------------
// Uniform tensor binary operators
//==-------------------------------------------------------------------------

#define DEFINE_BINARY_OPERATOR(op, fn)                                              \
template <typename LHS, typename RHS>                                               \
std::enable_if_t<is_same_tensor<LHS, RHS>::value, LHS&>                             \
inline operator op##=(LHS& lhs, RHS&& rhs) {                                        \
    return transformTo(lhs, std::forward<RHS>(rhs), lhs, xfn::fn<>());              \
}                                                                                   \
                                                                                    \
template <typename LHS, typename RHS>                                               \
std::enable_if_t<is_same_tensor<LHS, RHS>::value, LHS>                              \
inline operator op##=(LHS&& lhs, RHS&& rhs) {                                       \
    return std::move(operator op##=(lhs, std::forward<RHS>(rhs)));                  \
}                                                                                   \
                                                                                    \
template <typename LHS, typename RHS>                                               \
std::enable_if_t<is_tensor<LHS>::value && !is_tensor<RHS>::value, LHS&>             \
inline operator op##=(LHS& lhs, RHS&& rhs) {                                        \
    return operator op##=(lhs, tensor_scalar<LHS>(std::forward<RHS>(rhs)));         \
}                                                                                   \
                                                                                    \
template <typename LHS, typename RHS>                                               \
std::enable_if_t<is_tensor<LHS>::value && !is_tensor<RHS>::value, LHS>              \
inline operator op##=(LHS&& lhs, RHS&& rhs) {                                       \
    return std::move(operator op##=(lhs, std::forward<RHS>(rhs)));                  \
}                                                                                   \
                                                                                    \
template <typename LHS, typename RHS>                                               \
enable_if_tensors<LHS, RHS, xfn::fn<>>                                              \
inline operator op(LHS&& lhs, RHS&& rhs) {                                          \
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::fn<>());  \
}

DEFINE_BINARY_OPERATOR(+, plus)
DEFINE_BINARY_OPERATOR(-, minus)
DEFINE_BINARY_OPERATOR(*, multiplies)
DEFINE_BINARY_OPERATOR(/, divides)
DEFINE_BINARY_OPERATOR(%, modulus)
DEFINE_BINARY_OPERATOR(&, bit_and)
DEFINE_BINARY_OPERATOR(|, bit_or)
DEFINE_BINARY_OPERATOR(^, bit_xor)
#undef DEFINE_BINARY_OPERATOR

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::less<>>
inline operator<(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::less<>());
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::less_equal<>>
inline operator<=(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::less_equal<>());
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::greater<>>
inline operator>(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::greater<>());
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::greater_equal<>>
inline operator>=(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::greater_equal<>());
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::logical_and<>>
inline operator&&(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::logical_and<>());
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::logical_or<>>
inline operator||(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::logical_or<>());
}

template <typename TensorT>
enable_if_tensor<TensorT, Tensor<bool>>
inline operator!(TensorT&& x) {
    return transform(std::forward<TensorT>(x), xfn::logical_not<tensor_value_type<TensorT>>());
}

template <typename LHS, typename RHS>
inline enable_if_tensors<LHS, RHS, xfn::power<>>
pow(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::power<>());
}

//==-------------------------------------------------------------------------
// Tensor ternary operations
//==-------------------------------------------------------------------------

template <typename TensorC, typename TensorX, typename TensorY>
std::enable_if_t<
    is_cpu_tensor<TensorX>::value &&
    is_exactly_same_tensor<TensorX, TensorY>::value &&
    is_exactly_same_tensor<TensorC, tensor_type<TensorX, bool>>::value>
inline where(const TensorC& C, const TensorX& X, const TensorY& Y, tensor_type<TensorX>& Z) {
    auto z_shape = Shape::broadcast(C, X, Y);
    Z.resize(z_shape);

    auto c_shape = C.shape().broadcast(z_shape);
    auto x_shape = X.shape().broadcast(z_shape);
    auto y_shape = Y.shape().broadcast(z_shape);
    using T = tensor_value_type<TensorX>;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, Z.size(), GRAINSIZE), [&](auto r) {
        auto c_it = const_shaped_iterator<bool>(c_shape, C.data(), r.begin());
        auto x_it = const_shaped_iterator<T>(x_shape, X.data(), r.begin());
        auto y_it = const_shaped_iterator<T>(y_shape, Y.data(), r.begin());
        auto pz = Z.data() + r.begin();
        for (int count = r.size(); --count >= 0; ++pz, ++c_it, ++x_it, ++y_it) {
            *pz = *c_it ? *x_it : *y_it;
        }
    });
}

template <typename TensorC, typename TensorX, typename TensorY>
std::enable_if_t<
    is_gpu_tensor<TensorX>::value &&
    is_exactly_same_tensor<TensorX, TensorY>::value &&
    is_exactly_same_tensor<TensorC, tensor_type<TensorX, bool>>::value>
inline where(const TensorC& C, const TensorX& X, const TensorY& Y, tensor_type<TensorX>& Z) {
    auto final_shape = Shape::broadcast(C, X, Y);
    Z.resize(final_shape);

    auto c_shape = C.shape().broadcast(final_shape);
    auto x_shape = X.shape().broadcast(final_shape);
    auto y_shape = Y.shape().broadcast(final_shape);

    gpgpu::dnn::where(
        final_shape.size(), final_shape.rank(),
        C.data(), c_shape.offset(), c_shape.extents(), c_shape.strides(),
        X.data(), x_shape.offset(), x_shape.extents(), x_shape.strides(),
        Y.data(), y_shape.offset(), y_shape.extents(), y_shape.strides(),
        Z.data(), 0);
}

template <typename TensorC, typename TensorX, typename TensorY>
std::enable_if_t<
    is_exactly_same_tensor<TensorX, TensorY>::value &&
    is_exactly_same_tensor<TensorC, tensor_type<TensorX, bool>>::value,
    tensor_type<TensorX>>
where(const TensorC& C, const TensorX& X, const TensorY& Y) {
    tensor_type<TensorX> Z{};
    where(C, X, Y, Z);
    return Z;
}

} // namespace dlf
