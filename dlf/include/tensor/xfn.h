#pragma once

#include <functional>
#include <type_traits>
#include <string>
#include <cmath>

namespace dlf { namespace xfn {

#define DEFINE_UNARY_FUNCTION($fn, $op) \
template <typename T> \
struct $fn : std::unary_function<T,T> { \
    static const std::string name; \
    constexpr T operator()(const T& x) { return $op; } \
}; \
template <typename T> \
const std::string $fn<T>::name = #$fn;

DEFINE_UNARY_FUNCTION(abs, std::abs(x))
DEFINE_UNARY_FUNCTION(neg, T(-x))
DEFINE_UNARY_FUNCTION(sign, T((T(0)<x) - (x<T(0))))
DEFINE_UNARY_FUNCTION(reciprocal, T(T(1)/x))
DEFINE_UNARY_FUNCTION(floor, std::floor(x))
DEFINE_UNARY_FUNCTION(ceil, std::ceil(x))
DEFINE_UNARY_FUNCTION(round, std::round(x))
DEFINE_UNARY_FUNCTION(sqrt, std::sqrt(x))
DEFINE_UNARY_FUNCTION(exp, std::exp(x))
DEFINE_UNARY_FUNCTION(log, std::log(x))
DEFINE_UNARY_FUNCTION(sin, std::sin(x))
DEFINE_UNARY_FUNCTION(cos, std::cos(x))
DEFINE_UNARY_FUNCTION(tan, std::tan(x))
DEFINE_UNARY_FUNCTION(asin, std::asin(x))
DEFINE_UNARY_FUNCTION(acos, std::acos(x))
DEFINE_UNARY_FUNCTION(atan, std::atan(x))
DEFINE_UNARY_FUNCTION(sinh, std::sinh(x))
DEFINE_UNARY_FUNCTION(cosh, std::cosh(x))
DEFINE_UNARY_FUNCTION(tanh, std::tanh(x))
DEFINE_UNARY_FUNCTION(asinh, std::asinh(x))
DEFINE_UNARY_FUNCTION(acosh, std::acosh(x))
DEFINE_UNARY_FUNCTION(atanh, std::atanh(x))
DEFINE_UNARY_FUNCTION(erf, std::erf(x))
DEFINE_UNARY_FUNCTION(sigmoid, T(1)/(T(1)+std::exp(-x)))

#undef DEFINE_UNARY_FUNCTION

//==-------------------------------------------------------------------------

template <typename T = void>
struct plus : std::plus<T> {
    static const std::string name;
};
template <typename T>
const std::string plus<T>::name = "add";

template <typename T = void>
struct minus : std::minus<T> {
    static const std::string name;
};
template <typename T>
const std::string minus<T>::name = "sub";

template <typename T = void>
struct multiplies : std::multiplies<T> {
    static const std::string name;
};
template <typename T>
const std::string multiplies<T>::name = "mul";

template <typename T = void>
struct divides : std::divides<T> {
    static const std::string name;
};
template <typename T>
const std::string divides<T>::name = "div";

//==-------------------------------------------------------------------------

template <typename T>
struct activation_function : std::unary_function<T,T> {
    const T alpha{0}, beta{0};

    activation_function() = default;

    template <typename T1>
    activation_function(const T1& alpha)
        : alpha(static_cast<T>(alpha)) {}

    template <typename T1, typename T2>
    activation_function(const T1& alpha, const T2& beta)
        : alpha(static_cast<T>(alpha)), beta(static_cast<T>(beta)) {}
};

template <typename T>
struct relu : activation_function<T> {
    static const std::string name;
    constexpr T operator()(const T& x) {
        return std::max(T(0), x);
    }
};
template <typename T>
const std::string relu<T>::name = "relu";

template <typename T>
struct prelu : std::binary_function<T,T,T> {
    static const std::string name;
    constexpr T operator()(const T& x, const T& slope) {
        return x < T(0) ? x*slope : x;
    }
};
template <typename T>
const std::string prelu<T>::name = "prelu";

template <typename T>
struct leaky_relu : activation_function<T> {
    static const std::string name;
    using activation_function<T>::alpha;

    template <typename U>
    constexpr leaky_relu(const U& alpha)
        : activation_function<T>(alpha) {}

    constexpr T operator()(const T& x) {
        return x < T(0) ? x*alpha : x;
    }
};
template <typename T>
const std::string leaky_relu<T>::name = "leaky_relu";

template <typename T>
struct thresholded_relu : activation_function<T> {
    static const std::string name;
    using activation_function<T>::alpha;

    template <typename U>
    constexpr thresholded_relu(const U& alpha)
        : activation_function<T>(static_cast<T>(alpha)) {}

    constexpr T operator()(const T& x) {
        return x > alpha ? x : T(0);
    }
};
template <typename T>
const std::string thresholded_relu<T>::name = "thresholded_relu";

template <typename T>
struct selu : activation_function<T> {
    static const std::string name;
    using activation_function<T>::alpha;
    using activation_function<T>::beta;

    template <typename T1, typename T2>
    constexpr selu(const T1& alpha, const T2& gamma)
        : activation_function<T>(alpha, gamma) {}

    constexpr T operator()(const T& x) {
        return T(beta * (x < T(0) ? alpha * (std::exp(x) - T(1)) : x));
    }
};
template <typename T>
const std::string selu<T>::name = "selu";

template <typename T>
struct elu : activation_function<T> {
    static const std::string name;
    using activation_function<T>::alpha;

    template <typename U>
    constexpr elu(const U& alpha) : activation_function<T>(alpha) {}

    constexpr T operator()(const T& x) {
        return x < T(0) ? alpha * (std::exp(x) - T(1)) : x;
    }
};
template <typename T>
const std::string elu<T>::name = "elu";

template <typename T>
struct hard_sigmoid : activation_function<T> {
    static const std::string name;
    using activation_function<T>::alpha;
    using activation_function<T>::beta;

    template <typename T1, typename T2>
    constexpr hard_sigmoid(const T1& alpha, const T2& beta)
        : activation_function<T>(alpha, beta) {}

    constexpr T operator()(const T& x) {
        return std::max(T(0), std::min(T(1), alpha*x + beta));
    }
};
template <typename T>
const std::string hard_sigmoid<T>::name = "hard_sigmoid";

template <typename T>
struct softsign : activation_function<T> {
    static const std::string name;
    constexpr T operator()(const T& x) {
        return T(x / (T(1) + std::abs(x)));
    }
};
template <typename T>
const std::string softsign<T>::name = "softsign";

template <typename T>
struct softplus : activation_function<T> {
    static const std::string name;
    constexpr T operator()(const T& x) {
        return T(std::log(std::exp(x) + T(1)));
    }
};
template <typename T>
const std::string softplus<T>::name = "softplus";

}} // namespace dlf::xfn
