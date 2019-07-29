#pragma once

#include <functional>
#include <type_traits>
#include <string>
#include <cmath>

namespace dlf { namespace xfn {

template <typename T>
struct parameterized_function {
    const T alpha{0}, beta{0};
    constexpr parameterized_function() = default;
    constexpr explicit parameterized_function(const T& alpha)
        : alpha(alpha) {}
    constexpr explicit parameterized_function(const T& alpha, const T& beta)
        : alpha(alpha), beta(beta) {}
};

template <typename T = void>
struct abs : std::unary_function<T,T> {
    static constexpr auto name = "abs";
    constexpr T operator()(const T& x) const
        { return std::abs(x); }
};

template <>
struct abs<void> {
    static constexpr auto name = "abs";
    template <typename T>
    constexpr auto operator()(T&& x) const
    noexcept(noexcept(std::abs(std::forward<T>(x))))
    -> decltype      (std::abs(std::forward<T>(x)))
        { return      std::abs(std::forward<T>(x)); }
};

template <typename T = void>
struct negate : std::negate<T> {
    static constexpr auto name = "neg";
};

template <typename T = void>
struct sign : std::unary_function<T,T> {
    static constexpr auto name = "sign";
    constexpr T operator()(const T& x) const
        { return (T() < x) - (x < T()); }
};

template <>
struct sign<void> {
    static constexpr auto name = "sign";
    template <typename T>
    constexpr auto operator()(T&& x) const
    noexcept(noexcept((T() < x) - (x < T())))
    -> decltype      ((T() < x) - (x < T()))
        { return      (T() < x) - (x < T()); }
};

#define DEFINE_UNARY_FUNCTION(fn, op) \
template <typename T> \
struct fn : std::unary_function<T,T> { \
    static constexpr auto name = #fn; \
    T operator()(const T& x) const { return op; } \
};

DEFINE_UNARY_FUNCTION(reciprocal, T(1)/x)
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
    static constexpr auto name = "add_v";
};

template <typename T = void>
struct minus : std::minus<T> {
    static constexpr auto name = "sub_v";
};

template <typename T = void>
struct multiplies : std::multiplies<T> {
    static constexpr auto name = "mul_v";
};

template <typename T = void>
struct divides : std::divides<T> {
    static constexpr auto name = "div_v";
};

template <typename T = void>
struct power : std::binary_function<T,T,T> {
    static constexpr auto name = "pow";
    T operator()(const T& x, const T& y) const {
        return std::pow(x, y);
    }
};

template <typename T = void>
struct modulus : std::binary_function<T, T, T> {
    static constexpr auto name = "mod";
    T operator()(const T& x, const T& y) const {
        return x % y;
    }
};

template <>
inline float modulus<float>::operator()(const float& x, const float& y) const {
    return std::fmod(x, y);
}

template <>
inline double modulus<double>::operator()(const double& x, const double& y) const {
    return std::fmod(x, y);
}

template <>
struct modulus<void> {
    static constexpr auto name = "mod";

    template <class T1, class T2, std::enable_if_t<
        !(std::is_floating_point<T1>::value || std::is_floating_point<T2>::value), int> = 0>
    constexpr auto operator()(T1&& x, T2&& y) const
    noexcept(noexcept(std::forward<T1>(x) % std::forward<T2>(y)))
    -> decltype      (std::forward<T1>(x) % std::forward<T2>(y))
        { return      std::forward<T1>(x) % std::forward<T2>(y); }

    template <class T1, class T2, std::enable_if_t<
        std::is_floating_point<T1>::value || std::is_floating_point<T2>::value, int> = 0>
    constexpr auto operator()(T1 x, T2 y) const
    noexcept(noexcept(std::fmod(x, y)))
    -> decltype      (std::fmod(x, y))
        { return      std::fmod(x, y); }
};

template <>
struct power<void> {
    static constexpr auto name = "pow";
    template <class T1, class T2>
    auto operator()(T1&& x, T2&& y) const
    noexcept(noexcept(std::pow(std::forward<T1>(x), std::forward<T2>(y))))
    -> decltype      (std::pow(std::forward<T1>(x), std::forward<T2>(y)))
        { return      std::pow(std::forward<T1>(x), std::forward<T2>(y)); }
};

template <typename T, typename Compare = std::less<>>
struct max : std::binary_function<T,T,T> {
    static constexpr auto name = "max";
    const Compare comp{};
    constexpr max() = default;
    constexpr max(Compare comp) : comp(comp) {}
    constexpr T operator()(const T& x, const T& y) const {
        return std::max(x, y, comp);
    }
};

template <typename T, typename Compare = std::less<>>
struct min : std::binary_function<T,T,T> {
    static constexpr auto name = "min";
    const Compare comp{};
    constexpr min() = default;
    constexpr min(Compare comp) : comp(comp) {}
    constexpr T operator()(const T& x, const T& y) const {
        return std::min(x, y, comp);
    }
};

//==-------------------------------------------------------------------------

template <typename T = void>
struct bit_and : std::bit_and<T> {
    static constexpr auto name = "bit_and";
};

template <typename T = void>
struct bit_or : std::bit_or<T> {
    static constexpr auto name = "bit_or";
};

template <typename T = void>
struct bit_xor : std::bit_xor<T> {
    static constexpr auto name = "bit_xor";
};

template <typename T = void>
struct bit_not : std::bit_not<T> {
    static constexpr auto name = "bit_not";
};

template <typename T = void>
struct logical_and : std::logical_and<T> {
    static constexpr auto name = "logical_and";
};

template <typename T = void>
struct logical_or : std::logical_or<T> {
    static constexpr auto name = "logical_or";
};

template <typename T = void>
struct logical_not : std::logical_not<T> {
    static constexpr auto name = "logical_not";
};

//==-------------------------------------------------------------------------

template <typename T>
struct clip : std::unary_function<T,T>, parameterized_function<T> {
    static constexpr auto name = "clip";
    constexpr clip(const T& min, const T& max)
        : parameterized_function<T>(min, max) {}
    constexpr T operator()(const T& x) const {
        return cxx::clamp(x, this->alpha, this->beta);
    }
};

template <typename T>
struct shrink : std::unary_function<T,T>, parameterized_function<T> {
    static constexpr auto name = "shrink";
    constexpr shrink(const T& lambd, const T& bias)
        : parameterized_function<T>(lambd, bias) {}
    const T operator()(const T& x) const {
        return x < -this->alpha ? x + this->beta :
               x >  this->alpha ? x - this->beta : 0;
    }
};

template <typename T>
struct relu : std::unary_function<T,T>, parameterized_function<T> {
    static constexpr auto name = "relu";
    constexpr T operator()(const T& x) const {
        return std::max(T(), x);
    }
};

template <typename T>
struct prelu : std::binary_function<T,T,T> {
    static constexpr auto name = "prelu";
    constexpr T operator()(const T& x, const T& slope) const {
        return x < T() ? x*slope : x;
    }
};

template <typename T>
struct leaky_relu : std::unary_function<T,T>, parameterized_function<T> {
    static constexpr auto name = "leaky_relu";
    constexpr leaky_relu(const T& alpha)
        : parameterized_function<T>(alpha) {}
    constexpr T operator()(const T& x) const {
        return x < T() ? x*this->alpha : x;
    }
};

template <typename T>
struct thresholded_relu : std::unary_function<T,T>, parameterized_function<T> {
    static constexpr auto name = "thresholded_relu";
    constexpr thresholded_relu(const T& alpha)
        : parameterized_function<T>(alpha) {}
    constexpr T operator()(const T& x) const {
        return x > this->alpha ? x : T();
    }
};

template <typename T>
struct selu : std::unary_function<T,T>, parameterized_function<T> {
    static constexpr auto name = "selu";
    selu(const T& alpha, const T& gamma)
        : parameterized_function<T>(alpha, gamma) {}
    constexpr T operator()(const T& x) const {
        return this->beta * (x < T() ? this->alpha*(std::exp(x) - T(1)) : x);
    }
};

template <typename T>
struct elu : std::unary_function<T,T>, parameterized_function<T> {
    static constexpr auto name = "elu";
    elu(const T& alpha) : parameterized_function<T>(alpha) {}
    constexpr T operator()(const T& x) const {
        return x < T() ? this->alpha*(std::exp(x) - T(1)) : x;
    }
};

template <typename T>
struct hard_sigmoid : std::unary_function<T,T>, parameterized_function<T> {
    static constexpr auto name = "hard_sigmoid";
    constexpr hard_sigmoid(const T& alpha, const T& beta)
        : parameterized_function<T>(alpha, beta) {}
    constexpr T operator()(const T& x) const {
        return std::max(T(), std::min(T(1), this->alpha*x + this->beta));
    }
};

template <typename T>
struct softsign : std::unary_function<T,T>, parameterized_function<T> {
    static constexpr auto name = "softsign";
    constexpr T operator()(const T& x) const {
        return x / (T(1) + std::abs(x));
    }
};

template <typename T>
struct softplus : std::unary_function<T,T>, parameterized_function<T> {
    static constexpr auto name = "softplus";
    constexpr T operator()(const T& x) const {
        return std::log(std::exp(x) + T(1));
    }
};

}} // namespace dlf::xfn
