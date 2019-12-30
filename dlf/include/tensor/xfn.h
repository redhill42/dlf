#pragma once

namespace dlf { namespace xfn {

template <typename F = void>
struct transfer {
    F f;
    constexpr transfer() : f(F()) {}
    constexpr explicit transfer(F f) : f(f) {}
    template <typename R, typename... Args>
    void operator()(R& y, Args&&... args) const
    noexcept(noexcept(y = f(std::forward<Args>(args)...)))
        { y = f(std::forward<Args>(args)...); }
};

template <>
struct transfer<void> {
    template <typename T>
    void operator()(T& y, T&& x) const
    noexcept(noexcept(y = std::forward<T>(x)))
        { y = std::forward<T>(x); }
};

template <typename T>
std::enable_if_t<std::is_literal_type<T>::value, T>
inline constexpr zero() { return T{}; }

template <typename T>
std::enable_if_t<!std::is_literal_type<T>::value, const T&>
inline zero() { static T zero{}; return zero; }

template <typename T>
std::enable_if_t<std::is_literal_type<T>::value, T>
inline constexpr one() { return T{1}; }

template <typename T>
std::enable_if_t<!std::is_literal_type<T>::value, const T&>
inline one() { static T one{1}; return one; }

template <typename T>
std::enable_if_t<std::is_literal_type<T>::value, T>
inline constexpr neg_one() { return T{-1}; }

template <typename T>
std::enable_if_t<!std::is_literal_type<T>::value, const T&>
inline neg_one() { static T neg_one{-1}; return neg_one; }

//==-------------------------------------------------------------------------

template <typename T>
struct parameterized_function {
    const T alpha = zero<T>(), beta = zero<T>();
    constexpr parameterized_function() = default;
    constexpr explicit parameterized_function(const T& alpha)
        : alpha(alpha) {}
    constexpr explicit parameterized_function(const T& alpha, const T& beta)
        : alpha(alpha), beta(beta) {}
};

template <typename T = void>
struct identity : std::unary_function<T,T> {
    constexpr T operator()(const T& x) const noexcept { return x; }
};

template <>
struct identity<void> {
    template <typename T>
    constexpr T operator()(T&& x) const noexcept
        { return std::forward<T>(x); }
};

template <typename T = void>
struct abs : std::unary_function<T,T> {
    constexpr T operator()(const T& x) const
        { return std::abs(x); }
};

template <>
struct abs<void> {
    template <typename T>
    constexpr auto operator()(T&& x) const
    noexcept(noexcept(std::abs(std::forward<T>(x))))
    -> decltype      (std::abs(std::forward<T>(x)))
        { return      std::abs(std::forward<T>(x)); }
};

template <typename T> inline constexpr const char*
function_kernel_name(abs<T>) { return "abs"; }

template <typename T = void>
struct negate : std::negate<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::negate<T>) { return "neg"; }

template <typename T = void>
struct sign : std::unary_function<T,T> {
    constexpr T operator()(const T& x) const
        { return (zero<T>() < x) - (x < zero<T>()); }
};

template <>
struct sign<void> {
    template <typename T>
    constexpr auto operator()(T&& x) const
    noexcept(noexcept((zero<T>() < x) - (x < zero<T>())))
    -> decltype      ((zero<T>() < x) - (x < zero<T>()))
        { return      (zero<T>() < x) - (x < zero<T>()); }
};

template <typename T> inline constexpr const char*
function_kernel_name(sign<T>) { return "sign"; }

template <typename T = void>
struct conj : std::unary_function<T,T> {
    constexpr T operator()(const T& x) const
        { return x; }
};

template <typename T>
struct conj<std::complex<T>> {
    constexpr std::complex<T> operator()(const std::complex<T>& x) const
        { return std::conj(x); }
};

template <>
struct conj<void> {
    template <typename T>
    constexpr T operator()(T&& x) const noexcept
        { return std::forward<T>(x); }
    template <typename T>
    constexpr auto operator()(const std::complex<T>& x) const
    noexcept(noexcept(std::conj(x)))
    -> decltype      (std::conj(x))
        { return      std::conj(x); }
};

template <typename T> inline constexpr const char*
function_kernel_name(conj<T>) { return "conj"; }

template <typename T = void>
struct norm : std::unary_function<T,T> {
    constexpr T operator()(const T& x) const
        { return std::norm(x); }
};

template <>
struct norm<void> {
    template <typename T>
    constexpr auto operator()(T&& x) const
    noexcept(noexcept(std::norm(x)))
    -> decltype      (std::norm(x))
        { return      std::norm(x); }
};

template <typename T> inline constexpr const char*
function_kernel_name(norm<T>) { return "norm"; }

template <typename T>
struct chop : std::unary_function<T,T> {
    constexpr chop() = default;
    constexpr explicit chop(double) {}
    constexpr T operator()(const T& x) const { return x; }
    constexpr T operator()(T&& x) const { return std::move(x); }
};

template <>
struct chop<float> : std::unary_function<float, float> {
    const float tolerance = 1e-5f;
    chop() = default;
    explicit chop(float tol) : tolerance(tol) {}
    float operator()(float x) const
        { return std::abs(x) < tolerance ? 0.f : x; }
};

template <>
struct chop<double> : std::unary_function<double, double> {
    const double tolerance = 1e-10;
    chop() = default;
    explicit chop(double tol) : tolerance(tol) {}
    double operator()(double x) const
        { return std::abs(x) < tolerance ? 0.0 : x; }
};

template <>
struct chop<std::complex<float>> : std::unary_function<std::complex<float>, std::complex<float>> {
    const float tolerance = 1e-5f;
    chop() = default;
    explicit chop(float tol) : tolerance(tol) {}

    std::complex<float> operator()(const std::complex<float>& x) const {
        float re = std::abs(x.real()), im = std::abs(x.imag());
        if (re < tolerance || im < tolerance) {
            return std::complex<float>(re < tolerance ? 0.f : x.real(),
                                       im < tolerance ? 0.f : x.imag());
        } else {
            return x;
        }
    }
};

template <>
struct chop<std::complex<double>> : std::unary_function<std::complex<double>, std::complex<double>> {
    const double tolerance = 1e-10;
    chop() = default;
    explicit chop(double tol) : tolerance(tol) {}

    std::complex<double> operator()(const std::complex<double>& x) const {
        double re = std::abs(x.real()), im = std::abs(x.imag());
        if (re < tolerance || im < tolerance) {
            return std::complex<double>(re < tolerance ? 0.0 : x.real(),
                                        im < tolerance ? 0.0 : x.imag());
        } else {
            return x;
        }
    }
};

template <typename T> inline constexpr const char*
function_kernel_name(chop<T>) { return "chop"; }

#define DEFINE_UNARY_FUNCTION(fn, op) \
template <typename T> \
struct fn : std::unary_function<T,T> { \
    T operator()(const T& x) const { op; } \
}; \
template <typename T> inline constexpr const char* \
function_kernel_name(fn<T>) { return #fn; }

DEFINE_UNARY_FUNCTION(recip,    return one<T>()/x)
DEFINE_UNARY_FUNCTION(floor,    using std::floor; return floor(x))
DEFINE_UNARY_FUNCTION(ceil,     using std::ceil; return ceil(x))
DEFINE_UNARY_FUNCTION(round,    using std::round; return round(x))
DEFINE_UNARY_FUNCTION(sqrt,     using std::sqrt; return sqrt(x))
DEFINE_UNARY_FUNCTION(square,   return x*x)
DEFINE_UNARY_FUNCTION(exp,      using std::exp; return exp(x))
DEFINE_UNARY_FUNCTION(log,      using std::log; return log(x))
DEFINE_UNARY_FUNCTION(sin,      using std::sin; return sin(x))
DEFINE_UNARY_FUNCTION(cos,      using std::cos; return cos(x))
DEFINE_UNARY_FUNCTION(tan,      using std::tan; return tan(x))
DEFINE_UNARY_FUNCTION(asin,     using std::asin; return asin(x))
DEFINE_UNARY_FUNCTION(acos,     using std::acos; return acos(x))
DEFINE_UNARY_FUNCTION(atan,     using std::atan; return atan(x))
DEFINE_UNARY_FUNCTION(sinh,     using std::sinh; return sinh(x))
DEFINE_UNARY_FUNCTION(cosh,     using std::cosh; return cosh(x))
DEFINE_UNARY_FUNCTION(tanh,     using std::tanh; return tanh(x))
DEFINE_UNARY_FUNCTION(asinh,    using std::asinh; return asinh(x))
DEFINE_UNARY_FUNCTION(acosh,    using std::acosh; return acosh(x))
DEFINE_UNARY_FUNCTION(atanh,    using std::atanh; return atanh(x))
DEFINE_UNARY_FUNCTION(erf,      using std::erf; return erf(x))
DEFINE_UNARY_FUNCTION(sigmoid,  using std::exp; return one<T>()/(one<T>()+exp(-x)))
DEFINE_UNARY_FUNCTION(softsign, using std::abs; return x/(one<T>()+abs(x)))
DEFINE_UNARY_FUNCTION(softplus, using std::log; using std::exp; return log(exp(x)+one<T>()))

#undef DEFINE_UNARY_FUNCTION

//==-------------------------------------------------------------------------

template <typename T = void>
struct plus : std::plus<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::plus<T>) { return "add"; }
template <typename T> inline constexpr const char*
scan_kernel_name(std::plus<T>) { return "cumsum"; }

template <typename T = void>
struct minus : std::minus<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::minus<T>) { return "sub"; }

template <typename T = void>
struct multiplies : std::multiplies<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::multiplies<T>) { return "mul"; }
template <typename T> inline constexpr const char*
scan_kernel_name(std::multiplies<T>) { return "cumprod"; }

template <typename T = void>
struct divides : std::divides<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::divides<T>) { return "div"; }

template <typename T = void>
struct power : std::binary_function<T,T,T> {
    T operator()(const T& x, const T& y) const
        { return std::pow(x, y); }
};

template <>
struct power<void> {
    template <class T1, class T2>
    auto operator()(T1&& x, T2&& y) const
    noexcept(noexcept(std::pow(std::forward<T1>(x), std::forward<T2>(y))))
    -> decltype      (std::pow(std::forward<T1>(x), std::forward<T2>(y)))
        { return      std::pow(std::forward<T1>(x), std::forward<T2>(y)); }
};

template <typename T> inline constexpr const char*
function_kernel_name(power<T>) { return "pow"; }

template <typename T = void>
struct modulus : std::binary_function<T, T, T> {
    T operator()(const T& x, const T& y) const
        { return x % y; }
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

template <typename T> inline constexpr const char*
function_kernel_name(modulus<T>) { return "mod"; }

template <typename T, typename Compare = std::less<>>
struct max : std::binary_function<T,T,T> {
    const Compare comp{};
    constexpr max() = default;
    constexpr max(Compare comp) : comp(comp) {}
    constexpr T operator()(const T& x, const T& y) const
        { return std::max(x, y, comp); }
};

template <typename T, typename Compare> inline constexpr const char*
function_kernel_name(max<T, Compare>) { return "max"; }
template <typename T, typename Compare> inline constexpr const char*
scan_kernel_name(max<T, Compare>) { return "cummax"; }

template <typename T, typename Compare = std::less<>>
struct min : std::binary_function<T,T,T> {
    const Compare comp{};
    constexpr min() = default;
    constexpr min(Compare comp) : comp(comp) {}
    constexpr T operator()(const T& x, const T& y) const
        { return std::min(x, y, comp); }
};

template <typename T, typename Compare> inline constexpr const char*
function_kernel_name(min<T, Compare>) { return "min"; }
template <typename T, typename Compare> inline constexpr const char*
scan_kernel_name(min<T, Compare>) { return "cummin"; }

template <typename T, typename Compare = std::less<>>
struct amax : std::binary_function<T,T,T> {
    const Compare comp{};
    constexpr amax() = default;
    constexpr amax(Compare comp) : comp(comp) {}
    constexpr T operator()(const T& x, const T& y) const
        { return comp(std::abs(x), std::abs(y)) ? y : x; }
};

template <typename T, typename Compare = std::less<>>
struct amin : std::binary_function<T,T,T> {
    const Compare comp{};
    constexpr amin() = default;
    constexpr amin(Compare comp) : comp(comp) {}
    constexpr T operator()(const T& x, const T& y) const
        { return comp(std::abs(x), std::abs(y)) ? x : y; }
};

//==-------------------------------------------------------------------------

template <typename T = void>
struct equal_to : std::equal_to<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::equal_to<T>) { return "equal_to"; }

template <typename T = void>
struct not_equal_to : std::not_equal_to<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::not_equal_to<T>) { return "not_equal_to"; }

template <typename T = void>
struct less : std::less<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::less<T>) { return "less"; }

template <typename T = void>
struct less_equal : std::less_equal<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::less_equal<T>) { return "less_equal"; }

template <typename T = void>
struct greater : std::greater<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::greater<T>) { return "greater"; }

template <typename T = void>
struct greater_equal : std::greater_equal<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::greater_equal<T>) { return "greater_equal"; }

//==-------------------------------------------------------------------------

template <typename T = void>
struct bit_and : std::bit_and<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::bit_and<T>) { return "bit_and"; }

template <typename T = void>
struct bit_or : std::bit_or<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::bit_or<T>) { return "bit_or"; }

template <typename T = void>
struct bit_xor : std::bit_xor<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::bit_xor<T>) { return "bit_xor"; }

template <typename T = void>
struct bit_not : std::bit_not<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::bit_not<T>) { return "bit_not"; }

template <typename T = void>
struct logical_and : std::logical_and<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::logical_and<T>) { return "logical_and"; }

template <typename T = void>
struct logical_or : std::logical_or<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::logical_or<T>) { return "logical_or"; }

template <typename T = void>
struct logical_not : std::logical_not<T> {};

template <typename T> inline constexpr const char*
function_kernel_name(std::logical_not<T>) { return "logical_not"; }

//==-------------------------------------------------------------------------

template <typename T>
struct clip : std::unary_function<T,T>, parameterized_function<T> {
    constexpr clip(const T& min, const T& max)
        : parameterized_function<T>(min, max) {}
    constexpr T operator()(const T& x) const {
        return cxx::clamp(x, this->alpha, this->beta);
    }
};

template <typename T> inline constexpr const char*
function_kernel_name(clip<T>) { return "clip"; }

template <typename T>
struct shrink : std::unary_function<T,T>, parameterized_function<T> {
    constexpr shrink(const T& lambd, const T& bias)
        : parameterized_function<T>(lambd, bias) {}
    const T operator()(const T& x) const {
        return x < -this->alpha ? x + this->beta :
               x >  this->alpha ? x - this->beta : zero<T>();
    }
};

template <typename T> inline constexpr const char*
function_kernel_name(shrink<T>) { return "shrink"; }

template <typename T>
struct relu : std::unary_function<T,T>, parameterized_function<T> {
    constexpr T operator()(const T& x) const
        { return std::max(zero<T>(), x); }
};

template <typename T> inline constexpr const char*
function_kernel_name(relu<T>) { return "relu"; }

template <typename T>
struct prelu : std::binary_function<T,T,T> {
    constexpr T operator()(const T& x, const T& slope) const
        { return x < zero<T>() ? x*slope : x; }
};

template <typename T> inline constexpr const char*
function_kernel_name(prelu<T>) { return "prelu"; }

template <typename T>
struct leaky_relu : std::unary_function<T,T>, parameterized_function<T> {
    constexpr leaky_relu(const T& alpha)
        : parameterized_function<T>(alpha) {}
    constexpr T operator()(const T& x) const
        { return x < zero<T>() ? x*this->alpha : x; }
};

template <typename T> inline constexpr const char*
function_kernel_name(leaky_relu<T>) { return "leaky_relu"; }

template <typename T>
struct thresholded_relu : std::unary_function<T,T>, parameterized_function<T> {
    constexpr thresholded_relu(const T& alpha)
        : parameterized_function<T>(alpha) {}
    constexpr T operator()(const T& x) const
        { return x > this->alpha ? x : zero<T>(); }
};

template <typename T> inline constexpr const char*
function_kernel_name(thresholded_relu<T>) { return "thresholded_relu"; }

template <typename T>
struct selu : std::unary_function<T,T>, parameterized_function<T> {
    selu(const T& alpha, const T& gamma)
        : parameterized_function<T>(alpha, gamma) {}
    constexpr T operator()(const T& x) const
        { return this->beta * (x < zero<T>() ? this->alpha*(std::exp(x) - one<T>()) : x); }
};

template <typename T> inline constexpr const char*
function_kernel_name(selu<T>) { return "selu"; }

template <typename T>
struct elu : std::unary_function<T,T>, parameterized_function<T> {
    elu(const T& alpha) : parameterized_function<T>(alpha) {}
    constexpr T operator()(const T& x) const
        { return x < zero<T>() ? this->alpha*(std::exp(x) - one<T>()) : x; }
};

template <typename T> inline constexpr const char*
function_kernel_name(elu<T>) { return "elu"; }

template <typename T>
struct hard_sigmoid : std::unary_function<T,T>, parameterized_function<T> {
    constexpr hard_sigmoid(const T& alpha, const T& beta)
        : parameterized_function<T>(alpha, beta) {}
    constexpr T operator()(const T& x) const
        { return cxx::clamp(this->alpha*x + this->beta, zero<T>(), one<T>()); }
};

template <typename T> inline constexpr const char*
function_kernel_name(hard_sigmoid<T>) { return "hard_sigmoid"; }

//==-------------------------------------------------------------------------

template <typename F>
struct post_reduce {
    F f;
    template <typename T>
    constexpr T operator()(const T& acc, const int) const
    noexcept(noexcept(f(acc))) { return f(acc); }
};

template <typename T>
struct post_reduce_identity {
    constexpr T operator()(const T& acc, const int) const noexcept { return acc; }
};

template <typename T>
struct post_reduce_average {
    constexpr T operator()(const T& acc, const int n) const
    noexcept(noexcept(acc / static_cast<T>(n)))
        { return acc / static_cast<T>(n); }
};

template <typename T>
struct reduce_max {
    static constexpr T identity() { return std::numeric_limits<T>::lowest(); }
    using Map = xfn::identity<T>;
    using Reduce = xfn::max<T>;
    using Final = post_reduce_identity<T>;
};

template <typename T> inline constexpr const char*
reduction_kernel_name(reduce_max<T>) { return "reduce_max"; }

template <typename T>
struct reduce_amax {
    static constexpr T identity() { return zero<T>(); }
    using Map = xfn::abs<T>;
    using Reduce = xfn::amax<T>;
    using Final = post_reduce_identity<T>;
};

template <typename T> inline constexpr const char*
reduction_kernel_name(reduce_amax<T>) { return "reduce_amax"; }

template <typename T>
struct reduce_min {
    static constexpr T identity() { return std::numeric_limits<T>::max(); }
    using Map = xfn::identity<T>;
    using Reduce = xfn::min<T>;
    using Final = post_reduce_identity<T>;
};

template <typename T> inline constexpr const char*
reduction_kernel_name(reduce_min<T>) { return "reduce_min"; }

template <typename T>
struct reduce_amin {
    static constexpr T identity() { return std::numeric_limits<T>::max(); }
    using Map = xfn::abs<T>;
    using Reduce = xfn::amin<T>;
    using Final = post_reduce_identity<T>;
};

template <typename T> inline constexpr const char*
reduction_kernel_name(reduce_amin<T>) { return "reduce_amin"; }

template <typename T>
struct reduce_sum {
    static constexpr T identity() { return zero<T>(); }
    using Map = xfn::identity<T>;
    using Reduce = xfn::plus<T>;
    using Final = post_reduce_identity<T>;
};

template <typename T> inline constexpr const char*
reduction_kernel_name(reduce_sum<T>) { return "reduce_sum"; }

template <typename T>
struct reduce_asum : reduce_sum<T> {
    using Map = xfn::abs<T>;
};

template <typename T> inline constexpr const char*
reduction_kernel_name(reduce_asum<T>) { return "reduce_asum"; }

template <typename T>
struct reduce_mean : reduce_sum<T> {
    using Final = post_reduce_average<T>;
};

template <typename T> inline constexpr const char*
reduction_kernel_name(reduce_mean<T>) { return "reduce_mean"; }

template <typename T>
struct reduce_sum_square : reduce_sum<T> {
    using Map = xfn::square<T>;
};

template <typename T> inline constexpr const char*
reduction_kernel_name(reduce_sum_square<T>) { return "reduce_sum_square"; }

template <typename T>
struct reduce_nrm2 {
    static constexpr T identity() { return zero<T>(); }
    using Map = xfn::norm<T>;
    using Reduce = xfn::plus<T>;
    using Final = post_reduce<xfn::sqrt<T>>;
};

template <typename T> inline constexpr const char*
reduction_kernel_name(reduce_nrm2<T>) { return "reduce_nrm2"; }

template <typename T>
struct reduce_log_sum : reduce_sum<T> {
    using Final = post_reduce<xfn::log<T>>;
};

template <typename T> inline constexpr const char*
reduction_kernel_name(reduce_log_sum<T>) { return "reduce_log_sum"; }

template <typename T>
struct reduce_sum_exp : reduce_sum<T> {
    using Map = xfn::exp<T>;
};

template <typename T> inline constexpr const char*
reduction_kernel_name(reduce_sum_exp<T>) { return "reduce_sum_exp"; }

template <typename T>
struct reduce_log_sum_exp : reduce_sum_exp<T> {
    static constexpr T identity() { return zero<T>(); }
    using Final = post_reduce<xfn::log<T>>;
};

template <typename T> inline constexpr const char*
reduction_kernel_name(reduce_log_sum_exp<T>) { return "reduce_log_sum_exp"; }

template <typename T>
struct reduce_prod {
    static constexpr T identity() { return one<T>(); }
    using Map = xfn::identity<T>;
    using Reduce = xfn::multiplies<T>;
    using Final = post_reduce_identity<T>;
};

template <typename T> inline constexpr const char*
reduction_kernel_name(reduce_prod<T>) { return "reduce_prod"; }

}} // namespace dlf::xfn
