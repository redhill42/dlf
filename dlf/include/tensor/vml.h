#pragma once

#if HAS_MKL

struct mkl_basic_types {
    template <typename T, typename... Ts>
    static constexpr bool is_type_supported() {
        return cxx::disjunction<std::is_same<T, float>,
                                std::is_same<T, double>>::value &&
               cxx::conjunction<std::is_same<T, Ts>...>::value;
    }
};

struct mkl_all_types {
    template <typename T, typename... Ts>
    static constexpr bool is_type_supported() {
        return cxx::disjunction<std::is_same<T, float>,
                                std::is_same<T, double>,
                                std::is_same<T, std::complex<float>>,
                                std::is_same<T, std::complex<double>>>::value &&
               cxx::conjunction<std::is_same<T, Ts>...>::value;
    }
};

template <typename Types = mkl_all_types>
struct mkl_map_impl {
    template <typename... Args>
    std::enable_if_t<
        cxx::conjunction<is_tensor<Args>...>::value &&
        Types::template is_type_supported<std::remove_const_t<tensor_value_type<Args>>...>(),
        bool>
    static constexpr is_prefer_serial(Args&&... args) {
        return cxx::all_of([](auto&& x){ return x.shape().is_contiguous(); }, std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::enable_if_t<
        !cxx::conjunction<is_tensor<Args>...>::value ||
        !Types::template is_type_supported<std::remove_const_t<tensor_value_type<Args>>...>(),
        bool>
    static constexpr is_prefer_serial(Args&&...) {
        return false;
    }
};

//==-------------------------------------------------------------------------
// Binary Functions
//==-------------------------------------------------------------------------

#define DEFINE_VML_BINARY_FUNCTION(name, fn) \
template <typename T> \
struct map_impl<xfn::transfer<fn<T>>> : mkl_map_impl<> { \
    template <typename F> \
    static inline void vx##name(F, size_t n, float* y, const float* a, const float* b) \
        { ::vs##name(n, a, b, y); } \
    template <typename F> \
    static inline void vx##name(F, size_t n, double* y, const double* a, const double* b) \
        { ::vd##name(n, a, b, y); } \
    template <typename F> \
    static inline void vx##name(F, size_t n, std::complex<float>* y, const std::complex<float>* a, const std::complex<float>* b) \
        { ::vc##name(n, reinterpret_cast<const MKL_Complex8*>(a), reinterpret_cast<const MKL_Complex8*>(b), reinterpret_cast<MKL_Complex8*>(y)); } \
    template <typename F> \
    static inline void vx##name(F, size_t n, std::complex<double>* y, const std::complex<double>* a, const std::complex<double>* b) \
        { ::vz##name(n, reinterpret_cast<const MKL_Complex16*>(a), reinterpret_cast<const MKL_Complex16*>(b), reinterpret_cast<MKL_Complex16*>(y)); } \
    template <typename F, typename A, typename B, typename Y> \
    static inline void vx##name(F f, size_t n, Y y, A a, B b) \
        { std::transform(a, a + n, b, y, f); } \
    template <typename A, typename B, typename Y> \
    void operator()(xfn::transfer<fn<T>> f, size_t n, Y y, A a, B b) \
        { vx##name(f.f, n, y, a, b); } \
    template <typename A, typename B, typename Y> \
    void operator()(xfn::transfer<fn<T>> f, size_t n, Y* y, A* a, B* b) \
        { vx##name(f.f, n, y, static_cast<std::add_const_t<A>*>(a), static_cast<std::add_const_t<B>*>(b)); } \
};

DEFINE_VML_BINARY_FUNCTION(Add, std::plus)
DEFINE_VML_BINARY_FUNCTION(Add, xfn::plus)
DEFINE_VML_BINARY_FUNCTION(Sub, std::minus)
DEFINE_VML_BINARY_FUNCTION(Sub, xfn::minus)
DEFINE_VML_BINARY_FUNCTION(Mul, std::multiplies)
DEFINE_VML_BINARY_FUNCTION(Mul, xfn::multiplies)
DEFINE_VML_BINARY_FUNCTION(Div, std::divides)
DEFINE_VML_BINARY_FUNCTION(Div, xfn::divides)
DEFINE_VML_BINARY_FUNCTION(Pow, xfn::power)
#undef DEFINE_VML_BINARY_FUNCTION

//==-------------------------------------------------------------------------
// Unary Functions
//==-------------------------------------------------------------------------

#define DEFINE_VML_UNARY_FUNCTION(name, fn) \
template <typename T> \
struct map_impl<xfn::transfer<fn<T>>> : mkl_map_impl<> { \
    template <typename F> \
    inline static void vx##name(F, size_t n, float* y, const float* a) \
        { ::vs##name(n, a, y); } \
    template <typename F> \
    inline static void vx##name(F, size_t n, double* y, const double* a) \
        { ::vd##name(n, a, y); } \
    template <typename F> \
    inline static void vx##name(F, size_t n, std::complex<float>* y, const std::complex<float>* a) \
        { ::vc##name(n, reinterpret_cast<const MKL_Complex8*>(a), reinterpret_cast<MKL_Complex8*>(y)); } \
    template <typename F> \
    inline static void vx##name(F, size_t n, std::complex<double>* y, const std::complex<double>* a) \
        { ::vz##name(n, reinterpret_cast<const MKL_Complex16*>(a), reinterpret_cast<MKL_Complex16*>(y)); } \
    template <typename F, typename A, typename Y> \
    inline static void vx##name(F f, size_t n, Y y, A a) \
        { std::transform(a, a + n, y, f); } \
    template <typename A, typename Y> \
    void operator()(xfn::transfer<fn<T>> f, size_t n, Y y, A a) \
        { vx##name(f.f, n, y, a); } \
    template <typename A, typename Y> \
    void operator()(xfn::transfer<fn<T>> f, size_t n, Y* y, A* a) \
        { vx##name(f.f, n, y, static_cast<std::add_const_t<A>*>(a)); } \
};

DEFINE_VML_UNARY_FUNCTION(Exp, xfn::exp)
DEFINE_VML_UNARY_FUNCTION(Ln, xfn::log)
DEFINE_VML_UNARY_FUNCTION(Sqrt, xfn::sqrt)
DEFINE_VML_UNARY_FUNCTION(Sin, xfn::sin)
DEFINE_VML_UNARY_FUNCTION(Cos, xfn::cos)
DEFINE_VML_UNARY_FUNCTION(Tan, xfn::tan)
DEFINE_VML_UNARY_FUNCTION(Asin, xfn::asin)
DEFINE_VML_UNARY_FUNCTION(Acos, xfn::acos)
DEFINE_VML_UNARY_FUNCTION(Atan, xfn::atan)
DEFINE_VML_UNARY_FUNCTION(Sinh, xfn::sinh)
DEFINE_VML_UNARY_FUNCTION(Cosh, xfn::cosh)
DEFINE_VML_UNARY_FUNCTION(Tanh, xfn::tanh)
DEFINE_VML_UNARY_FUNCTION(Asinh, xfn::asinh)
DEFINE_VML_UNARY_FUNCTION(Acosh, xfn::acosh)
DEFINE_VML_UNARY_FUNCTION(Atanh, xfn::atanh)
#undef DEFINE_VML_UNARY_FUNCTION

//==-------------------------------------------------------------------------
// Unary Functions without complex type
//==-------------------------------------------------------------------------

#define DEFINE_VML_UNARY_FUNCTION(name, fn) \
template <typename T> \
struct map_impl<xfn::transfer<fn<T>>> : mkl_map_impl<mkl_basic_types> { \
    template <typename F> \
    inline static void vx##name(F, size_t n, float* y, const float* a) \
        { ::vs##name(n, a, y); } \
    template <typename F> \
    inline static void vx##name(F, size_t n, double* y, const double* a) \
        { ::vd##name(n, a, y); } \
    template <typename F, typename A, typename Y> \
    inline static void vx##name(F f, size_t n, Y y, A a) \
        { std::transform(a, a + n, y, f); } \
    template <typename A, typename Y> \
    void operator()(xfn::transfer<fn<T>> f, size_t n, Y y, A a) \
        { vx##name(f.f, n, y, a); } \
    template <typename A, typename Y> \
    void operator()(xfn::transfer<fn<T>> f, size_t n, Y* y, A* a) \
        { vx##name(f.f, n, y, static_cast<std::add_const_t<A>*>(a)); } \
};

DEFINE_VML_UNARY_FUNCTION(Abs, xfn::abs)
DEFINE_VML_UNARY_FUNCTION(Inv, xfn::reciprocal)
DEFINE_VML_UNARY_FUNCTION(Sqr, xfn::square)
DEFINE_VML_UNARY_FUNCTION(Floor, xfn::floor)
DEFINE_VML_UNARY_FUNCTION(Ceil, xfn::ceil)
DEFINE_VML_UNARY_FUNCTION(Round, xfn::round)
DEFINE_VML_UNARY_FUNCTION(Erf, xfn::erf)
#undef DEFINE_VML_UNARY_FUNCTION

#elif defined(__APPLE__)

#define DEFINE_VML_UNARY_FUNCTION(name, fn) \
template <typename T> \
struct map_impl<xfn::transfer<fn<T>>> : map_impl_base { \
    template <typename F> \
    static inline void vx##name(F, const int n, float* y, const float* a) \
        { ::vv##name##f(y, a, &n); } \
    template <typename F> \
    static inline void vx##name(F, const int n, double* y, const double* a) \
        { ::vv##name(y, a, &n); } \
    template <typename F, typename A, typename Y> \
    static inline void vx##name(F f, const int n, Y y, A a) \
        { std::transform(a, a + n, y, f); } \
    template <typename A, typename Y> \
    void operator()(xfn::transfer<fn<T>> f, size_t n, Y y, A a) \
        { vx##name(f.f, n, y, a); } \
    template <typename A, typename Y> \
    void operator()(xfn::transfer<fn<T>> f, size_t n, Y* y, A* a) \
        { vx##name(f.f, n, y, static_cast<std::add_const_t<A>*>(a)); } \
};

DEFINE_VML_UNARY_FUNCTION(rec, xfn::reciprocal)
DEFINE_VML_UNARY_FUNCTION(sqrt, xfn::sqrt)
DEFINE_VML_UNARY_FUNCTION(exp, xfn::exp)
DEFINE_VML_UNARY_FUNCTION(log, xfn::log)
DEFINE_VML_UNARY_FUNCTION(fabs, xfn::abs)
DEFINE_VML_UNARY_FUNCTION(sin, xfn::sin)
DEFINE_VML_UNARY_FUNCTION(cos, xfn::cos)
DEFINE_VML_UNARY_FUNCTION(tan, xfn::tan)
DEFINE_VML_UNARY_FUNCTION(asin, xfn::asin)
DEFINE_VML_UNARY_FUNCTION(acos, xfn::acos)
DEFINE_VML_UNARY_FUNCTION(atan, xfn::atan)
DEFINE_VML_UNARY_FUNCTION(sinh, xfn::sinh)
DEFINE_VML_UNARY_FUNCTION(cosh, xfn::cosh)
DEFINE_VML_UNARY_FUNCTION(tanh, xfn::tanh)
DEFINE_VML_UNARY_FUNCTION(asinh, xfn::asinh)
DEFINE_VML_UNARY_FUNCTION(acosh, xfn::acosh)
DEFINE_VML_UNARY_FUNCTION(atanh, xfn::atanh)
DEFINE_VML_UNARY_FUNCTION(ceil, xfn::ceil)
DEFINE_VML_UNARY_FUNCTION(floor, xfn::floor)
#undef DEFINE_VML_UNARY_FUNCTION

#endif
