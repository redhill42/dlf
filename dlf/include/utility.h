#pragma once

#include <type_traits>

namespace cxx {

#if __cplusplus > 201402L
template <typename... B>
using conjunction = std::conjunction<B...>;
template <typename... B>
using disjunction = std::disjunction<B...>;
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
#endif

#if __cplusplus > 201402L
template <typename Fn, typename... Args>
using invoke_result = std::invoke_result<Fn, Args...>;
template <typename Fn, typename... Args>
using invoke_result_t = std::invoke_result_t<Fn, Args...>;
#else
template <typename Fn, typename... Args>
using invoke_result = std::result_of<Fn(Args...)>;
template <typename Fn, typename... Args>
using invoke_result_t = std::result_of_t<Fn(Args...)>;
#endif

template <typename T, typename U = T>
T exchange(T& obj, U&& new_value) {
    T old_value = std::move(obj);
    obj = std::forward<U>(new_value);
    return old_value;
}

} // namespace cxx