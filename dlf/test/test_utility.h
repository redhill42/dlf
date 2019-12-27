#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>
#include "gtest/gtest.h"

template <typename Body>
void timing(const std::string& name, int iteration, Body&& body) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iteration; i++)
        body();
    auto elapsed = high_resolution_clock::now() - start;

    auto seconds = duration_cast<duration<double>>(elapsed).count();
    std::cout << name << " elapsed "
              << std::fixed
              #if __cplusplus > 201402L
              << std::setprecision(6)
              #endif
              << seconds << 's';
    if (iteration > 1) {
        #if __cplusplus > 201402L
        std::cout << std::setprecision(4);
        #endif
        std::cout << " (" << (seconds*1000/iteration) << "ms per iteration)";
    }
    std::cout << std::endl;
}

template <typename T1, typename T2>
bool isValueNear(const T1& x, const T2& y, double e) {
    if (x == y)
        return true;
    if (std::isnan(x))
        return std::isnan(y);
    if (std::isinf(x))
        return std::isinf(y);
    if (std::isnan(y) || std::isinf(y))
        return false;
    auto d = std::abs(x - y);
    return d <= e || d / std::sqrt(x*x + y*y) <= e;
}

template <typename T1, typename T2>
bool isValueNear(const std::complex<T1>& x, const std::complex<T2>& y, double e) {
    return isValueNear(x.real(), y.real(), e) &&
           isValueNear(x.imag(), y.imag(), e);
}

template <typename T1, typename T2>
bool isElementsNear(const T1& lhs, const T2& rhs, double abs_error) {
    auto it = rhs.begin();
    for (auto x : lhs) {
        if (!isValueNear(x, *it, abs_error))
            return false;
        ++it;
    }
    return true;
}

template <typename T1, typename T2>
testing::AssertionResult NearFailure(
    const char* lhs_expr, const char* rhs_expr, const T1& lhs, const T2& rhs)
{
    auto lhs_value = testing::internal::FormatForComparisonFailureMessage(lhs, rhs);
    auto rhs_value = testing::internal::FormatForComparisonFailureMessage(rhs, lhs);

    testing::Message msg;
    msg << "      Expected: " << lhs_expr;
    if (lhs_value != lhs_expr)
        msg << "\n      Which is: " << lhs_value;
    msg << "\nTo be near to: " << rhs_expr;
    if (rhs_value != rhs_expr)
        msg << "\n      Which is: " << rhs_value;
    return testing::AssertionFailure() << msg;
}

template <typename T1, typename T2>
testing::AssertionResult ValueNearPredFormat(
    const char* lhs_expr, const char* rhs_expr, const char* abs_error_expr,
    const T1& lhs, const T2& rhs, double abs_error)
{
    if (isValueNear(lhs, rhs, abs_error))
        return testing::AssertionSuccess();
    return NearFailure(lhs_expr, rhs_expr, lhs, rhs);
}

template <typename T1, typename T2>
testing::AssertionResult ElementsNearPredFormat(
    const char* lhs_expr, const char* rhs_expr, const char* abs_error_expr,
    const T1& lhs, const T2& rhs, double abs_error)
{
    if (isElementsNear(lhs, rhs, abs_error))
        return testing::AssertionSuccess();
    return NearFailure(lhs_expr, rhs_expr, lhs, rhs);
}

#define EXPECT_VALUE_NEAR(val1, val2) \
    EXPECT_PRED_FORMAT3(ValueNearPredFormat, val1, val2, 1e-4)
#define ASSERT_VALUE_NEAR(val1, val2) \
    ASSERT_PRED_FORMAT3(ValueNearPredFormat, val1, val2, 1e-4)

#define EXPECT_ELEMENTS_NEAR(val1, val2) \
    EXPECT_PRED_FORMAT3(ElementsNearPredFormat, val1, val2, 1e-4)
#define ASSERT_ELEMENTS_NEAR(val1, val2) \
    ASSERT_PRED_FORMAT3(ElementsNearPredFormat, val1, val2, 1e-4)
