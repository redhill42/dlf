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

template <typename T>
inline void ExpectEQ(T a, T b) {
    EXPECT_EQ(a, b);
}

template <typename T>
bool isFloatEQ(T a, T b, T eps = T{1e-4}) {
    if (std::isnan(a))
        return std::isnan(b);
    if (std::isinf(a))
        return std::isinf(b);
    if (std::signbit(a) != std::signbit(b))
        return std::abs(a - b) <= eps;
    if (std::signbit(a)) {
        a = -a; b = -b;
    }

    int e = static_cast<int>(std::log10(a));
    if (e != static_cast<int>(std::log10(b)))
        return false;
    T scale = static_cast<T>(std::pow(10, std::abs(e)));
    return e >= 0
        ? std::abs(a/scale - b/scale) <= eps
        : std::abs(a*scale - b*scale) <= eps;
}

inline void ExpectEQ(float a, float b) {
    if (!isFloatEQ(a, b))
        FAIL() << a << " and " << b << " are not equal";
}

inline void ExpectEQ(double a, double b) {
    if (!isFloatEQ(a, b))
        FAIL() << a << " and " << b << " are not equal";
}

template <typename T>
inline void ExpectEQ(std::complex<T> a, std::complex<T> b) {
    ExpectEQ(a.real(), b.real());
    ExpectEQ(a.imag(), b.imag());
}
