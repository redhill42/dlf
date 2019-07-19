#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>

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

#ifdef NDEBUG
#define PERFORMANCE_TEST(test_case_name, test_name) TEST(test_case_name, test_name)
#define PERFORMANCE_TEST_F(test_case_name, test_name) TEST_F(test_case_name, test_name)
#define TYPED_PERFORMANCE_TEST(test_case_name, test_name) TYPED_TEST(test_case_name, test_name)
#else
#define PERFORMANCE_TEST(test_case_name, test_name) TEST(test_case_name, DISABLED_##test_name)
#define PERFORMANCE_TEST_F(test_case_name, test_name) TEST_F(test_case_name, DISABLED_##test_name)
#define TYPED_PERFORMANCE_TEST(test_case_name, test_name) TYPED_TEST(test_case_name, DISABLED_##test_name)
#endif
