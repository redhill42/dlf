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
    if (iteration > 1)
        std::cout << " (" << (seconds/iteration) << "s per iteration)";
    std::cout << std::endl;
}
