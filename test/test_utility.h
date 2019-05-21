#ifndef _TEST_UTILITY_H
#define _TEST_UTILITY_H

#include <chrono>
#include <iostream>

template <typename Body>
void timing(const std::string& name, int iteration, Body&& body) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iteration; i++)
        body();
    auto elapsed = high_resolution_clock::now() - start;
    auto seconds = duration_cast<duration<double>>(elapsed).count();
    std::cout << name << " elapsed "
              << std::fixed << std::setprecision(6)
              << seconds
              << "s, takes " << (seconds/iteration) << "s per iteration."
              << std::endl;
}

#endif //_TEST_UTILITY_H
