#ifndef _UTILITY_H
#define _UTILITY_H

#include <chrono>
#include <iostream>

template <typename Body>
void timing(const std::string& name, Body&& body) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    body();
    auto elapsed = high_resolution_clock::now() - start;
    std::cout << name << " elapsed "
              << std::fixed << std::setprecision(8)
              << duration_cast<duration<double>>(elapsed).count()
              << std::endl;
}

#endif //_UTILITY_H
