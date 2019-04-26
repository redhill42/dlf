#include <iostream>
#include "fibonacci.h"

extern void onnx_test();

int main() {
    for (int i = 0; i <= 12; i++)
        std::cout << fibonacci(i) << ", ";
    std::cout << std::endl;

    for (int i = 0; i <= 12; i++)
        std::cout << fibonacci(-i) << ", ";
    std::cout << std::endl;

    onnx_test();
}
