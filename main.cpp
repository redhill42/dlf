#include <string>
#include <vector>
#include <array>
#include <cstdio>
#include <iostream>
#include <cinttypes>
#include "gpgpu.h"
#include "gblas.h"

static void show_info() {
    const auto platform = gpgpu::probe();

    printf("\n## Printing platform information...\n");
    printf(" > Platform name                %s\n", platform.name().c_str());
    printf(" > Platform vendor              %s\n", platform.vendor().c_str());
    printf(" > Platform version             %s\n", platform.version().c_str());

    for (const auto& device : platform.devices(gpgpu::DeviceType::All)) {
        printf("\n## Printing device information...\n");
        printf(" > Framework version            %s\n", device.version().c_str());
        printf(" > Vendor                       %s\n", device.vendor().c_str());
        printf(" > Device name                  %s\n", device.name().c_str());
        printf(" > Max work-group size          %zu\n", device.maxWorkGroupSize());
        printf(" > Max thread dimensions        %zu\n", device.maxWorkItemDimensions());
        printf(" > Max work-group sizes:\n");
        for (size_t i = 0; i < device.maxWorkItemDimensions(); ++i)
            printf("   - in the %zu-dimension         %zu\n", i, device.maxWorkItemSizes()[i]);
        printf(" > Local memory per work-group  %" PRIu64 " bytes\n", device.localMemSize());
        printf(" > Device capabilities          %s\n", device.capabilities().c_str());
        printf(" > Core clock rate              %u MHz\n", device.coreClock());
        printf(" > Number of compute units      %u\n", device.computeUnits());
        printf(" > Total memory size            %" PRIu64 " bytes\n", device.memorySize());
        printf(" > Maximum allocatable memory   %" PRIu64 " bytes\n", device.maxAllocSize());
    }
}

template <typename T, size_t N>
static void print_matrix(const char* title, std::array<T, N> A) {
    std::cout << title;
    for (auto x : A)
        std::cout << x << " ";
    std::cout << std::endl;
}

static void gemm_test() {
    constexpr size_t M = 3, K = 6, N = 4;

    auto A = std::array<float, M*K> {
        5, 10, 9, 1, 10, 3,
        7,  6, 6, 6,  1, 1,
        6,  2, 6, 10, 9, 3
    };

    auto A_t = std::array<float, M*K> {
        5, 7, 6,
        10, 6, 2,
        9, 6, 6,
        1, 6, 10,
        10, 1, 9,
        3, 1, 3
    };

    auto B = std::array<float, K*N> {
        7,  1, 8,  7,
        9,  5, 2,  6,
        7,  8, 5,  7,
        6,  9, 1,  1,
        4, 10, 1, 10,
        3,  8, 8,  5
    };

    auto B_t = std::array<float, K*N> {
        7, 9, 7, 6, 4, 3,
        1, 5, 8, 9, 10, 8,
        8, 2, 5, 1, 1, 8,
        7, 6, 7, 1, 10, 5
    };

    auto C = std::array<float, M*N> {
        230, 254, 116, 199,
        219, 236, 201, 252,
        173, 148, 155, 167
    };

    auto R = std::array<float, M*N> {
        1176, 1282, 628, 1145,
        1033, 1022, 829, 1052,
        933,  980, 715,  923
    };

    auto T = std::array<float, M*N> {};

    auto context = gpgpu::current::context();
    auto queue = gpgpu::current::queue();

    auto dev_A = context.createBuffer<float>(A.size());
    auto dev_A_t = context.createBuffer<float>(A_t.size());
    auto dev_B = context.createBuffer<float>(B.size());
    auto dev_B_t = context.createBuffer<float>(B_t.size());
    auto dev_C = context.createBuffer<float>(C.size());

    dev_A.write(queue, A.data(), A.size());
    dev_A_t.write(queue, A_t.data(), A_t.size());
    dev_B.write(queue, B.data(), B.size());
    dev_B_t.write(queue, B_t.data(), B_t.size());

    std::cout << "\nGEMM Test:\n";

    dev_C.write(queue, C.data(), C.size());
    gblas::gemm(gblas::Layout::RowMajor,
                gblas::Transpose::NoTrans,
                gblas::Transpose::NoTrans,
                M, N, K,
                2.0f, dev_A, 0, K,
                dev_B, 0, N,
                3.0f, dev_C, 0, N,
                queue);
    dev_C.read(queue, T.data(), T.size());
    print_matrix("A . B:      ", T);

    dev_C.write(queue, C.data(), C.size());
    gblas::gemm(gblas::Layout::RowMajor,
                gblas::Transpose::Trans,
                gblas::Transpose::NoTrans,
                M, N, K,
                2.0f, dev_A_t, 0, M,
                dev_B, 0, N,
                3.0f, dev_C, 0, N,
                queue);
    dev_C.read(queue, T.data(), T.size());
    print_matrix("T(A) . B:   ", T);

    dev_C.write(queue, C.data(), C.size());
    gblas::gemm(gblas::Layout::RowMajor,
                gblas::Transpose::NoTrans,
                gblas::Transpose::Trans,
                M, N, K,
                2.0f, dev_A, 0, K,
                dev_B_t, 0, K,
                3.0f, dev_C, 0, N,
                queue);
    dev_C.read(queue, T.data(), T.size());
    print_matrix("A . T(B):   ", T);

    dev_C.write(queue, C.data(), C.size());
    gblas::gemm(gblas::Layout::RowMajor,
                gblas::Transpose::Trans,
                gblas::Transpose::Trans,
                M, N, K,
                2.0f, dev_A_t, 0, M,
                dev_B_t, 0, K,
                3.0f, dev_C, 0, N,
                queue);
    dev_C.read(queue, T.data(), T.size());
    print_matrix("T(A) . T(B):", T);

    print_matrix("Expected:   ", R);
}

int main() {
    show_info();
    gemm_test();
}
