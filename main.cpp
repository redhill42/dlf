#include <string>
#include <vector>
#include <array>
#include <cstdio>
#include <iostream>
#include <cinttypes>

#include "gpgpu.h"
#include "gblas.h"
#include "tensor.h"

using namespace dlf;

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

template <typename T>
inline void print_matrix(const char* title, Tensor<T> A) {
    std::cout << title << A << std::endl;
}

template <typename T = int>
static void gemm_test() {
    constexpr size_t M = 3, K = 6, N = 4;

    auto A = Tensor<T>({M, K}, {
        5, 10, 9, 1, 10, 3,
        7,  6, 6, 6,  1, 1,
        6,  2, 6, 10, 9, 3
    });

    auto A_t = Tensor<T>({K, M}, {
        5, 7, 6,
        10, 6, 2,
        9, 6, 6,
        1, 6, 10,
        10, 1, 9,
        3, 1, 3
    });

    auto B = Tensor<T>({K, N}, {
        7,  1, 8,  7,
        9,  5, 2,  6,
        7,  8, 5,  7,
        6,  9, 1,  1,
        4, 10, 1, 10,
        3,  8, 8,  5
    });

    auto B_t = Tensor<T>({N, K}, {
        7, 9, 7, 6, 4, 3,
        1, 5, 8, 9, 10, 8,
        8, 2, 5, 1, 1, 8,
        7, 6, 7, 1, 10, 5
    });

    auto C = Tensor<T>({M, N}, {
        230, 254, 116, 199,
        219, 236, 201, 252,
        173, 148, 155, 167
    });

    auto R = Tensor<T>({M, N}, {
        1176, 1282, 628, 1145,
        1033, 1022, 829, 1052,
        933,  980, 715,  923
    });

    auto dev_A = DevTensor<T>(A);
    auto dev_A_t = DevTensor<T>(A_t);
    auto dev_B = DevTensor<T>(B);
    auto dev_B_t = DevTensor<T>(B_t);
    auto dev_C = DevTensor<T>(C);

    auto alpha = T(2), beta = T(3);

    std::cout << "\nGEMM Test:\n";

    dev_C.write(C);
    gblas::gemm(gblas::Layout::RowMajor,
                gblas::Transpose::NoTrans,
                gblas::Transpose::NoTrans,
                M, N, K,
                alpha, dev_A.data(), dev_A.stride(0),
                dev_B.data(), dev_B.stride(0),
                beta, dev_C.data(), dev_C.stride(0));
    print_matrix("A . B:      ", dev_C.read());

    dev_C.write(C);
    gblas::gemm(gblas::Layout::RowMajor,
                gblas::Transpose::Trans,
                gblas::Transpose::NoTrans,
                M, N, K,
                alpha, dev_A_t.data(), dev_A_t.stride(0),
                dev_B.data(), dev_B.stride(0),
                beta, dev_C.data(), dev_C.stride(0));
    print_matrix("T(A) . B:   ", dev_C.read());

    dev_C.write(C);
    gblas::gemm(gblas::Layout::RowMajor,
                gblas::Transpose::NoTrans,
                gblas::Transpose::Trans,
                M, N, K,
                alpha, dev_A.data(), dev_A.stride(0),
                dev_B_t.data(), dev_B_t.stride(0),
                beta, dev_C.data(), dev_C.stride(0));
    print_matrix("A . T(B):   ", dev_C.read());

    dev_C.write(C);
    gblas::gemm(gblas::Layout::RowMajor,
                gblas::Transpose::Trans,
                gblas::Transpose::Trans,
                M, N, K,
                alpha, dev_A_t.data(), dev_A_t.stride(0),
                dev_B_t.data(), dev_B_t.stride(0),
                beta, dev_C.data(), dev_C.stride(0));
    print_matrix("T(A) . T(B):", dev_C.read());

    print_matrix("Expected:   ", R);
}

int main() {
    show_info();
    gemm_test();
}
