#include <string>
#include <vector>
#include <array>
#include <cstdio>
#include <iostream>
#include <cinttypes>
#include "gpgpu.h"
#include "gpblas.h"

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

    printf("\nThe default device: %s\n", gpgpu::probe().device().name().c_str());
}

static void run_test() {
    auto device = gpgpu::probe().device();
    auto context = device.createContext();
    auto queue = context.createQueue();
    constexpr size_t N = 1024;

    auto host_A = std::array<float, N>();
    auto host_B = std::array<float, N>();

    for (size_t i = 0; i < N; i++) {
        host_A[i] = i;
    }

    auto dev_A = context.createBuffer<float>(N);
    dev_A.write(queue, host_A.data(), host_A.size());

    gpgpu::blas::scal(N, 3.0f, dev_A, 0, 1, queue);
    dev_A.read(queue, host_B.data(), host_B.size());

    for (auto index : {4, 500, 1000}) {
        std::cout << host_A[index] << " " << host_B[index] << std::endl;
    }
}

int main() {
    show_info();
    run_test();
}
