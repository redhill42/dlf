#include <string>
#include <vector>
#include <cstdio>
#include "gpgpu.h"

int main() {
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
        printf(" > Local memory per work-group  %llu bytes\n", device.localMemSize());
        printf(" > Device capabilities          %s\n", device.capabilities().c_str());
        printf(" > Core clock rate              %u MHz\n", device.coreClock());
        printf(" > Number of compute units      %u\n", device.computeUnits());
        printf(" > Total memory size            %llu bytes\n", device.memorySize());
        printf(" > Maximum allocatable memory   %llu bytes\n", device.maxAllocSize());
    }

    printf("\nThe default device: %s\n", gpgpu::probe().device().name().c_str());

    return 0;
}
