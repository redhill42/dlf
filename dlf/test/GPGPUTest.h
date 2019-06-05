#pragma once

#include "gpgpu.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

class GPGPUTest : public ::testing::Test {
protected:
    static std::vector<gpgpu::Queue> queues;

    static void SetUpTestCase() {
        auto platform = gpgpu::probe();

        if (platform.api() == gpgpu::APITypes::OpenCL) {
            // Initialize the GPGPU platform and devices. This initializes the
            // OpenCL/CUDA back-end selects all devices on the platform.
            auto devices = platform.devices(gpgpu::DeviceType::GPU);

            // Create GPGPU context and queue for each device. The queue can
            // be used to schedule commands such as launching a kernel or
            // perform a device-host memory copy.
            for (auto& dev : devices) {
                queues.push_back(dev.createContext().createQueue());
            }
        } else {
            // FIXME: CUDA associate a context with current CPU thread, We'll
            // implement this mode for OpenCL in the future. In the mean time,
            // only one context should be created.
            queues.push_back(platform.device().createContext().createQueue());
        }
    }

    static void TearDownTestCase() {
        queues.clear();
    }

    template <typename Test>
    static void doTest(Test&& test) {
        // Run the test on the queues
        for (auto const& queue : queues) {
            test(queue);
        }
    }
};
