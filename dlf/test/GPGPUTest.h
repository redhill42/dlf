#pragma once

#include "gpgpu.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

class GPGPUTest : public ::testing::Test {
protected:
    static std::vector<gpgpu::Context> contexts;

    static void SetUpTestCase() {
        auto platform = gpgpu::probe();

        // Initialize the GPGPU platform and devices. This initializes the
        // OpenCL/CUDA back-end selects all devices on the platform.
        auto devices = platform.devices(gpgpu::DeviceType::GPU);

        // Create GPGPU context for each device.
        for (auto& dev : devices) {
            try {
                contexts.push_back(dev.createContext());
            } catch (gpgpu::APIError& e) {
                std::cerr << "Warning: " << e.what() << std::endl;
            }
        }
    }

    static void TearDownTestCase() {
        contexts.clear();
    }

    template <typename Test>
    static void doTest(Test&& test) {
        EXPECT_FALSE(contexts.empty());

        for (auto& context : contexts) {
            // Create GPGPU queue for each context. The queue can be used to
            // schedule commands such as launching a kernel or perform a
            // device-host memory copy.
            auto queue = context.createQueue();

            // Activate the context and associate it to current thread.
            // The context will be deactivated when control leaves the scope.
            gpgpu::current current(queue);

            // Run the test on the queues
            test();
        }
    }
};
