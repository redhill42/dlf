// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the common routine functions (see the header for more information).
//
// =================================================================================================

#include <vector>
#include <chrono>
#include <iostream>

#include "routines/common.hpp"

namespace gpgpu { namespace blas {

// Enqueues a kernel, waits for completion, and checks for errors
void RunKernel(const Kernel& kernel, const Queue& queue, const Device& device,
               std::vector<size_t> global, const std::vector<size_t>& local,
               Event* event)
{
    if (!local.empty()) {
        // Tests for validity of the local thread sizes
        if (local.size() > device.maxWorkItemDimensions())
            throw RuntimeErrorCode(StatusCode::kInvalidLocalNumDimensions);

        const auto max_work_item_sizes = device.maxWorkItemSizes();
        for (size_t i = 0; i < local.size(); ++i) {
            if (local[i] > max_work_item_sizes[i])
                throw RuntimeErrorCode(StatusCode::kInvalidLocalThreadsDim);
        }

        size_t local_size = std::accumulate(local.begin(), local.end(), 1, std::multiplies<>());
        if (local_size > device.maxWorkGroupSize()) {
            throw RuntimeErrorCode(StatusCode::kInvalidLocalThreadsTotal,
                ToString(local_size) + " is larger than " + ToString(device.maxWorkGroupSize()));
        }

        for (size_t i = 0; i < global.size(); ++i) {
            // Make sure the global thread sizes are at least equal to the local sizes
            if (global[i] < local[i])
                global[i] = local[i];
            // Verify that the global thread sizes are a multiple of the local sizes
            if (global[i] % local[i] != 0) {
                throw RuntimeErrorCode(StatusCode::kInvalidLocalThreadsDim,
                    ToString(global[i]) + " is not divisible by " + ToString(local[i]));
            }
        }
    }

    // Tests for local memory usage
    if (kernel.localMemoryUsage(device) > device.localMemSize()) {
        throw RuntimeErrorCode(StatusCode::kInvalidLocalMemUsage);
    }

    // Prints the name of the kernel to launch in case of debugging in verbose mode
#ifdef VERBOSE
    queue.finish();
    printf("[DEBUG] Running kernel\n");
    const auto start_time = std::chrono::steady_clock::now();
#endif

    // Launches the kernel (and checks for launch errors)
    kernel.launch(queue, global, local, event);

    // Prints the elapsed execution time in case of debugging in verbose mode
#ifdef VERBOSE
    queue.finish();
    const auto elapsed_time = std::chrono::steady_clock::now() - start_time;
    const auto timing = std::chrono::duration<double,std::milli>(elapsed_time).count();
    printf("[DEBUG] Completed kernel in %.2lf ms\n", timing);
#endif
}

// =================================================================================================

// Sets all elements of a matrix to a constant value
template <typename T>
void FillMatrix(const Queue& queue, const Device& device,
                const Program& program, Event* event,
                const size_t m, const size_t n, const size_t ld, const size_t offset,
                Buffer<T>& dest, const T constant_value, const size_t local_size)
{
    auto kernel = program.getKernel("FillMatrix");
    kernel.setArgument(0, static_cast<int>(m));
    kernel.setArgument(1, static_cast<int>(n));
    kernel.setArgument(2, static_cast<int>(ld));
    kernel.setArgument(3, static_cast<int>(offset));
    kernel.setArgument(4, dest);
    kernel.setArgument(5, GetRealArg(constant_value));

    auto global = std::vector<size_t>{Ceil(m, local_size), n};
    auto local = std::vector<size_t>{local_size, 1};
    RunKernel(kernel, queue, device, global, local, event);
}

// Compiles the above function
template void FillMatrix<half>(
    const Queue&, const Device&, const Program&,
    Event*, const size_t, const size_t, const size_t,
    const size_t, Buffer<half>&, const half, const size_t);
template void FillMatrix<float>(
    const Queue&, const Device&, const Program&,
    Event*, const size_t, const size_t, const size_t,
    const size_t, Buffer<float>&, const float, const size_t);
template void FillMatrix<double>(
    const Queue&, const Device&, const Program&,
    Event*, const size_t, const size_t, const size_t,
    const size_t, Buffer<double>&, const double, const size_t);
template void FillMatrix<float2>(
    const Queue&, const Device&, const Program&,
    Event*, const size_t, const size_t, const size_t,
    const size_t, Buffer<float2>&, const float2, const size_t);
template void FillMatrix<double2>(
    const Queue&, const Device&, const Program&,
    Event*, const size_t, const size_t, const size_t,
    const size_t, Buffer<double2>&, const double2, const size_t);

// Sets all elements of a vector to a constant value
template <typename T>
void FillVector(const Queue &queue, const Device &device,
                const Program& program, Event* event,
                const size_t n, const size_t inc, const size_t offset,
                Buffer<T>& dest, const T constant_value, const size_t local_size)
{
    auto kernel = program.getKernel("FillVector");
    kernel.setArgument(0, static_cast<int>(n));
    kernel.setArgument(1, static_cast<int>(inc));
    kernel.setArgument(2, static_cast<int>(offset));
    kernel.setArgument(3, dest);
    kernel.setArgument(4, GetRealArg(constant_value));

    auto global = std::vector<size_t>{Ceil(n, local_size)};
    auto local = std::vector<size_t>{local_size};
    RunKernel(kernel, queue, device, global, local, event);
}

// Compiles the above function
template void FillVector<half>(
    const Queue&, const Device&, const Program&,
    Event*, const size_t, const size_t, const size_t,
    Buffer<half>&, const half, const size_t);
template void FillVector<float>(
    const Queue&, const Device&, const Program&,
    Event*, const size_t, const size_t, const size_t,
    Buffer<float>&, const float, const size_t);
template void FillVector<double>(
    const Queue&, const Device&, const Program&,
    Event*, const size_t, const size_t, const size_t,
    Buffer<double>&, const double, const size_t);
template void FillVector<float2>(
    const Queue&, const Device&, const Program&,
    Event*, const size_t, const size_t, const size_t,
    Buffer<float2>&, const float2, const size_t);
template void FillVector<double2>(
    const Queue&, const Device&, const Program&,
    Event*, const size_t, const size_t, const size_t,
    Buffer<double2>&, const double2, const size_t);

}} // namespace gpgpu::blas
