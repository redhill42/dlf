#include "tensor.h"
#include "gpgpu.h"
#include "gtest/gtest.h"
#include "test_utility.h"

using namespace kneron::model;

static auto program_source = R"(
__kernel void multiply(__global float* x, __global float* y, const int factor) {
  const int tid = get_global_id(0);
  y[tid] = x[tid] * factor;
})";

static auto cl2cu =
    #include "cl2cu.inc"

TEST(GPGPU, API) {
    // Initialize the GPGPU platform and device. This initializes the
    // OpenCL/CUDA back-end selects a specific device on the platform.
    auto device = gpgpu::probe().device();

    // Creates a new GPGPU context and queue for this device. The queue
    // can be used to schedule commands such as launching a kernel or
    // performing a device-host memory copy.
    auto context = device.createContext();
    auto queue = context.createQueue();

    // Translate OpenCL kernel code into CUDA.
    std::string source = program_source;
    if (device.platform().api() == gpgpu::APITypes::CUDA) {
        source = cl2cu + source;
    }

    // Creates a new program based on the kernel string. Then, builds
    // this program and checks for any compilation errors. If there
    // are any, they are printed and execution is halted.
    auto cc = device.createContext();
    auto program = cc.compileProgram(source.c_str(), {});

    // Test for load program from binary
    program = context.loadProgram(program.getIR());
    auto kernel = program.getKernel("multiply");

    // Populate regular host vectors with example data
    auto host_a = Tensor<float>::range({2048, 2048}, 0);
    auto host_b = Tensor<float>({2048, 2048});

    // Creates two new device buffers and copies the host data to these
    // device buffer
    auto dev_a = context.createBuffer<float>(gpgpu::BufferAccess::kReadWrite, host_a.size());
    auto dev_b = context.createBuffer<float>(gpgpu::BufferAccess::kReadWrite, host_b.size());
    dev_a.writeAsync(queue, host_a.data(), host_a.size());

    // Creates a 1-dimensional thread configuration with thread-blocks/work-groups
    // of 256 threads and a total number of threads equal to the number of elements
    // in the input/output vectors.
    const auto kWorkGroupSize = device.maxWorkItemSizes();

    // Enqueues the kernel. Note that launching the kernel is always asynchronous
    // and thus requires finishing the queue in order to complete the operation.
    kernel.setArguments(dev_a, dev_b, 2);
    kernel.launch(queue, {host_a.size()}, {kWorkGroupSize});

    // Launch again by swapping parameter and multiply factor
    kernel.setArguments(dev_b, dev_a, 3);
    kernel.launch(queue, {host_a.size()}, {kWorkGroupSize});

    // Reads the results back to the host memory
    dev_a.readAsync(queue, host_b.data(), host_b.size());

    // Wait for queue completion
    queue.finish();

    // Verify the result
    for (auto index : {4, 900, 1500}) {
        EXPECT_EQ(host_a(index, index) * 6, host_b(index, index));
    }

    // End of execution: no frees or clean-up needed
}

void test_gemm(const size_t N) {
    using real = float;

    auto A = Tensor<real>::random({N, N}, 0, 1);
    auto B = Tensor<real>::random({N, N}, 0, 1);
    auto C = Tensor<real>({N, N});

    timing("CPU gemm " + std::to_string(N), 5, [&]() {
        cblas::gemm(cblas::Layout::RowMajor,
                    cblas::Transpose::NoTrans,
                    cblas::Transpose::NoTrans,
                    N, N, N, 1.0,
                    A.data(), N,
                    B.data(), N, 0.0,
                    C.data(), N);
    });

    auto device = gpgpu::probe().devices(gpgpu::DeviceType::GPU)[1];
    auto context = device.createContext();
    auto queue = context.createQueue();

    auto dev_A = context.createBuffer<real>(gpgpu::BufferAccess::kWriteOnly, A.size());
    auto dev_B = context.createBuffer<real>(gpgpu::BufferAccess::kWriteOnly, B.size());
    auto dev_C = context.createBuffer<real>(gpgpu::BufferAccess::kReadOnly, C.size());

    timing("GPU gemm " + std::to_string(N), 5, [&]() {
        dev_A.writeAsync(queue, A.data(), A.size());
        dev_B.writeAsync(queue, B.data(), B.size());

        gpgpu::blas::gemm(cblas::Layout::RowMajor,
                          cblas::Transpose::NoTrans,
                          cblas::Transpose::NoTrans,
                          N, N, N, 1.0f,
                          dev_A, N,
                          dev_B, N, 0.0f,
                          dev_C, N,
                          queue);

        dev_C.readAsync(queue, C.data(), C.size());
        queue.finish();
    });
}

TEST(GPGPU, GEMMPerformance) {
    for (int n = 512; n <= 8192; n *= 2) {
        test_gemm(n);
    }
}
