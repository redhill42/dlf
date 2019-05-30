#include "tensor.h"
#include "gpgpu.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test_utility.h"

using namespace tensor;

static auto cl2cu =
    #include "cl2cu.inc"

class GPGPUTest : public ::testing::Test {
protected:
    gpgpu::Device device;
    gpgpu::Context context;
    gpgpu::Queue queue;

    GPGPUTest() {
        // Initialize the GPGPU platform and device. This initializes the
        // OpenCL/CUDA back-end selects a specific device on the platform.
        device = gpgpu::probe().device();

        // Creates a new GPGPU context and queue for this device. The queue
        // can be used to schedule commands such as launching a kernel or
        // performing a device-host memory copy.
        context = device.createContext();
        queue = context.createQueue();
    }

    gpgpu::Program compile(std::string source) {
        // Translate OpenCL kernel code into CUDA.
        if (device.platform().api() == gpgpu::APITypes::CUDA) {
            source = cl2cu + source;
        }

        // Creates a new program based on the kernel string. Then, builds
        // this program and checks for any compilation errors. If there
        // are any, they are printed and execution is halted.
        return context.compileProgram(source.c_str(), {});
    }
};

static auto program_source = R"(
__kernel void multiply(__global float* x, __global float* y, const int factor) {
  const int tid = get_global_id(0);
  y[tid] = x[tid] * factor;
})";

TEST_F(GPGPUTest, CompileProgram) {
    auto kernel = compile(program_source).getKernel("multiply");

    // Populate regular host vectors with example data
    auto host_a = Tensor<float>::range({2048, 2048}, 0);

    // Creates two new device buffers and copies the host data to these
    // device buffer
    auto dev_a = DevTensor(host_a, queue);
    auto dev_b = DevTensor<float>(host_a.shape(), queue);

    // Creates a 1-dimensional thread configuration with thread-blocks/work-groups
    // of 256 threads and a total number of threads equal to the number of elements
    // in the input/output vectors.
    const auto kWorkGroupSize = 128;

    // Enqueues the kernel. Note that launching the kernel is always asynchronous
    // and thus requires finishing the queue in order to complete the operation.
    kernel.setArguments(dev_a.buffer(), dev_b.buffer(), 2);
    kernel.launch(queue, {dev_a.size()}, {kWorkGroupSize});

    // Reads the results back to the host memory
    auto host_b = dev_b.read();

    // Verify the result
    for (auto index : {4, 900, 1500}) {
        EXPECT_EQ(host_a(index, index) * 2, host_b(index, index));
    }

    // End of execution: no frees or clean-up needed
}

TEST_F(GPGPUTest, Operator) {
    auto A = Tensor<float>::range({2, 3, 4}, 11);
    auto B = Tensor<float>::range({2, 3, 4}, 5);

    auto dev_A = DevTensor(A, queue);
    auto dev_B = DevTensor(B, queue);

    EXPECT_EQ((dev_A + dev_B).read(), A + B);
    EXPECT_EQ((dev_A - dev_B).read(), A - B);
    EXPECT_EQ((dev_A * dev_B).read(), A * B);
    EXPECT_EQ((dev_A * 7.0f).read(), A * 7.0f);
    EXPECT_EQ((7.0f * dev_A).read(), 7.0f * A);
    EXPECT_EQ(((dev_A + dev_B) * 3.0f).read(), (A + B) * 3.0f);
    EXPECT_EQ((3.0f * (dev_A - dev_B)).read(), 3.0f * (A - B));
    EXPECT_EQ(((dev_A + dev_B) * (dev_A - dev_B)).read(), ((A + B) * (A - B)));
}

TEST_F(GPGPUTest, VectorDotVector) {
    auto A = Tensor<float>({4}, {2, 7, 3, 4});
    auto B = Tensor<float>({4}, {4, 1, 9, 6});
    auto R = Tensor<float>({1}, {66});

    auto dev_A = DevTensor(A, queue);
    auto dev_B = DevTensor(B, queue);
    auto dev_C = DevTensor<float>({1}, queue);

    inner(dev_A, dev_B, &dev_C);
    EXPECT_EQ(dev_C.read(), R);

    auto dev_T = inner(dev_A, dev_B);
    EXPECT_EQ(dev_T.read(), R);
}

TEST_F(GPGPUTest, MatrixDotVector) {
    auto A = Tensor<float>({2, 3}, {2, 7, 3, 5, 9, 6});
    auto B = Tensor<float>({3}, {9, 6, 7});
    auto R = Tensor<float>({2}, {81, 141});

    auto dev_A = DevTensor(A, queue);
    auto dev_B = DevTensor(B, queue);
    auto dev_C = DevTensor<float>({2}, queue);

    inner(dev_A, dev_B, &dev_C);
    EXPECT_EQ(dev_C.read(), R);

    auto dev_T = inner(dev_A, dev_B);
    EXPECT_EQ(dev_T.read(), R);
}

TEST_F(GPGPUTest, VectorDoMatrix) {
    auto A = Tensor<float>({3}, {9, 6, 7});
    auto B = Tensor<float>({3, 2}, {2, 7, 3, 5, 9, 6});
    auto R = Tensor<float>({2}, {99, 135});

    auto dev_A = DevTensor(A, queue);
    auto dev_B = DevTensor(B, queue);
    auto dev_C = DevTensor<float>({2}, queue);

    inner(dev_A, dev_B, &dev_C);
    EXPECT_EQ(dev_C.read(), R);

    auto dev_T = inner(dev_A, dev_B);
    EXPECT_EQ(dev_T.read(), R);

}

TEST_F(GPGPUTest, MatrixDotMatrix) {
    auto A = Tensor<float>({2, 3}, {2, 7, 3, 5, 9, 6});
    auto B = Tensor<float>({3, 2}, {2, 7, 3, 5, 9, 6});
    auto R = Tensor<float>({2, 2}, {52, 67, 91, 116});

    auto dev_A = DevTensor(A, queue);
    auto dev_B = DevTensor(B, queue);
    auto dev_C = DevTensor<float>({2, 2}, queue);

    inner(dev_A, dev_B, &dev_C);
    EXPECT_EQ(dev_C.read(), R);

    auto dev_T = inner(dev_A, dev_B);
    EXPECT_EQ(dev_T.read(), R);
}

TEST_F(GPGPUTest, GEMM) {
    Tensor<float> A({3, 6}, {
        5, 10, 9, 1, 10, 3,
        7,  6, 6, 6,  1, 1,
        6,  2, 6, 10, 9, 3
    });

    Tensor<float> B({6, 4}, {
        7,  1, 8,  7,
        9,  5, 2,  6,
        7,  8, 5,  7,
        6,  9, 1,  1,
        4, 10, 1, 10,
        3,  8, 8,  5
    });

    Tensor<float> C({3, 4}, {
        230, 254, 116, 199,
        219, 236, 201, 252,
        173, 148, 155, 167
    });

    Tensor<float> R({3, 4}, {
        1176, 1282, 628, 1145,
        1033, 1022, 829, 1052,
         933,  980, 715,  923
    });

    auto dev_A = DevTensor(A, queue);
    auto dev_B = DevTensor(B, queue);
    auto dev_C = DevTensor(C, queue);

    auto dev_T = gemm(2.0f, dev_A, dev_B, 3.0f, dev_C);
    EXPECT_THAT(dev_T.read(), R);
}

static void clamp_test(const size_t n, const gpgpu::Queue& queue) {
    using namespace testing;
    constexpr auto min = -5.0f, max = 5.0f;

    auto A = Tensor<float>::random({n}, -10.0f, 10.0f);
    auto dev_A = DevTensor(A, queue);

    clamp(dev_A, min, max);
    auto B = dev_A.read();

    clamp(A, min, max);

    EXPECT_THAT(A, Each(AllOf(Ge(min), Le(max))));
    EXPECT_THAT(B, Each(AllOf(Ge(min), Le(max))));
    EXPECT_EQ(A, B);
}

TEST_F(GPGPUTest, Xclamp) {
    clamp_test(500, queue);
    clamp_test(1024, queue);
}
