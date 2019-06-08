#include <future>
#include "tensor.h"
#include "test_utility.h"
#include "GPGPUTest.h"

using namespace dlf;

static gpgpu::Program compile(const gpgpu::Queue& queue, std::string source) {
    auto context = queue.context();
    auto device = context.device();

    // Translate OpenCL kernel code into CUDA.
    if (device.platform().api() == gpgpu::APIType::CUDA) {
        auto cl2cu =
            #include "kernels/opencl_to_cuda.cl"
            ;
        source = cl2cu + source;
    }

    // Creates a new program based on the kernel string. Then, builds
    // this program and checks for any compilation errors. If there
    // are any, they are printed and execution is halted.
    return context.compileProgram(source.c_str(), {});
}

std::vector<gpgpu::Context> GPGPUTest::contexts;

static auto program_source = R"(
__kernel void multiply(__global float* x, __global float* y, const int factor) {
  const int tid = get_global_id(0);
  y[tid] = x[tid] * factor;
})";

TEST_F(GPGPUTest, CompileProgram) {
    doTest([](auto const& queue) {
        auto kernel = compile(queue, program_source).getKernel("multiply");

        // Populate regular host vectors with example data
        auto host_a = Tensor<float>::range({2048, 2048}, 0);

        // Creates two new device buffers and copies the host data to these
        // device buffer
        auto dev_a = DevTensor<float>(host_a, queue);
        auto dev_b = DevTensor<float>(host_a.shape(), queue);

        // Creates a 1-dimensional thread configuration with thread-blocks/work-groups
        // of 256 threads and a total number of threads equal to the number of elements
        // in the input/output vectors.
        const auto kWorkGroupSize = 128;

        // Enqueues the kernel. Note that launching the kernel is always asynchronous
        // and thus requires finishing the queue in order to complete the operation.
        kernel.setArguments(dev_a.data(), dev_b.data(), 2);
        kernel.launch(queue, {dev_a.size()}, {kWorkGroupSize});

        // Reads the results back to the host memory
        auto host_b = dev_b.read(queue);

        // Verify the result
        for (auto index : {4, 900, 1500}) {
            EXPECT_EQ(host_a(index, index) * 2, host_b(index, index));
        }

        // End of execution: no frees or clean-up needed
    });
}

TEST_F(GPGPUTest, DevTensorCopyConstructor) {
    auto A = Tensor<float>::range({2, 3, 4}, 11);
    auto dev_A = DevTensor<float>(A);
    auto dev_B = dev_A;
    EXPECT_EQ(dev_B.read(), A);
}

TEST_F(GPGPUTest, DevTensorCopyAssignment) {
    Shape shape{2, 3, 4};
    auto A = Tensor<float>::range(shape, 1);
    auto B = Tensor<float>::range(shape, 3);
    auto dev_A = DevTensor<float>(shape);
    auto dev_B = DevTensor<float>(shape);
    dev_A.write(A);
    dev_B.write(B);
    EXPECT_EQ(dev_B.read(), B);
    dev_B = dev_A;
    EXPECT_EQ(dev_B.read(), A);
}

template <typename T>
static void dev_tensor_operator_test() {
    auto A = Tensor<T>::range({2, 3, 4}, 11);
    auto B = Tensor<T>::range({2, 3, 4}, 5);

    auto dev_A = DevTensor<T>(A);
    auto dev_B = DevTensor<T>(B);

    EXPECT_EQ((dev_A + dev_B).read(), A + B);
    EXPECT_EQ((dev_A - dev_B).read(), A - B);
    EXPECT_EQ((dev_A * dev_B).read(), A * B);
    EXPECT_EQ((dev_A * T(7)).read(), A * T(7));
    EXPECT_EQ((T(7) * dev_A).read(), T(7) * A);

    EXPECT_EQ(((dev_A + dev_B) * T(3)).read(), (A + B) * T(3));
    EXPECT_EQ((T(3) * (dev_A - dev_B)).read(), T(3) * (A - B));
    EXPECT_EQ(((dev_A + dev_B) * (dev_A - dev_B)).read(), (A + B) * (A - B));

    EXPECT_EQ((dev_A * T(3) + dev_B).read(), A * T(3) + B);
    EXPECT_EQ((dev_A + dev_B * T(3)).read(), A + B * T(3));
    EXPECT_EQ((dev_A * T(3) + dev_B * T(7)).read(), A * T(3) + B * T(7));

    EXPECT_EQ((dev_A * T(3) - dev_B).read(), A * T(3) - B);
    EXPECT_EQ((dev_A - dev_B * T(3)).read(), A - B * T(3));
    EXPECT_EQ((dev_A * T(3) - dev_B * T(7)).read(), A * T(3) - B * T(7));
}

TEST_F(GPGPUTest, DevTensorOperators) {
    dev_tensor_operator_test<float>();
    dev_tensor_operator_test<int32_t>();
    dev_tensor_operator_test<int64_t>();
}

template <typename T>
static void vector_dot_vector_test(const gpgpu::Queue& queue) {
    auto A = Tensor<T>({4}, {2, 7, 3, 4});
    auto B = Tensor<T>({4}, {4, 1, 9, 6});
    auto R = Tensor<T>({1}, {66});

    auto dev_A = DevTensor<T>(A, queue);
    auto dev_B = DevTensor<T>(B, queue);
    auto dev_C = DevTensor<T>({1}, queue);

    inner(dev_A, dev_B, &dev_C, queue);
    EXPECT_EQ(dev_C.read(queue), R);

    auto dev_T = inner(dev_A, dev_B, queue);
    EXPECT_EQ(dev_T.read(queue), R);

}
TEST_F(GPGPUTest, VectorDotVector) {
    doTest([](auto const& queue) {
        vector_dot_vector_test<float>(queue);
        vector_dot_vector_test<int32_t>(queue);
        vector_dot_vector_test<int64_t>(queue);
    });
}

template <typename T>
static void matrix_dot_vector_test(const gpgpu::Queue& queue) {
    auto A = Tensor<T>({2, 3}, {2, 7, 3, 5, 9, 6});
    auto B = Tensor<T>({3}, {9, 6, 7});
    auto R = Tensor<T>({2}, {81, 141});

    auto dev_A = DevTensor<T>(A, queue);
    auto dev_B = DevTensor<T>(B, queue);
    auto dev_C = DevTensor<T>({2}, queue);

    inner(dev_A, dev_B, &dev_C, queue);
    EXPECT_EQ(dev_C.read(queue), R);

    auto dev_T = inner(dev_A, dev_B, queue);
    EXPECT_EQ(dev_T.read(queue), R);
}

TEST_F(GPGPUTest, MatrixDotVector) {
    doTest([](auto const& queue) {
         matrix_dot_vector_test<float>(queue);
         matrix_dot_vector_test<int32_t>(queue);
         matrix_dot_vector_test<int64_t>(queue);
    });
}

template <typename T>
static void vector_dot_matrix_test(const gpgpu::Queue& queue) {
    auto A = Tensor<T>({3}, {9, 6, 7});
    auto B = Tensor<T>({3, 2}, {2, 7, 3, 5, 9, 6});
    auto R = Tensor<T>({2}, {99, 135});

    auto dev_A = DevTensor<T>(A, queue);
    auto dev_B = DevTensor<T>(B, queue);
    auto dev_C = DevTensor<T>({2}, queue);

    inner(dev_A, dev_B, &dev_C, queue);
    EXPECT_EQ(dev_C.read(queue), R);

    auto dev_T = inner(dev_A, dev_B, queue);
    EXPECT_EQ(dev_T.read(queue), R);
}

TEST_F(GPGPUTest, VectorDoMatrix) {
    doTest([](auto const& queue) {
        vector_dot_matrix_test<float>(queue);
        vector_dot_matrix_test<int32_t>(queue);
        vector_dot_matrix_test<int64_t>(queue);
    });
}

template <typename T>
static void matrix_dot_matrix_test(const gpgpu::Queue& queue) {
    auto A = Tensor<T>({2, 3}, {2, 7, 3, 5, 9, 6});
    auto B = Tensor<T>({3, 2}, {2, 7, 3, 5, 9, 6});
    auto R = Tensor<T>({2, 2}, {52, 67, 91, 116});

    auto dev_A = DevTensor<T>(A, queue);
    auto dev_B = DevTensor<T>(B, queue);
    auto dev_C = DevTensor<T>({2, 2}, queue);

    inner(dev_A, dev_B, &dev_C, queue);
    EXPECT_EQ(dev_C.read(queue), R);

    auto dev_T = inner(dev_A, dev_B, queue);
    EXPECT_EQ(dev_T.read(queue), R);
}

TEST_F(GPGPUTest, MatrixDotMatrix) {
    doTest([](auto const& queue) {
        matrix_dot_matrix_test<float>(queue);
        matrix_dot_matrix_test<int32_t>(queue);
        matrix_dot_matrix_test<int64_t>(queue);
    });
}

template <typename T>
static void gemm_test(const gpgpu::Queue& queue) {
    Tensor<T> A({3, 6}, {
        5, 10, 9,  1, 10, 3,
        7,  6, 6,  6,  1, 1,
        6,  2, 6, 10,  9, 3
    });

    Tensor<T> A_t({6, 3}, {
         5,  7,  6,
        10,  6,  2,
         9,  6,  6,
         1,  6, 10,
        10,  1,  9,
         3,  1,  3
    });

    Tensor<T> B({6, 4}, {
        7,  1,  8,  7,
        9,  5,  2,  6,
        7,  8,  5,  7,
        6,  9,  1,  1,
        4, 10,  1, 10,
        3,  8,  8,  5
    });

    Tensor<T> B_t({4, 6}, {
        7, 9, 7, 6,  4, 3,
        1, 5, 8, 9, 10, 8,
        8, 2, 5, 1,  1, 8,
        7, 6, 7, 1, 10, 5
    });

    Tensor<T> C({3, 4}, {
        230, 254, 116, 199,
        219, 236, 201, 252,
        173, 148, 155, 167
    });

    Tensor<T> R({3, 4}, {
        1176, 1282, 628, 1145,
        1033, 1022, 829, 1052,
         933,  980, 715,  923
    });

    auto dev_A = DevTensor<T>(A, queue);
    auto dev_A_t = DevTensor<T>(A_t, queue);
    auto dev_B = DevTensor<T>(B, queue);
    auto dev_B_t = DevTensor<T>(B_t, queue);
    auto dev_C = DevTensor<T>(C, queue);

    T alpha = T(2), beta = T(3);

    EXPECT_EQ(R, gemm(alpha, dev_A, dev_B, beta, dev_C, false, false, queue).read(queue));
    EXPECT_EQ(R, gemm(alpha, dev_A_t, dev_B, beta, dev_C, true, false, queue).read(queue));
    EXPECT_EQ(R, gemm(alpha, dev_A, dev_B_t, beta, dev_C, false, true, queue).read(queue));
    EXPECT_EQ(R, gemm(alpha, dev_A_t, dev_B_t, beta, dev_C, true, true, queue).read(queue));
}

TEST_F(GPGPUTest, GEMM) {
    doTest([](auto const& queue) {
        gemm_test<float>(queue);
        gemm_test<int32_t>(queue);
        gemm_test<int64_t>(queue);
    });
}

template <typename T, int N = 10, typename CBlas, typename GBlas>
void blas_level1_test(const gpgpu::Queue& queue, CBlas&& cblas, GBlas&& gblas) {
    int a;
    do {
        a = rand() % N;
    } while (a == 0 || a == 1);
    T alpha = static_cast<T>(a);

    auto A = Tensor<int>::random({N}, -N, N).cast<T>();
    auto B = Tensor<int>::random({N}, -N, N).cast<T>();

    auto dev_A = DevTensor<T>(A, queue);
    auto dev_B = DevTensor<T>(B, queue);

    cblas(N, A, B, alpha);
    gblas(N, dev_A, dev_B, alpha);

    EXPECT_EQ(A, dev_A.read(queue));
    EXPECT_EQ(B, dev_B.read(queue));
}

template <typename T, int N = 10, typename CBlas, typename GBlas>
void blas_level1_r_test(const gpgpu::Queue& queue, CBlas&& cblas, GBlas&& gblas) {
    auto A = Tensor<int>::random({N}, -N, N).cast<T>();
    auto B = Tensor<int>::random({N}, -N, N).cast<T>();

    auto dev_A = DevTensor<T>(A, queue);
    auto dev_B = DevTensor<T>(B, queue);
    auto dev_R = DevTensor<T>({1}, queue);

    auto R = cblas(N, A, B);
    gblas(N, dev_A, dev_B, dev_R);
    EXPECT_NEAR(R, dev_R.read(queue)(0), 0.0001f);
}

TEST_F(GPGPUTest, Xcopy) {
    SCOPED_TRACE("Xcopy");
    doTest([](auto const& queue) {
        blas_level1_test<float>(queue,
            [](auto N, auto& A, auto& B, auto) {
                cblas::copy(N, A.data(), 1, B.data(), 1);
            },
            [&](auto N, auto& A, auto& B, auto) {
                gblas::copy(N, A.data(), 1, B.data(), 1, queue);
            });

        blas_level1_test<int32_t>(queue,
            [](auto N, auto& A, auto& B, auto) {
                std::copy(A.begin(), A.end(), B.begin());
            },
            [&](auto N, auto& A, auto& B, auto) {
                gblas::copy(N, A.data(), 1, B.data(), 1, queue);
            });

        blas_level1_test<int64_t>(queue,
            [](auto N, auto& A, auto& B, auto) {
                std::copy(A.begin(), A.end(), B.begin());
            },
            [&](auto N, auto& A, auto& B, auto) {
                gblas::copy(N, A.data(), 1, B.data(), 1, queue);
            });
    });
}

TEST_F(GPGPUTest, Xswap) {
    SCOPED_TRACE("Xswap");
    doTest([](auto const& queue) {
        blas_level1_test<float>(queue,
            [](auto N, auto& A, auto& B, auto) {
                cblas::swap(N, A.data(), 1, B.data(), 1);
            },
            [&](auto N, auto& A, auto& B, auto) {
                gblas::swap(N, A.data(), 1, B.data(), 1, queue);
            });

        blas_level1_test<int32_t>(queue,
            [](auto N, auto& A, auto& B, auto) {
                std::swap(A, B);
            },
            [&](auto N, auto& A, auto& B, auto) {
                gblas::swap(N, A.data(), 1, B.data(), 1, queue);
            });

        blas_level1_test<int64_t>(queue,
            [](auto N, auto& A, auto& B, auto) {
                std::swap(A, B);
            },
            [&](auto N, auto& A, auto& B, auto) {
                gblas::swap(N, A.data(), 1, B.data(), 1, queue);
            });
    });
}

TEST_F(GPGPUTest, Xscal) {
    SCOPED_TRACE("Xscal");
    doTest([](auto const& queue) {
        blas_level1_test<float>(queue,
            [](auto N, auto& A, auto&, auto alpha) {
                cblas::scal(N, alpha, A.data(), 1);
            },
            [&](auto N, auto& A, auto&, auto alpha) {
                gblas::scal(N, alpha, A.data(), 1, queue);
            });

        blas_level1_test<int32_t>(queue,
            [](auto N, auto& A, auto&, auto alpha) {
                A *= alpha;
            },
            [&](auto N, auto& A, auto&, auto alpha) {
                gblas::scal(N, alpha, A.data(), 1, queue);
            });

        blas_level1_test<int64_t>(queue,
            [](auto N, auto& A, auto&, auto alpha) {
                A *= alpha;
            },
            [&](auto N, auto& A, auto&, auto alpha) {
                gblas::scal(N, alpha, A.data(), 1, queue);
            });
    });
}

TEST_F(GPGPUTest, Xaxpy) {
    SCOPED_TRACE("Xaxpy");
    doTest([](auto const& queue) {
        blas_level1_test<float>(queue,
            [](auto N, auto& A, auto& B, auto alpha) {
                cblas::axpy(N, alpha, A.data(), 1, B.data(), 1);
            },
            [&](auto N, auto& A, auto& B, auto alpha) {
                gblas::axpy(N, alpha, A.data(), 1, B.data(), 1, queue);
            });

        blas_level1_test<int32_t>(queue,
            [](auto N, auto& A, auto& B, auto alpha) {
                A.transformTo(B, B, [=](auto a, auto b) { return alpha*a+b; });
            },
            [&](auto N, auto& A, auto& B, auto alpha) {
                gblas::axpy(N, alpha, A.data(), 1, B.data(), 1, queue);
            });

        blas_level1_test<int64_t>(queue,
            [](auto N, auto& A, auto& B, auto alpha) {
                A.transformTo(B, B, [=](auto a, auto b) { return alpha*a+b; });
            },
            [&](auto N, auto& A, auto& B, auto alpha) {
                gblas::axpy(N, alpha, A.data(), 1, B.data(), 1, queue);
            });
    });
}

TEST_F(GPGPUTest, Xdot) {
    SCOPED_TRACE("Xdot");
    doTest([](auto const& queue) {
        blas_level1_r_test<float>(queue,
            [](auto N, auto& A, auto& B) {
                return cblas::dot(N, A.data(), 1, B.data(), 1);
            },
            [&](auto N, auto& A, auto& B, auto& R) {
                gblas::dot(N, A.data(), 1, B.data(), 1, R.data(), queue);
            });

        blas_level1_r_test<int32_t>(queue,
            [](auto N, auto& A, auto& B) {
                Tensor<int32_t> C({1});
                inner(A, B, &C);
                return C(0);
            },
            [&](auto N, auto& A, auto& B, auto& R) {
                gblas::dot(N, A.data(), 1, B.data(), 1, R.data(), queue);
            });

        blas_level1_r_test<int64_t>(queue,
            [](auto N, auto& A, auto& B) {
                Tensor<int64_t> C({1});
                inner(A, B, &C);
                return C(0);
            },
            [&](auto N, auto& A, auto& B, auto& R) {
                gblas::dot(N, A.data(), 1, B.data(), 1, R.data(), queue);
            });
    });
}

TEST_F(GPGPUTest, Xnrm2) {
    SCOPED_TRACE("Xnrm2");
    doTest([](auto const& queue) {
        blas_level1_r_test<float>(queue,
            [](auto N, auto& A, auto&) {
                return cblas::nrm2(N, A.data(), 1);
            },
            [&](auto N, auto& A, auto&, auto& R) {
                gblas::nrm2(N, A.data(), 1, R.data(), queue);
            });
    });
}

TEST_F(GPGPUTest, Xasum) {
    SCOPED_TRACE("Xasum");
    doTest([](auto const& queue) {
        blas_level1_r_test<float>(queue,
            [](auto N, auto& A, auto&) {
                return cblas::asum(N, A.data(), 1);
            },
            [&](auto N, auto& A, auto&, auto& R) {
                gblas::asum(N, A.data(), 1, R.data(), queue);
            });

        blas_level1_r_test<int32_t>(queue,
            [](auto N, auto& A, auto&) {
                int32_t r = 0;
                for (auto x : A)
                    r += abs(x);
                return r;
            },
            [&](auto N, auto& A, auto&, auto& R) {
                gblas::asum(N, A.data(), 1, R.data(), queue);
            });

        blas_level1_r_test<int64_t>(queue,
            [](auto N, auto& A, auto&) {
                int64_t r = 0;
                for (auto x : A)
                    r += abs(x);
                return r;
            },
            [&](auto N, auto& A, auto&, auto& R) {
                gblas::asum(N, A.data(), 1, R.data(), queue);
            });
    });
}

static void device_filter_test(int num_devices, const char* env, const char* expect) {
    auto filter = gpgpu::parseDeviceFilter(num_devices, env);
    std::string actual;
    for (size_t i = 0; i < filter.size(); i++) {
        if (filter[i])
            actual += std::to_string(i);
    }
    EXPECT_STREQ(expect, actual.c_str());
}

TEST_F(GPGPUTest, ParseDeviceFilter) {
    device_filter_test(6, nullptr, "012345");
    device_filter_test(6, "", "012345");

    device_filter_test(6, "3", "3");
    device_filter_test(6, "1,3,5", "135");
    device_filter_test(6, "4,2", "24");

    device_filter_test(6, "-3", "01245");
    device_filter_test(6, "-1,-3,-5", "024");
    device_filter_test(6, "-4,-2", "0135");

    device_filter_test(6, "2-5", "2345");
    device_filter_test(6, "2-4,3-5", "2345");

    device_filter_test(6, "2-5,-3", "245");
    device_filter_test(6, "-3,2-5", "245");
    device_filter_test(6, "-2-4", "015");

    device_filter_test(6, "3-10", "345");
    device_filter_test(6, "0-2", "012");
    device_filter_test(6, "0-10", "012345");

    device_filter_test(6, "-0", "12345");
    device_filter_test(6, "-10", "012345");

    device_filter_test(6, "2-1", "012345");
    device_filter_test(6, "2-1,3", "3");
    device_filter_test(6, "2-,3", "3");
    device_filter_test(6, "-,3", "3");
    device_filter_test(6, "3,-", "3");
    device_filter_test(6, ",,1,,2,,3,,", "123");
    device_filter_test(6, ".3-5", "345");
}

TEST(GPGPU, MultipleThreadContextActivation) {
    constexpr int N = 100;
    auto A = Tensor<int>::random({N}, -N, N).cast<float>();
    auto B = Tensor<int>::random({N}, -N, N).cast<float>();

    auto task = [&]() {
        auto dev_R = DevTensor<float>({1});
        auto dev_A = DevTensor<float>(A);
        auto dev_B = DevTensor<float>(B);

        inner(dev_A, dev_B, &dev_R);
        return dev_R.read()(0);
    };

    auto r1 = std::async(std::launch::async, task);
    auto r2 = std::async(std::launch::async, task);
    float r = cblas::dot(N, A.data(), 1, B.data(), 1);

    EXPECT_EQ(r1.get(), r);
    EXPECT_EQ(r2.get(), r);
}
