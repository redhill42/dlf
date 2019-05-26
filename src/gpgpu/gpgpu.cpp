#include <complex>
#include "clblast.h"
#include "cl.hpp"
#include "cu.hpp"

namespace gpgpu {

Platform probe() {
    auto cu = gpgpu::cu::probe();
    if (cu != nullptr)
        return Platform(std::move(cu));

    auto cl = gpgpu::cl::probe();
    if (cl != nullptr)
        return Platform(std::move(cl));

    throw RuntimeError("No OpenCL or CUDA platform available");
}

std::vector<Device> Platform::devices(DeviceType type) const {
    auto raw_devices = m_raw->devices(type);
    auto devices = std::vector<Device>();
    for (const auto& dev : raw_devices)
        devices.emplace_back(*this, dev);
    return devices;
}

} // namespace gpgpu

//==-------------------------------------------------------------------------

namespace gpgpu::blas {

using namespace gpgpu::cl;
using namespace gpgpu::cu;

using float2 = std::complex<float>;
using double2 = std::complex<double>;

static_assert(static_cast<clblast::Layout>(blas::Layout::RowMajor) == clblast::Layout::kRowMajor);
static_assert(static_cast<clblast::Layout>(blas::Layout::ColMajor) == clblast::Layout::kColMajor);
static_assert(static_cast<clblast::Transpose>(blas::Transpose::NoTrans) == clblast::Transpose::kNo);
static_assert(static_cast<clblast::Transpose>(blas::Transpose::Trans) == clblast::Transpose::kYes);
static_assert(static_cast<clblast::Transpose>(blas::Transpose::ConjTrans) == clblast::Transpose::kConjugate);

template <typename T>
constexpr cudaDataType CudaType = static_cast<cudaDataType>(-1);

template <> constexpr cudaDataType CudaType<float>   = cudaDataType::CUDA_R_32F;
template <> constexpr cudaDataType CudaType<double>  = cudaDataType::CUDA_R_64F;
template <> constexpr cudaDataType CudaType<float2>  = cudaDataType::CUDA_C_32F;
template <> constexpr cudaDataType CudaType<double2> = cudaDataType::CUDA_C_64F;

static cublasOperation_t cudaOp(Transpose trans) {
    switch (trans) {
    case Transpose::NoTrans:
        return cublasOperation_t::CUBLAS_OP_N;
    case Transpose::Trans:
        return cublasOperation_t::CUBLAS_OP_T;
    case Transpose::ConjTrans:
        return cublasOperation_t::CUBLAS_OP_C;
    }
}

template <typename T, typename CL, typename CU>
static void dispatch(Queue& queue, Event* event, CL&& cl, CU&& cu) {
    if (isOpenCL()) {
        auto q = clQueue::unwrap(queue.raw());
        auto e = event==nullptr ? nullptr : clEvent::unwrap(event->raw());
        cl(q, e);
    } else {
        auto q = static_cast<cuQueue&>(queue.raw());
        cu(q.getCublasHandle(), CudaType<T>);
    }
}

//==-------------------------------------------------------------------------
// BLAS level-1 (vector-vector) routines
//==-------------------------------------------------------------------------

template <typename T>
void asum(size_t N, const Buffer<T>& X, size_t incX, Buffer<T>& result, Queue& queue, Event* event) {
    dispatch<T>(queue, event,
        [&](auto q, auto e) {
            clblast::Asum<T>(N, *clBuffer::unwrap(result), 0, *clBuffer::unwrap(X), 0, incX, q, e);
        },

        [&](auto h, auto t) {
            cublasAsumEx(h, N, cuBuffer::unwrap(X), t, incX, cuBuffer::unwrap(result), t, t);
        });
}

template void asum(size_t, const Buffer<float>&, size_t, Buffer<float>&, Queue&, Event*);
template void asum(size_t, const Buffer<double>&, size_t, Buffer<double>&, Queue&, Event*);
template void asum(size_t, const Buffer<float2>&, size_t, Buffer<float2>&, Queue&, Event*);
template void asum(size_t, const Buffer<double2>&, size_t, Buffer<double2>&, Queue&, Event*);

template <typename T>
void axpy(size_t N, const T alpha, const Buffer<T>& X, size_t incX, Buffer<T>& Y, size_t incY,
          Queue& queue, Event* event) {
    dispatch<T>(queue, event,
        [&](auto q, auto e) {
            clblast::Axpy<T>(N, alpha, *clBuffer::unwrap(X), 0, incX, *clBuffer::unwrap(Y), 0, incY, q, e);
        },

        [&](auto h, auto t) {
            cublasAxpyEx(h, N, &alpha, t, cuBuffer::unwrap(X), t, incX, cuBuffer::unwrap(Y), t, incY, t);
        });
}

template void axpy(size_t, const float, const Buffer<float>&, size_t, Buffer<float>&, size_t, Queue&, Event*);
template void axpy(size_t, const double, const Buffer<double>&, size_t, Buffer<double>&, size_t, Queue&, Event*);
template void axpy(size_t, const float2, const Buffer<float2>&, size_t, Buffer<float2>&, size_t, Queue&, Event*);
template void axpy(size_t, const double2, const Buffer<double2>&, size_t, Buffer<double2>&, size_t, Queue&, Event*);

template <typename T>
void copy(size_t N, const Buffer<T>& X, size_t incX, Buffer<T>& Y, size_t incY, Queue& queue, Event* event) {
    dispatch<T>(queue, event,
        [&](auto q, auto e) {
            clblast::Copy<T>(N, *clBuffer::unwrap(X), 0, incX, *clBuffer::unwrap(Y), 0, incY, q, e);
        },

        [&](auto h, auto t) {
            cublasCopyEx(h, N, cuBuffer::unwrap(X), t, incX, cuBuffer::unwrap(Y), t, incY);
        });
}

template void copy(size_t, const Buffer<float>&, size_t, Buffer<float>&, size_t, Queue&, Event*);
template void copy(size_t, const Buffer<double>&, size_t, Buffer<double>&, size_t, Queue&, Event*);
template void copy(size_t, const Buffer<float2>&, size_t, Buffer<float2>&, size_t, Queue&, Event*);
template void copy(size_t, const Buffer<double2>&, size_t, Buffer<double2>&, size_t, Queue&, Event*);

template <typename T>
void dot(size_t N, const Buffer<T>& X, size_t incX, const Buffer<T>& Y, size_t incY,
         Buffer<T>& result, Queue& queue, Event* event) {
    dispatch<T>(queue, event,
        [&](auto q, auto e) {
            clblast::Dot<T>(N, *clBuffer::unwrap(result), 0,
                            *clBuffer::unwrap(X), 0, incX, *clBuffer::unwrap(Y), 0, incY,
                            q, e);
        },

        [&](auto h, auto t) {
            cublasDotEx(h, N, cuBuffer::unwrap(X), t, incX, cuBuffer::unwrap(Y), t, incY,
                        cuBuffer::unwrap(result), t, t);
        });
}

template void dot(size_t, const Buffer<float>&, size_t, const Buffer<float>&, size_t, Buffer<float>&, Queue&, Event*);
template void dot(size_t, const Buffer<double>&, size_t, const Buffer<double>&, size_t, Buffer<double>&, Queue&, Event*);

template <typename T>
void nrm2(size_t N, Buffer<T>& X, size_t incX, Buffer<T>& result, Queue& queue, Event* event) {
    dispatch<T>(queue, event,
        [&](auto q, auto e) {
            clblast::Nrm2<T>(N, *clBuffer::unwrap(result), 0, *clBuffer::unwrap(X), 0, incX, q, e);
        },

        [&](auto h, auto t) {
            cublasNrm2Ex(h, N, cuBuffer::unwrap(X), t, incX, cuBuffer::unwrap(result), t, t);
        });
}

template void nrm2(size_t, Buffer<float>&, size_t, Buffer<float>&, Queue&, Event*);
template void nrm2(size_t, Buffer<double>&, size_t, Buffer<double>&, Queue&, Event*);
template void nrm2(size_t, Buffer<float2>&, size_t, Buffer<float2>&, Queue&, Event*);
template void nrm2(size_t, Buffer<double2>&, size_t, Buffer<double2>&, Queue&, Event*);

template <typename T>
void scal(const size_t N, const T alpha, Buffer<T>& X, const size_t incX,
          Queue& queue, Event* event) {
    dispatch<T>(queue, event,
        [&](auto q, auto e) {
            clblast::Scal<T>(N, alpha, *clBuffer::unwrap(X), 0, incX, q, e);
        },

        [&](auto h, auto t) {
            cublasScalEx(h, N, &alpha, t, cuBuffer::unwrap(X), t, incX, t);
        });
}

template void scal(const size_t, const float, Buffer<float>&, const size_t, Queue&, Event*);
template void scal(const size_t, const double, Buffer<double>&, const size_t, Queue&, Event*);
template void scal(const size_t, const float2, Buffer<float2>&, const size_t, Queue&, Event*);
template void scal(const size_t, const double2, Buffer<double2>&, const size_t, Queue&, Event*);

template <typename T>
void swap(size_t N, Buffer<T>& X, size_t incX, Buffer<T>& Y, size_t incY, Queue& queue, Event* event) {
    dispatch<T>(queue, event,
        [&](auto q, auto e) {
            clblast::Swap<T>(N, *clBuffer::unwrap(X), 0, incX, *clBuffer::unwrap(Y), 0, incY, q, e);
        },

        [&](auto h, auto t) {
            cublasSwapEx(h, N, cuBuffer::unwrap(X), t, incX, cuBuffer::unwrap(Y), t, incY);
        });
}

template void swap(size_t, Buffer<float>&, size_t, Buffer<float>&, size_t, Queue&, Event*);
template void swap(size_t, Buffer<double>&, size_t, Buffer<double>&, size_t, Queue&, Event*);
template void swap(size_t, Buffer<float2>&, size_t, Buffer<float2>&, size_t, Queue&, Event*);
template void swap(size_t, Buffer<double2>&, size_t, Buffer<double2>&, size_t, Queue&, Event*);

//==-------------------------------------------------------------------------
// BLAS level-2 (matrix-vector) routines
//==-------------------------------------------------------------------------

template <typename T>
void cublasGemv(cublasHandle_t handle, cublasOperation_t transA,
                size_t M, size_t N, const T* alpha,
                const T* A, size_t lda,
                const T* X, size_t incX, const T* beta,
                T* Y, size_t incY);

template <>
inline void cublasGemv(cublasHandle_t handle, cublasOperation_t transA,
                       size_t M, size_t N, const float* alpha,
                       const float* A, size_t lda,
                       const float* X, size_t incX,
                       const float* beta, float* Y, size_t incY) {
    cublasSgemv(handle, transA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template <>
inline void cublasGemv(cublasHandle_t handle, cublasOperation_t transA,
                       size_t M, size_t N, const double* alpha,
                       const double* A, size_t lda,
                       const double* X, size_t incX,
                       const double* beta, double* Y, size_t incY) {
    cublasDgemv(handle, transA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template <>
inline void cublasGemv(cublasHandle_t handle, cublasOperation_t transA,
                       size_t M, size_t N, const float2* alpha,
                       const float2* A, size_t lda,
                       const float2* X, size_t incX,
                       const float2* beta, float2* Y, size_t incY) {
    cublasCgemv(handle, transA, M, N,
                reinterpret_cast<const cuComplex*>(alpha),
                reinterpret_cast<const cuComplex*>(A), lda,
                reinterpret_cast<const cuComplex*>(X), incX,
                reinterpret_cast<const cuComplex*>(beta),
                reinterpret_cast<cuComplex*>(Y), incY);
}

template <>
inline void cublasGemv(cublasHandle_t handle, cublasOperation_t transA,
                       size_t M, size_t N, const double2* alpha,
                       const double2* A, size_t lda,
                       const double2* X, size_t incX,
                       const double2* beta, double2* Y, size_t incY) {
    cublasZgemv(handle, transA, M, N,
                reinterpret_cast<const cuDoubleComplex*>(alpha),
                reinterpret_cast<const cuDoubleComplex*>(A), lda,
                reinterpret_cast<const cuDoubleComplex*>(X), incX,
                reinterpret_cast<const cuDoubleComplex*>(beta),
                reinterpret_cast<cuDoubleComplex*>(Y), incY);
}

// General matrix-vector multiplication
template <typename T>
void gemv(Layout layout, Transpose transA, size_t M, size_t N, const T alpha,
          const Buffer<T>& A, size_t lda, const Buffer<T>& X, size_t incX,
          const T beta, Buffer<T>& Y, size_t incY,
          Queue& queue, Event* event)
{
    dispatch<T>(queue, event,
        [&](auto q, auto e) {
            clblast::Gemv<T>(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Transpose>(transA),
                             M, N, alpha,
                             *clBuffer::unwrap(A), 0, lda,
                             *clBuffer::unwrap(X), 0, incX, beta,
                             *clBuffer::unwrap(Y), 0, incY,
                             q, e);
        },

        [&](auto h, auto t) {
            cublasGemv<T>(h, cudaOp(transA), M, N,
                          reinterpret_cast<const T*>(&alpha),
                          reinterpret_cast<const T*>(cuBuffer::unwrap(A)), lda,
                          reinterpret_cast<const T*>(cuBuffer::unwrap(X)), incX,
                          reinterpret_cast<const T*>(&beta),
                          reinterpret_cast<T*>(cuBuffer::unwrap(Y)), incY);
        });
}

template void gemv(Layout, Transpose, size_t, size_t, const float,
                   const Buffer<float>&, size_t, const Buffer<float>&, size_t,
                   const float, Buffer<float>&, size_t, Queue&, Event*);
template void gemv(Layout, Transpose, size_t, size_t, const double,
                   const Buffer<double>&, size_t, const Buffer<double>&, size_t,
                   const double, Buffer<double>&, size_t, Queue&, Event*);
template void gemv(Layout, Transpose, size_t, size_t, const float2,
                   const Buffer<float2>&, size_t, const Buffer<float2>&, size_t,
                   const float2, Buffer<float2>&, size_t, Queue&, Event*);
template void gemv(Layout, Transpose, size_t, size_t, const double2,
                   const Buffer<double2>&, size_t, const Buffer<double2>&, size_t,
                   const double2, Buffer<double2>&, size_t, Queue&, Event*);

//==-------------------------------------------------------------------------
// BLAS level-3 (matrix-matrix) routines
//==-------------------------------------------------------------------------

template <typename T>
void gemm(const Layout layout, const Transpose transA, const Transpose transB,
          const size_t M, const size_t N, const size_t K,
          const T alpha,
          const Buffer<T>& A, const size_t lda,
          const Buffer<T>& B, const size_t ldb,
          const T beta,
          Buffer<T>& C, const size_t ldc,
          Queue& queue, Event* event)
{
    dispatch<T>(queue, event,
        [&](auto q, auto e) {
            clblast::Gemm<T>(static_cast<clblast::Layout>(layout),
                             static_cast<clblast::Transpose>(transA),
                             static_cast<clblast::Transpose>(transB),
                             M, N, K, alpha,
                             *clBuffer::unwrap(A), 0, lda,
                             *clBuffer::unwrap(B), 0, ldb, beta,
                             *clBuffer::unwrap(C), 0, ldc,
                             q, e);
        },

        [&](auto handle, auto type) {
            cublasGemmEx(handle, cudaOp(transA), cudaOp(transB),
                         M, N, K, &alpha,
                         cuBuffer::unwrap(A), type, lda,
                         cuBuffer::unwrap(B), type, ldb, &beta,
                         cuBuffer::unwrap(C), type, ldc,
                         type, CUBLAS_GEMM_DEFAULT);
        });
}

template void gemm(const Layout, const Transpose, const Transpose,
                   const size_t, const size_t, const size_t, const float,
                   const Buffer<float>&, const size_t, const Buffer<float>&, const size_t,
                   const float, Buffer<float>&, const size_t, Queue&, Event*);
template void gemm(const Layout, const Transpose, const Transpose,
                   const size_t, const size_t, const size_t, const double,
                   const Buffer<double>&, const size_t, const Buffer<double>&, const size_t,
                   const double, Buffer<double>&, const size_t, Queue&, Event*);
template void gemm(const Layout, const Transpose, const Transpose,
                   const size_t, const size_t, const size_t, const float2,
                   const Buffer<float2>&, const size_t, const Buffer<float2>&, const size_t,
                   const float2, Buffer<float2>&, const size_t, Queue&, Event*);
template void gemm(const Layout, const Transpose, const Transpose,
                   const size_t, const size_t, const size_t, const double2,
                   const Buffer<double2>&, const size_t, const Buffer<double2>&, const size_t,
                   const double2, Buffer<double2>&, const size_t, Queue&, Event*);

} // namespace gpgpu::blas
