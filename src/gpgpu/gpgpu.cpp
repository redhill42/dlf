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

using float2 = std::complex<float>;
using double2 = std::complex<double>;

static_assert(static_cast<clblast::Layout>(cblas::Layout::RowMajor) == clblast::Layout::kRowMajor);
static_assert(static_cast<clblast::Layout>(cblas::Layout::ColMajor) == clblast::Layout::kColMajor);
static_assert(static_cast<clblast::Transpose>(cblas::Transpose::NoTrans) == clblast::Transpose::kNo);
static_assert(static_cast<clblast::Transpose>(cblas::Transpose::Trans) == clblast::Transpose::kYes);
static_assert(static_cast<clblast::Transpose>(cblas::Transpose::ConjTrans) == clblast::Transpose::kConjugate);

template <typename T>
void scal(const size_t N, const T alpha, Buffer<T>& X, const size_t incX,
          Queue& queue, Event* event) {
    if (isOpenCL()) {
        auto xBuffer = *clBuffer::unwrap(X.raw());
        auto q = clQueue::unwrap(queue.raw());
        auto ev = event==nullptr ? nullptr : clEvent::unwrap(event->raw());
        clblast::Scal(N, alpha, xBuffer, 0, incX, q, ev);
    }
}

template void scal(const size_t, const float, Buffer<float>&, const size_t, Queue&, Event*);
template void scal(const size_t, const double, Buffer<double>&, const size_t, Queue&, Event*);
template void scal(const size_t, const float2, Buffer<float2>&, const size_t, Queue&, Event*);
template void scal(const size_t, const double2, Buffer<double2>&, const size_t, Queue&, Event*);

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
    if (isOpenCL()) {
        auto aBuffer = *clBuffer::unwrap(A.raw());
        auto bBuffer = *clBuffer::unwrap(B.raw());
        auto cBuffer = *clBuffer::unwrap(C.raw());
        auto q = clQueue::unwrap(queue.raw());
        auto e = event==nullptr ? nullptr : clEvent::unwrap(event->raw());
        clblast::Gemm(static_cast<clblast::Layout>(layout),
                      static_cast<clblast::Transpose>(transA),
                      static_cast<clblast::Transpose>(transB),
                      M, N, K,
                      alpha,
                      aBuffer, 0, lda,
                      bBuffer, 0, ldb,
                      beta,
                      cBuffer, 0, ldc,
                      q, e);
    }
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
