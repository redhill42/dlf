#include "xgetrf.hpp"

namespace gpgpu { namespace blas {

template <typename T>
Xgetrf<T>::Xgetrf(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/levelx/xgetrf.cl"
}) {}

template <typename T>
inline void xger(const size_t m, const size_t n, const size_t i,
                 Buffer<T>& A, const size_t a_offset, const size_t lda)
{
    ger(Layout::RowMajor, m-i-1, n-i-1, PrecisionTraits<T>::NegOne,
        A, a_offset + (i+1)*lda + i, lda,
        A, a_offset + i*lda + (i+1), 1,
        A, a_offset + (i+1)*lda + (i+1), lda);
}

template <typename T>
inline void xger(const size_t m, const size_t n, const size_t i,
                 Buffer<std::complex<T>>& A, const size_t a_offset, const size_t lda)
{
    geru(Layout::RowMajor, m-i-1, n-i-1, PrecisionTraits<std::complex<T>>::NegOne,
         A, a_offset + (i+1)*lda + i, lda,
         A, a_offset + i*lda + (i+1), 1,
         A, a_offset + (i+1)*lda + (i+1), lda);
}

template <typename T>
void Xgetrf<T>::DoGetrf(const size_t m, const size_t n,
                        Buffer<T>& A, const size_t a_offset, const size_t lda,
                        Buffer<int32_t>& ipiv, const size_t ipiv_offset)
{
    auto mnmin = std::min(m, n);
    for (size_t i = 0; i < mnmin; ++i) {
        auto kernel = program_.getKernel("pivot");
        kernel.setArguments(
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(i),
            A, static_cast<int>(a_offset), static_cast<int>(lda),
            ipiv, static_cast<int>(ipiv_offset));

        auto global = std::vector<size_t>{db_["WGS"]};
        auto local = std::vector<size_t>{db_["WGS"]};
        RunKernel(kernel, queue_, device_, global, local, nullptr);

        if (i < mnmin - 1) {
            xger(m, n, i, A, a_offset, lda);
        }
    }
}

template <typename T>
void Xgetrf<T>::DoGetrs(Transpose trans, size_t n, size_t nrhs,
                        const Buffer<T>& A, const size_t a_offset, const size_t lda,
                        const Buffer<int32_t>& ipiv, const size_t ipiv_offset,
                        Buffer<T>& B, const size_t b_offset, const size_t ldb)
{
    if (trans == Transpose::NoTrans) {
        // Apply row interchanges to the right hand sides
        DoLaswp(nrhs, B, b_offset, ldb, 0, n, ipiv, ipiv_offset, 1);

        // Solve L*X = B, overwriting B with X.
        trsm(Layout::RowMajor, Side::Left, Triangle::Lower, Transpose::NoTrans, Diagonal::Unit,
             n, nrhs, PrecisionTraits<T>::One, A, a_offset, lda, B, b_offset, ldb);

        // Solve U*X = B, overwriting B with X.
        trsm(Layout::RowMajor, Side::Left, Triangle::Upper, Transpose::NoTrans, Diagonal::NonUnit,
             n, nrhs, PrecisionTraits<T>::One, A, a_offset, lda, B, b_offset, ldb);
    } else {
        // Solve U**T * X = B, overwriting B with X
        trsm(Layout::RowMajor, Side::Left, Triangle::Upper, Transpose::Trans, Diagonal::Unit,
             n, nrhs, PrecisionTraits<T>::One, A, a_offset, lda, B, b_offset, ldb);

        // Solve L**T * X = B, overwriting B with X
        trsm(Layout::RowMajor, Side::Left, Triangle::Lower, Transpose::Trans, Diagonal::NonUnit,
             n, nrhs, PrecisionTraits<T>::One, A, a_offset, lda, B, b_offset, ldb);

        // Apply row interchanges to the solution vectors.
        DoLaswp(nrhs, B, b_offset, ldb, 0, n, ipiv, ipiv_offset, -1);
    }
}

template <typename T>
void Xgetrf<T>::DoLaswp(size_t n, Buffer<T>& A, const size_t a_offset, const size_t lda,
                        size_t k1, size_t k2,
                        const Buffer<int32_t> &ipiv, const size_t ip_offset, int ip_inc)
{
    auto kernel = program_.getKernel("laswp");
    kernel.setArguments(
        static_cast<int>(n), A, static_cast<int>(a_offset), static_cast<int>(lda),
        static_cast<int>(k1), static_cast<int>(k2),
        ipiv, static_cast<int>(ip_offset), ip_inc);

    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, nullptr);
}

template class Xgetrf<float>;
template class Xgetrf<double>;
template class Xgetrf<float2>;
template class Xgetrf<double2>;

}} // namespace gpgpu::blas
