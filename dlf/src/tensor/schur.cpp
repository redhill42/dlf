#include "tensor.h"

namespace dlf {

Tensor<std::complex<float>>
schur(const Tensor<float>& A, Tensor<float>& Q, Tensor<float>& U) {
    if (!A.is_square())
        throw shape_error("schur: expected square matrix");
    reorder(A, U);
    Q.resize(A.shape());

    auto n = A.extent(0);
    std::vector<float> wr(n), wi(n);
    lapack_int sdim;

    cblas::gees('V', 'N', nullptr, n, U.data(), U.stride(0),
                &sdim, wr.data(), wi.data(), Q.data(), Q.stride(0));

    auto W = Tensor<std::complex<float>>({n});
    for (int i = 0; i < n; ++i)
        W(i) = std::complex<float>(wr[i], wi[i]);
    return W;
}

Tensor<std::complex<double>>
schur(const Tensor<double>& A, Tensor<double>& Q, Tensor<double>& U) {
    if (!A.is_square())
        throw shape_error("schur: expected square matrix");
    reorder(A, U);
    Q.resize(A.shape());

    auto n = A.extent(0);
    std::vector<double> wr(n), wi(n);
    lapack_int sdim;

    cblas::gees('V', 'N', nullptr, n, U.data(), U.stride(0),
                &sdim, wr.data(), wi.data(), Q.data(), Q.stride(0));

    auto W = Tensor<std::complex<double>>({n});
    for (int i = 0; i < n; ++i)
        W(i) = std::complex<float>(wr[i], wi[i]);
    return W;
}

Tensor<std::complex<float>>
schur(const Tensor<std::complex<float>>& A,
      Tensor<std::complex<float>>& Q,
      Tensor<std::complex<float>>& U)
{
    if (!A.is_square())
        throw shape_error("schur: expected square matrix");
    reorder(A, U);
    Q.resize(A.shape());

    auto n = A.extent(0);
    auto W = Tensor<std::complex<float>>({n});
    lapack_int sdim;

    cblas::gees('V', 'N', nullptr, n, U.data(), U.stride(0),
                &sdim, W.data(), Q.data(), Q.stride(0));
    return W;
}

Tensor<std::complex<double>>
schur(const Tensor<std::complex<double>>& A,
      Tensor<std::complex<double>>& Q,
      Tensor<std::complex<double>>& U)
{
    if (!A.is_square())
        throw shape_error("schur: expected square matrix");
    reorder(A, U);
    Q.resize(A.shape());

    auto n = A.extent(0);
    auto W = Tensor<std::complex<double>>({n});
    lapack_int sdim;

    cblas::gees('V', 'N', nullptr, n, U.data(), U.stride(0),
                &sdim, W.data(), Q.data(), Q.stride(0));
    return W;
}

} // namespace dlf
