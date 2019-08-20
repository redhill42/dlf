#pragma once

#include <unordered_set>

namespace dlf {

//==-------------------------------------------------------------------------
// General matrix multiplication (Host)
//==-------------------------------------------------------------------------

template <typename T>
std::enable_if_t<!cblas::RequireBlasType<T>>
gemm(const T& alpha, const Tensor<T>& A, const Tensor<T>& B,
     const T& beta, Tensor<T>* C,
     bool transA = false, bool transB = false,
     Tensor<T>* = nullptr)
{
    assert(A.is_matrix() && B.is_matrix() && C->is_matrix());
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);
    const auto lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);
    if (k != p)
        throw shape_error("gemm: incompatible shape");
    C->resize({m, n});

    if (alpha == T(0)) {
        *C *= Tensor<T>::scalar(beta);
        return;
    }

    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, m, 32, 0, n, 32), [&](auto&& r) {
        size_t incX = transA ? lda : 1;
        size_t incY = transB ? 1 : ldb;
        for (size_t i = r.rows().begin(); i != r.rows().end(); i++) {
            T* pz = &C->data()[i * ldc + r.cols().begin()];
            for (size_t j = r.cols().begin(); j != r.cols().end(); j++) {
                const T* px = A.data() + (transA ? i : i*lda);
                const T* py = B.data() + (transB ? j*ldb : j);
                T v = beta == T(0) ? T(0) : *pz * beta;
                for (size_t t = 0; t < k; t++) {
                    v += alpha * *px * *py;
                    px += incX;
                    py += incY;
                }
                *pz++ = std::move(v);
            }
        }
    });
}

template <typename T>
std::enable_if_t<cblas::RequireBlasType<T>>
gemm(const T& alpha, const Tensor<T>& A, const Tensor<T>& B,
     const T& beta, Tensor<T>* C,
     bool transA = false, bool transB = false,
     Tensor<T>* = nullptr)
{
    assert(A.is_matrix() && B.is_matrix() && C->is_matrix());
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);

    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);
    if (k != p)
        throw shape_error("gemm: incompatible shape");
    C->resize({m, n});

    cblas::gemm(cblas::Layout::RowMajor,
                transA ? cblas::Transpose::Trans : cblas::Transpose::NoTrans,
                transB ? cblas::Transpose::Trans : cblas::Transpose::NoTrans,
                m, n, k, alpha, A.data(), A.stride(0), B.data(), B.stride(0),
                beta, C->data(), C->stride(0));
}

template <typename T>
inline size_t gemmWorkspaceSize(
    const Tensor<T>&, const Tensor<T>&, const Tensor<T>&,
    bool = false, bool = false)
{
    return 0; // API compatible to DevTensor
}

//==-------------------------------------------------------------------------
// General matrix multiplication (Device)
//==-------------------------------------------------------------------------

template <typename T>
void gemm(const T& alpha, const DevTensor<T>& A, const DevTensor<T>& B,
          const T& beta, DevTensor<T>* C,
          bool transA = false, bool transB = false,
          DevTensor<T>* work = nullptr)
{
    assert(A.is_matrix() && B.is_matrix() && C->is_matrix());
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);

    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);
    if (k != p)
        throw shape_error("gemm: incompatible shape");
    C->resize({m, n});

    gblas::gemm(gblas::Layout::RowMajor,
                transA ? gblas::Transpose::Trans : gblas::Transpose::NoTrans,
                transB ? gblas::Transpose::Trans : gblas::Transpose::NoTrans,
                m, n, k,
                alpha,
                A.data(), A.stride(0),
                B.data(), B.stride(0),
                beta,
                C->data(), C->stride(0),
                work == nullptr ? nullptr : &work->data());
}

template <typename T>
inline size_t gemmWorkspaceSize(
    const DevTensor<T>& A, const DevTensor<T>& B, const DevTensor<T>& C,
    bool transA = false, bool transB = false)
{
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);

    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);
    if (k != p || m != C.extent(0) || n != C.extent(1))
        throw shape_error("gemm: incompatible shape");

    return gblas::gemmTempBufferSize<T>(
        gblas::Layout::RowMajor,
        transA ? gblas::Transpose::Trans : gblas::Transpose::NoTrans,
        transB ? gblas::Transpose::Trans : gblas::Transpose::NoTrans,
        m, n, k, 0, A.stride(0), 0, B.stride(0), 0, C.stride(0));
}

//==-------------------------------------------------------------------------
// General matrix multiplication (Uniform)
//==-------------------------------------------------------------------------

template <typename TensorT>
inline enable_if_non_view_tensor<TensorT, void>
gemm(const tensor_value_type<TensorT>& alpha, const TensorT& A, const TensorT& B,
     const tensor_value_type<TensorT>& beta, const TensorT& C, TensorT& Y,
     bool transA = false, bool transB = false, TensorT* work = nullptr)
{
    broadcast(C, Y);
    gemm(alpha, A, B, beta, &Y, transA, transB, work);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
gemm(const tensor_value_type<TensorT>& alpha, const TensorT& A, const TensorT& B,
     const tensor_value_type<TensorT>& beta, const TensorT& C,
     bool transA = false, bool transB = false, TensorT* work = nullptr)
{
    assert(A.is_matrix() && B.is_matrix());
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);

    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);
    assert(k == p);

    tensor_type<TensorT> Y = C.broadcast({m, n});
    gemm(alpha, A, B, beta, &Y, transA, transB, work);
    return Y;
}

//==-------------------------------------------------------------------------
// Extended matrix multiplication
//==-------------------------------------------------------------------------

namespace detail {
inline int matmul_broadcast(Shape& shapeA, Shape& shapeB, Shape& shapeC) {
    if (shapeA.rank() == 0 || shapeB.rank() == 0)
        throw shape_error("matmul: Input tensors of wrong rank (0).");

    // First promote each shape to at least rank-2. This logic is
    // specific to matmul, not generic broadcasting.
    if (shapeA.rank() == 1)
        shapeA = shapeA.unsqueeze(0);
    if (shapeB.rank() == 1)
        shapeB = shapeB.unsqueeze(1);

    auto dimsA = shapeA.extents();
    auto dimsB = shapeB.extents();

    // Check for compatible matrix multiply dimensions
    auto m = dimsA[dimsA.size() - 2];
    auto k = dimsA[dimsA.size() - 1];
    auto p = dimsB[dimsB.size() - 2];
    auto n = dimsB[dimsB.size() - 1];

    if (k != p)
        throw shape_error("matmul: Incompatible dimensions for matrix multiplication");

    // Now call out to generic multidimensional broadcasting for
    // the broadcastable prefixes.
    auto prefixShape = Shape::broadcast(
        Shape(std::vector<size_t>{dimsA.begin(), dimsA.end() - 2}),
        Shape(std::vector<size_t>{dimsB.begin(), dimsB.end() - 2})
    );

    // Back to matmul-specific. Add the trailing dimensions back in.
    dimsA = prefixShape.extents();
    dimsA.push_back(m);
    dimsA.push_back(k);
    shapeA = shapeA.broadcast(Shape(dimsA));

    dimsB = prefixShape.extents();
    dimsB.push_back(k);
    dimsB.push_back(n);
    shapeB = shapeB.broadcast(Shape(dimsB));

    auto dimsC = prefixShape.extents();
    dimsC.push_back(m);
    dimsC.push_back(n);
    shapeC = Shape(dimsC);

    return prefixShape.rank() == 0 ? 1 : prefixShape.size();
}

template <int = 0>
int batch_offset(const Shape& shape, int batch) {
    int ret = shape.offset();
    for (int i = shape.rank() - 3; i >= 0; --i) {
        auto dim = shape.extent(i);
        auto index = batch % dim;
        batch /= dim;
        ret += index * shape.stride(i);
    }
    return ret;
}

template <int = 0>
bool is_contiguous_strides(const Shape& shape) {
    if (shape.rank() <= 2)
        return true;

    int stride = shape.stride(shape.rank() - 3);
    if (stride == 0) {
        for (int i = shape.rank() - 4; i >= 0; --i) {
            if (shape.stride(i) != 0)
                return false;
        }
    } else {
        for (int i = shape.rank() - 4; i >= 0; --i) {
            if (shape.stride(i) != stride*shape.extent(i))
                return false;
            stride = shape.stride(i);
        }
    }
    return true;
}

template <typename T>
std::enable_if_t<cblas::RequireBlasType<T>>
matmul_cpu(int m, int n, int k,
           const T* A, int lda,
           const T* B, int ldb,
           T* C, int ldc)
{
    if (m == 1 && n == 1) {
        *C = cblas::dot(k, A, 1, B, 1);
    } else {
        cblas::gemm(
            cblas::Layout::RowMajor,
            cblas::Transpose::NoTrans, cblas::Transpose::NoTrans,
            m, n, k, 1.0, A, lda, B, ldb, 0.0, C, ldc);
    }
}

template <typename T>
std::enable_if_t<!cblas::RequireBlasType<T>>
matmul_cpu(int m, int n, int k,
           const T* A, int lda,
           const T* B, int ldb,
           T* C, int ldc)
{
    if (m == 1 && n == 1) {
        *C = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, k, GRAINSIZE),
            T{},
            [=](auto r, T sum) {
                auto px = A + r.begin();
                auto py = B + r.begin();
                for (size_t k = r.size(); k-- != 0; )
                    sum += *px++ * *py++;
                return sum;
            },
            std::plus<T>());
    } else {
        tbb::parallel_for(tbb::blocked_range2d<size_t>(0, m, 32, 0, n, 32), [=](auto r) {
            for (size_t i = r.rows().begin(); i != r.rows().end(); i++) {
                for (size_t j = r.cols().begin(); j != r.cols().end(); j++) {
                    T v{};
                    for (size_t t = 0; t < k; t++)
                        v += A[i * lda + t] * B[t * ldb + j];
                    C[i * ldc + j] = std::move(v);
                }
            }
        });
    }
}

template <typename T>
void matmul(int m, int n, int k,
            const Shape& shapeA, const T* A, int lda,
            const Shape& shapeB, const T* B, int ldb,
            T* C, int ldc,
            int batch_size)
{
    if (batch_size == 1) {
        matmul_cpu(m, n, k, A+shapeA.offset(), lda, B+shapeB.offset(), ldb, C, ldc);
    } else {
        tbb::parallel_for(tbb::blocked_range<int>(0, batch_size, 16), [=](auto r) {
            for (int p = r.begin(); p < r.end(); p++) {
                matmul_cpu(
                    m, n, k,
                    A + batch_offset(shapeA, p), lda,
                    B + batch_offset(shapeB, p), ldb,
                    C + p*m*n, ldc);
            }
        });
    }
}

template <typename T>
void matmul(int m, int n, int k,
            const Shape& shapeA, const gpgpu::Buffer<T>& A, int lda,
            const Shape& shapeB, const gpgpu::Buffer<T>& B, int ldb,
            gpgpu::Buffer<T>& C, int ldc,
            int batch_size)
{
    constexpr auto RowMajor = gpgpu::blas::Layout::RowMajor;
    constexpr auto NoTrans = gpgpu::blas::Transpose::NoTrans;

    if (batch_size == 1) {
        if (m == 1 && n == 1) {
            gpgpu::blas::dot(
                k,
                A, shapeA.offset(), shapeA.stride(-1),
                B, shapeB.offset(), shapeB.stride(-2),
                C, 0);
        } else {
            gpgpu::blas::gemm(
                RowMajor, NoTrans, NoTrans,
                m, n, k,
                T{1}, A, lda, B, ldb,
                T{0}, C, ldc);
        }
    } else if (is_contiguous_strides(shapeA) && is_contiguous_strides(shapeB)) {
        gpgpu::blas::gemmStridedBatched(
            RowMajor, NoTrans, NoTrans,
            m, n, k,
            T{1},
            A, shapeA.offset(), lda, shapeA.stride(-3),
            B, shapeB.offset(), ldb, shapeB.stride(-3),
            T{0},
            C, 0, ldc, m*n,
            batch_size);
    } else {
        std::vector<T> alphas(batch_size);
        std::vector<T> betas(batch_size);
        std::vector<size_t> a_offsets(batch_size);
        std::vector<size_t> b_offsets(batch_size);
        std::vector<size_t> c_offsets(batch_size);

        for (int p = 0; p < batch_size; p++) {
            alphas[p]    = T{1};
            betas[p]     = T{0};
            a_offsets[p] = batch_offset(shapeA, p);
            b_offsets[p] = batch_offset(shapeB, p);
            c_offsets[p] = p*m*n;
        }

        gpgpu::blas::gemmBatched(
            RowMajor, NoTrans, NoTrans,
            m, n, k,
            &alphas[0],
            A, &a_offsets[0], lda,
            B, &b_offsets[0], ldb,
            &betas[0],
            C, &c_offsets[0], ldc,
            batch_size);
    }
}
} // namespace detail

/**
 * Matrix product of two tensors.
 *
 * The behavior depends on the arguments in the following way.
 *
 *  - If both arguments are 2-D they are multiplied like conventional matrices.
 *  - If either argument is N-D, N > 2, it is treated as a stack of matrices
 *    residing in the last two indexes and broadcast accordingly.
 *  - If the first argument is 1-D, it is promoted to a matrix by prepending
 *    a 1 to its dimensions. After matrix multiplication the prepended 1 is
 *    removed.
 *  - If the second argument is 1-D, it is promoted to a matrix by appending
 *    a 1 to its dimensions. After matrix multiplication the appended 1 is
 *    removed.
 *
 * Multiplication by a scalar is not allowed, use * instead. Note that
 * multiplying a stack of matrices with a vector will result in a stack of
 * vectors, but matmul will not recognize it as such.
 */
template <typename TensorT>
enable_if_non_view_tensor<TensorT, tensor_type<TensorT>&>
matmul(const TensorT& A, const TensorT& B, tensor_type<TensorT>& C) {
    Shape shapeA = A.shape();
    Shape shapeB = B.shape();
    Shape shapeC;

    int batch_size = detail::matmul_broadcast(shapeA, shapeB, shapeC);

    int m = shapeA.extent(-2);
    int k = shapeA.extent(-1);
    int n = shapeB.extent(-1);

    int lda = std::max<int>(shapeA.stride(-2), k);
    int ldb = std::max<int>(shapeB.stride(-2), n);
    int ldc = std::max<int>(shapeC.stride(-2), n);

    if (A.rank() == 1)
        shapeC = shapeC.squeeze(-2);
    if (B.rank() == 1)
        shapeC = shapeC.squeeze(-1);
    if (shapeC.rank() == 0)
        shapeC = Shape({1});
    C.resize(shapeC);

    detail::matmul(
        m, n, k,
        shapeA, A.data(), lda,
        shapeB, B.data(), ldb,
        C.data(), ldc,
        batch_size);

    return C;
}

template <typename TensorT>
inline enable_if_non_view_tensor<TensorT> matmul(const TensorT& A, const TensorT& B) {
    TensorT C;
    matmul(A, B, C);
    return C;
}

//==-------------------------------------------------------------------------
// Matrix exponentiation
//==-------------------------------------------------------------------------

template <typename TensorT>
enable_if_non_view_tensor<TensorT> matpow(TensorT&& A, long n) {
    if (!A.is_inner_square())
        throw shape_error("matpow: square matrix required");
    if (n < 0)
        throw std::logic_error("matpow: negative exponentiation is not supported");

    if (n == 0)
        return Tensor<tensor_value_type<TensorT>>::identity(2, A.extent(0)); // FIXME
    if (n == 1)
        return std::forward<TensorT>(A);
    n--;

    auto x = std::forward<TensorT>(A);
    auto y = x, t = x;
    while (n > 0) {
        if (n & 1)
            std::swap(y, matmul(x, y, t));
        std::swap(x, matmul(x, x, t));
        n >>= 1;
    }
    return y;
}

//==-------------------------------------------------------------------------
// Dot product
//==-------------------------------------------------------------------------

template <typename TensorT>
inline enable_if_non_view_tensor<TensorT>
dot(const TensorT& A, const TensorT& B) {
    if (A.rank() > 2 || B.rank() > 2)
        throw shape_error("dot: unsupported tensor shape");
    TensorT C{};
    matmul(A, B, C);
    return C;
}

namespace dot_product {
/**
 * We use comma operator to represent dot product, because C++ doesn't have dot
 * operator yet, and comma and dot are looks similar. To use the comma operator
 * be sure to enclose the expression in parentheses to avoid ambiguity. That is,
 * use
 *     auto z = (x , y)
 * instead of
 *     auto z = x, y
 */
template <typename TensorT>
enable_if_non_view_tensor<TensorT>
inline operator,(const TensorT& lhs, const TensorT& rhs) {
    return dot(lhs, rhs);
}
} // namespace dot_product

//==-------------------------------------------------------------------------
// Cross product
//==-------------------------------------------------------------------------

/**
 * The cross product on tensors is typically referred to as the tensor product.
 * Given a tensor a of order q with dimensions (i1, ..., iq), and a tensor b
 * of order r with dimensions (j1, ..., jr), their cross product c is of order
 * q + r with dimensions (k1, ..., kq+r) which are the i dimensions followed
 * by the j dimensions.
 */
template <typename LHS, typename RHS, typename Fn>
enable_if_tensors<LHS, RHS, Fn>
cross(const LHS& A, const RHS& B, Fn f) {
    std::vector<int> axesA(B.rank()), axesB(A.rank());
    std::iota(axesA.begin(), axesA.end(), A.rank()); // unsqueeze right
    std::iota(axesB.begin(), axesB.end(), 0);        // unsqueeze left
    return transform(tensor_view_type<LHS>(A.shape().unsqueeze(axesA), A),
                     tensor_view_type<RHS>(B.shape().unsqueeze(axesB), B),
                     f);
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::multiplies<>>
inline cross(const LHS& lhs, const RHS& rhs) {
    return cross(lhs, rhs, xfn::multiplies<>());
}

} // namespace dlf
