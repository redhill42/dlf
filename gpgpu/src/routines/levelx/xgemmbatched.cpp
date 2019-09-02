
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the XgemmBatched class (see the header for information about the class).
//
// =================================================================================================

#include "routines/levelx/xgemmbatched.hpp"
#include "routines/level3/xgemm.hpp"

#include <string>
#include <vector>

namespace gpgpu { namespace blas {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
XgemmBatched<T>::XgemmBatched(const Queue& queue, Event* event, const std::string& name):
    Routine(queue, event, name,
            {"Copy","Pad","Transpose","Padtranspose","Xgemm","XgemmDirect","GemmRoutine"},
            PrecisionValue<T>(), {}, {
    #include "../../kernels/level3/level3.cl"
    #include "../../kernels/level3/copy_fast.cl"
    #include "../../kernels/level3/copy_pad.cl"
    #include "../../kernels/level3/transpose_fast.cl"
    #include "../../kernels/level3/transpose_pad.cl"
    , // separated in multiple parts to prevent C1091 in MSVC 2013
    #include "../../kernels/level3/xgemm_direct_part1.cl"
    #include "../../kernels/level3/xgemm_direct_part2.cl"
    #include "../../kernels/level3/xgemm_direct_part3.cl"
    , // separated in multiple parts to prevent C1091 in MSVC 2013
    #include "../../kernels/level3/xgemm_part1.cl"
    #include "../../kernels/level3/xgemm_part2.cl"
    , // separated in multiple parts to prevent C1091 in MSVC 2013
    #include "../../kernels/level3/xgemm_part3.cl"
    #include "../../kernels/level3/xgemm_part4.cl"
    , // separated in multiple parts to prevent C1091 in MSVC 2013
    #include "../../kernels/level3/xgemm_batched.cl"
    #include "../../kernels/level3/xgemm_direct_batched.cl"
    }) {
}

template <typename T>
void XgemmBatched<T>::DoGemmBatched(
    const Layout layout, const Transpose transA, const Transpose transB,
    const size_t m, const size_t n, const size_t k,
    const std::vector<T>& alpha,
    const Buffer<T>& a_buffer, const std::vector<size_t>& a_offsets, const size_t lda,
    const Buffer<T>& b_buffer, const std::vector<size_t>& b_offsets, const size_t ldb,
    const std::vector<T>& beta,
    Buffer<T>& c_buffer, const std::vector<size_t>& c_offsets, const size_t ldc,
    const size_t batch_count)
{
    // Tests for a valid batch count
    if ((batch_count < 1) ||
        (alpha.size() != batch_count) ||
        (beta.size() != batch_count) ||
        (a_offsets.size() != batch_count) ||
        (b_offsets.size() != batch_count) ||
        (c_offsets.size() != batch_count)) {
        throw BLASError(StatusCode::kInvalidBatchCount);
    }

    // Two methods to choose from, select which one to run
    const auto do_gemm_direct = Xgemm<T>::UseDirectKernel(m, n, k, db_["XGEMM_MIN_INDIRECT_SIZE"]);
    const auto gemm_kernel_id = do_gemm_direct ? 0 : db_["GEMMK"];

    // Computes the transpose/conjugate options and sets the a/b/c sizes based on that
    bool a_trans, b_trans, c_trans, a_conj, b_conj;
    size_t a_one, a_two, b_one, b_two, c_one, c_two;
    Xgemm<T>::ProcessArguments(
        layout, transA, transB, m, n, k,
        a_one, a_two, b_one, b_two, c_one, c_two,
        a_trans, b_trans, c_trans, a_conj, b_conj,
        gemm_kernel_id);

    // Tests the matrices for validty
#ifndef NDEBUG
    for (size_t batch = 0; batch < batch_count; ++batch) {
        TestMatrixA(a_one, a_two, a_buffer, a_offsets[batch], lda, false); // don't test for invalid LD
        TestMatrixB(b_one, b_two, b_buffer, b_offsets[batch], ldb, false); // don't test for invalid LD
        TestMatrixC(c_one, c_two, c_buffer, c_offsets[batch], ldc);
    }
#endif

    // Upload the scalar arguments to the device
    auto alpha_dev = context_.getSharedBuffer<T>(alpha.data(), batch_count, queue_);
    auto beta_dev = context_.getSharedBuffer<T>(beta.data(), batch_count, queue_);

    // Converts the offset to integers
    auto a_offsets_int = std::vector<int>(a_offsets.begin(), a_offsets.end());
    auto b_offsets_int = std::vector<int>(b_offsets.begin(), b_offsets.end());
    auto c_offsets_int = std::vector<int>(c_offsets.begin(), c_offsets.end());

    // Selects which version of the batched GEMM to run
    if (do_gemm_direct) { // single generic kernel
        BatchedGemmDirect(
            m, n, k, alpha_dev,
            a_buffer, a_offsets_int, lda, b_buffer, b_offsets_int, ldb,
            beta_dev, c_buffer, c_offsets_int, ldc,
            a_trans, b_trans, c_trans, a_conj, b_conj,
            batch_count);
    } else { // pre/post-processing plus a very fast kernel
        BatchedGemmIndirect(
            m, n, k, alpha_dev,
            a_buffer, a_offsets_int, lda, b_buffer, b_offsets_int, ldb,
            beta_dev, c_buffer, c_offsets_int, ldc,
            a_trans, b_trans, c_trans, a_conj, b_conj,
            a_one, a_two, b_one, b_two, c_one, c_two, batch_count);
    }
}

// The indirect version of batched GEMM. This uses the faster but non-general kernel.
// It has specific requirements, but several pre and post-processing kernels take
// care of those. However, the overhead of these extra kernels might not be ideal
// for certain devices/arguments.
template <typename T>
void XgemmBatched<T>::BatchedGemmIndirect(
    const size_t m, const size_t n, const size_t k,
    const Buffer<T>& alpha_dev,
    const Buffer<T>& a_buffer, const std::vector<int>& a_offsets, const size_t lda,
    const Buffer<T>& b_buffer, const std::vector<int>& b_offsets, const size_t ldb,
    const Buffer<T>& beta_dev,
    Buffer<T>& c_buffer, const std::vector<int>& c_offsets, const size_t ldc,
    const bool a_trans, const bool b_trans, const bool c_trans,
    const bool a_conj, const bool b_conj,
    const size_t a_one, const size_t a_two,
    const size_t b_one, const size_t b_two,
    const size_t c_one, const size_t c_two,
    const size_t batch_count)
{
    // Calculates the ceiled versions of m, n, and k
    const auto m_ceiled = Ceil(Ceil(m, db_["MWG"]), db_["VWM"]);
    const auto n_ceiled = Ceil(Ceil(n, db_["NWG"]), db_["VWN"]);
    const auto k_ceiled = Ceil(Ceil(k, db_["KWG"]), db_["VWM"]);

    // Computes the first and second "internal" (ceiled) dimensions of the 3 matrices
    // taking into account whether the matrices need to be rotated or not for the kernel.
    size_t a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i;
    Xgemm<T>::CalculateInternalDimensions(
        m, n, k, db_["MWG"], db_["NWG"], db_["KWG"],
        a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i,
        db_["GEMMK"]);

    // Sets the "internal" offsets, i.e. the perfect offsets
    auto a_offsets_i = std::vector<int>(batch_count);
    auto b_offsets_i = std::vector<int>(batch_count);
    auto c_offsets_i = std::vector<int>(batch_count);
    for (size_t batch = 0; batch < batch_count; ++batch) {
        a_offsets_i[batch] = static_cast<int>(batch * a_one_i * a_two_i);
        b_offsets_i[batch] = static_cast<int>(batch * b_one_i * b_two_i);
        c_offsets_i[batch] = static_cast<int>(batch * c_one_i * c_two_i);
    }

    // Determines whether or not temporary matrices are needed
    auto a_no_temp = a_one == a_one_i && a_two == a_two_i &&
                     lda == a_one && a_offsets == a_offsets_i &&
                     !a_trans && !a_conj;
    auto b_no_temp = b_one == b_one_i && b_two == b_two_i &&
                     ldb == b_one && b_offsets == b_offsets_i &&
                     !b_trans && !b_conj;
    auto c_no_temp = c_one == c_one_i && c_two == c_two_i &&
                     ldc == c_one && c_offsets == c_offsets_i &&
                     !c_trans;

    // Creates the temporary matrices
    TemporaryBuffer<T> a_temp, b_temp, c_temp;

    // Runs the pre-processing kernel for matrix A. This transposes the matrix, but also
    // pads zeros to fill it up until it reaches a certain multiple of size (kernel
    // parameter dependent). In case nothing has to be done, there kernels can be skipped.
    if (!a_no_temp) {
        a_temp = context_.getTemporaryBuffer<T>(batch_count * a_one_i * a_two_i);
        for (size_t batch = 0; batch < batch_count; ++batch)
            a_offsets_i[batch] += a_temp.offset();

        auto a_offsets_dev = context_.getSharedBuffer<int>(a_offsets.data(), batch_count, queue_);
        auto a_offsets_i_dev = context_.getSharedBuffer<int>(b_offsets_i.data(), batch_count, queue_);
        PadCopyTransposeMatrixBatched(
            queue_, device_, db_, nullptr,
            a_one, a_two, lda, a_offsets_dev, a_buffer,
            a_one_i, a_two_i, a_one_i, a_offsets_i_dev, a_temp,
            program_, true, a_trans, a_conj, batch_count);
    }

    // As above, but now for matrix B
    if (!b_no_temp) {
        b_temp = context_.getTemporaryBuffer<T>(batch_count * b_one_i * b_two_i);
        for (size_t batch = 0; batch < batch_count; ++batch)
            b_offsets_i[batch] += b_temp.offset();

        auto b_offsets_dev = context_.getSharedBuffer<int>(b_offsets.data(), batch_count, queue_);
        auto b_offsets_i_dev = context_.getSharedBuffer<int>(b_offsets_i.data(), batch_count, queue_);
        PadCopyTransposeMatrixBatched(
            queue_, device_, db_, nullptr,
            b_one, b_two, ldb, b_offsets_dev, b_buffer,
            b_one_i, b_two_i, b_one_i, b_offsets_i_dev, b_temp,
            program_, true, b_trans, b_conj, batch_count);
    }

    // As above, but now for matrix C
    Buffer<int> c_offsets_dev, c_offsets_i_dev;
    if (!c_no_temp) {
        c_temp = context_.getTemporaryBuffer<T>(batch_count * c_one_i * c_two_i);
        for (size_t batch = 0; batch < batch_count; ++batch)
            c_offsets_i[batch] += c_temp.offset();

        c_offsets_dev = context_.getSharedBuffer<int>(c_offsets.data(), batch_count, queue_);
        c_offsets_i_dev = context_.getSharedBuffer<int>(c_offsets_i.data(), batch_count, queue_);
        PadCopyTransposeMatrixBatched(
            queue_, device_, db_, nullptr,
            c_one, c_two, ldc, c_offsets_dev, c_buffer,
            c_one_i, c_two_i, c_one_i, c_offsets_i_dev, c_temp,
            program_, true, c_trans, false, batch_count);
    }

    // Retrieves the Xgemm kernel from the compiled binary
    auto kernel = program_.getKernel("XgemmBatched");

    // Sets the kernel arguments
    kernel.setArguments(
        static_cast<int>(m_ceiled),
        static_cast<int>(n_ceiled),
        static_cast<int>(k_ceiled),
        alpha_dev, beta_dev,
        a_no_temp ? a_buffer : a_temp,
        static_cast<int>(a_no_temp ? 0 : a_temp.offset()),
        static_cast<int>(a_one_i), static_cast<int>(a_two_i),
        b_no_temp ? b_buffer : b_temp,
        static_cast<int>(b_no_temp ? 0 : b_temp.offset()),
        static_cast<int>(b_one_i), static_cast<int>(b_two_i),
        c_no_temp ? c_buffer : c_temp,
        static_cast<int>(c_no_temp ? 0 : c_temp.offset()),
        static_cast<int>(c_one_i), static_cast<int>(c_two_i));

    // Computes the global and local thread sizes
    const auto global = std::vector<size_t>{
        (c_one_i * db_["MDIMC"]) / db_["MWG"],
        (c_two_i * db_["NDIMC"]) / db_["NWG"],
        batch_count
    };
    const auto local = std::vector<size_t>{db_["MDIMC"], db_["NDIMC"], 1};

    // Launches the kernel
    auto eventPointer = !c_no_temp ? nullptr : event_;
    RunKernel(kernel, queue_, device_, global, local, eventPointer);

    // Runs the post-processing kernel if needed
    if (!c_no_temp) {
        PadCopyTransposeMatrixBatched(
            queue_, device_, db_, event_,
            c_one_i, c_two_i, c_one_i, c_offsets_i_dev, c_temp,
            c_one, c_two, ldc, c_offsets_dev, c_buffer,
            program_, false, c_trans, false, batch_count);
    }
}

// The direct version of batched GEMM, requiring just one kernel, no pre or
// post-processing kernels.
template <typename T>
void XgemmBatched<T>::BatchedGemmDirect(
    const size_t m, const size_t n, const size_t k,
    const Buffer<T>& alpha_dev,
    const Buffer<T>& a_buffer, const std::vector<int>& a_offsets, const size_t lda,
    const Buffer<T>& b_buffer, const std::vector<int>& b_offsets, const size_t ldb,
    const Buffer<T>& beta_dev,
    Buffer<T>& c_buffer, const std::vector<int>& c_offsets, const size_t ldc,
    const bool a_trans, const bool b_trans, const bool c_trans,
    const bool a_conj, const bool b_conj,
    const size_t batch_count)
{
    // Uploads the offsets to the device
    auto a_offsets_dev = context_.getSharedBuffer<int>(a_offsets.data(), batch_count, queue_);
    auto b_offsets_dev = context_.getSharedBuffer<int>(b_offsets.data(), batch_count, queue_);
    auto c_offsets_dev = context_.getSharedBuffer<int>(c_offsets.data(), batch_count, queue_);

    // Retrieves the proper XgemmDirect kernel from the compiled binary
    const auto name =
        a_trans ? (b_trans ? "XgemmDirectBatchedTT" : "XgemmDirectBatchedTN")
                : (b_trans ? "XgemmDirectBatchedNT" : "XgemmDirectBatchedNN");
    auto kernel = program_.getKernel(name);

    // Sets the kernel arguments
    kernel.setArguments(
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        alpha_dev, beta_dev,
        a_buffer, a_offsets_dev, static_cast<int>(lda),
        b_buffer, b_offsets_dev, static_cast<int>(ldb),
        c_buffer, c_offsets_dev, static_cast<int>(ldc),
        static_cast<int>(c_trans),
        static_cast<int>(a_conj),
        static_cast<int>(b_conj));

    // Computes the global and local thread sizes
    const auto m_ceiled = Ceil(m, db_["WGD"]);
    const auto n_ceiled = Ceil(n, db_["WGD"]);
    const auto global = std::vector<size_t>{
        (m_ceiled * db_["MDIMCD"]) / db_["WGD"],
        (n_ceiled * db_["NDIMCD"]) / db_["WGD"],
        batch_count
    };
    const auto local = std::vector<size_t>{db_["MDIMCD"], db_["NDIMCD"], 1};

    // Launches the kernel
    RunKernel(kernel, queue_, device_, global, local, event_);
}

// =================================================================================================

// Compiles the templated class
template class XgemmBatched<int16_t>;
template class XgemmBatched<int32_t>;
template class XgemmBatched<int64_t>;
template class XgemmBatched<half>;
template class XgemmBatched<float>;
template class XgemmBatched<double>;
template class XgemmBatched<float2>;
template class XgemmBatched<double2>;

}} // namespace gpgpu::blas
