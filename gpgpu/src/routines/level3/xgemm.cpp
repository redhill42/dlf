
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgemm class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level3/xgemm.hpp"

#include <string>
#include <vector>

namespace gpgpu { namespace blas {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xgemm<T>::Xgemm(const Queue &queue, Event* event, const std::string &name):
    Routine(queue, event, name,
            {"Copy","Pad","Transpose","Padtranspose","Xgemm","XgemmDirect","GemmRoutine"},
            PrecisionValue<T>(), {}, {
    #include "../../kernels/level3/level3.cl"
    #include "../../kernels/level3/copy_fast.cl"
    #include "../../kernels/level3/copy_pad.cl"
    #include "../../kernels/level3/transpose_fast.cl"
    #include "../../kernels/level3/transpose_pad.cl"
    #include "../../kernels/level3/convert_symmetric.cl"
    #include "../../kernels/level3/convert_triangular.cl"
    #include "../../kernels/level3/convert_hermitian.cl"
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
    }) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xgemm<T>::DoGemm(const Layout layout,
                      const Transpose a_transpose, const Transpose b_transpose,
                      const size_t m, const size_t n, const size_t k,
                      const T alpha,
                      const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                      const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
                      const T beta,
                      Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld) {

  // Two methods to choose from, select which one to run
  const auto do_gemm_direct = UseDirectKernel(m, n, k, db_["XGEMM_MIN_INDIRECT_SIZE"]);
  const auto gemm_kernel_id = (do_gemm_direct) ? 0 : db_["GEMMK"];

  // Computes the transpose/conjugate options and sets the a/b/c sizes based on that
  bool a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate;
  size_t a_one, a_two, b_one, b_two, c_one, c_two;
  ProcessArguments(layout, a_transpose, b_transpose, m, n, k,
                   a_one, a_two, b_one, b_two, c_one, c_two,
                   a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate,
                   gemm_kernel_id);

  // Tests three matrices (A, B, C) for validity, first from a perspective of the OpenCL buffers and
  // their sizes, and then from a perspective of parameter values (e.g. m, n, k). Tests whether the
  // OpenCL buffers are valid and non-zero and whether the OpenCL buffers have sufficient storage
  // space. Also tests that the leading dimensions of:
  //    matrix A cannot be less than K when rotated, or less than M when not-rotated
  //    matrix B cannot be less than N when rotated, or less than K when not-rotated
  //    matrix C cannot be less than N when rotated, or less than M when not-rotated
  TestMatrixA(a_one, a_two, a_buffer, a_offset, a_ld);
  TestMatrixB(b_one, b_two, b_buffer, b_offset, b_ld);
  TestMatrixC(c_one, c_two, c_buffer, c_offset, c_ld);

  // Selects which version of GEMM to run
  if (do_gemm_direct) { // for small sizes (single kernel)
    GemmDirect(m, n, k, alpha,
               a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, beta,
               c_buffer, c_offset, c_ld,
               a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate);
  }
  else { // for larger sizes (pre/post-processing plus a very fast kernel)
    GemmIndirect(m, n, k, alpha,
                 a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, beta,
                 c_buffer, c_offset, c_ld,
                 a_do_transpose, b_do_transpose, c_do_transpose, a_conjugate, b_conjugate,
                 a_one, a_two, b_one, b_two, c_one, c_two);
  }
}

// =================================================================================================

// The indirect version of GEMM. This uses the faster but non-general kernel. It has specific
// requirements, but several pre and post-processing kernels take care of those. However, the
// overhead of these extra kernels might not be ideal for certain devices/arguments.
template <typename T>
void Xgemm<T>::GemmIndirect(const size_t m, const size_t n, const size_t k,
                            const T alpha,
                            const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                            const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
                            const T beta,
                            Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld,
                            const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                            const bool a_conjugate, const bool b_conjugate,
                            const size_t a_one, const size_t a_two,
                            const size_t b_one, const size_t b_two,
                            const size_t c_one, const size_t c_two) {

  // Calculates the ceiled versions of m, n, and k
  const auto m_ceiled = Ceil(m, db_["MWG"]);
  const auto n_ceiled = Ceil(n, db_["NWG"]);
  const auto k_ceiled = Ceil(k, db_["KWG"] * db_["KREG"]);

  // Computes the first and second "internal" (ceiled) dimensions of the 3 matrices taking into account
  // whether the matrices need to be rotated or not for the kernel.
  size_t a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i;
  CalculateInternalDimensions(m, n, k, db_["MWG"], db_["NWG"], db_["KWG"] * db_["KREG"],
                              a_one_i, a_two_i, b_one_i, b_two_i, c_one_i, c_two_i,
                              db_["GEMMK"]);

  // Determines whether or not temporary matrices are needed
  auto a_no_temp = NoTempBuffer(a_one, a_one_i, a_two, a_two_i, a_ld, a_offset, a_do_transpose, a_conjugate);
  auto b_no_temp = NoTempBuffer(b_one, b_one_i, b_two, b_two_i, b_ld, b_offset, b_do_transpose, b_conjugate);
  auto c_no_temp = NoTempBuffer(c_one, c_one_i, c_two, c_two_i, c_ld, c_offset, c_do_transpose, false);

  // Computes the sizes and offsets for (optional) temporary buffers for the 3 matrices
  auto b_temp_offset = size_t{0};
  auto c_temp_offset = size_t{0};
  const auto temp_size = ComputeTempSize(
      a_no_temp, b_no_temp, c_no_temp,
      a_one_i*a_two_i, b_one_i*b_two_i, c_one_i*c_two_i,
      b_temp_offset, c_temp_offset);
  if (!IsMultiple(b_temp_offset, db_["VWN"]))
    throw BLASError(StatusCode::kUnexpectedError);
  if (!IsMultiple(c_temp_offset, db_["VWM"]))
    throw BLASError(StatusCode::kUnexpectedError);

  // Creates the buffer for the (optional) temporary matrices. Note that we use 'a_buffer' in case
  // when no temporary buffer is needed, but that's just to make it compile: it is never used.
  TemporaryBuffer<T> temp_buffer;
  if (temp_size > 0) {
    temp_buffer = context_.getTemporaryBuffer<T>(temp_size);
  }

  // Runs the pre-processing kernel for matrix A. This transposes the matrix, but also pads zeros
  // to fill it up until it reaches a certain multiple of size (kernel parameter dependent). In
  // case nothing has to be done, these kernels can be skipped.
  if (!a_no_temp) {
    PadCopyTransposeMatrix(queue_, device_, db_, nullptr,
                           a_one, a_two, a_ld, a_offset, a_buffer,
                           a_one_i, a_two_i, a_one_i,
                           temp_buffer.offset(), temp_buffer,
                           ConstantOne<T>(), program_,
                           true, a_do_transpose, a_conjugate);
  }

  // As above, but now for matrix B
  if (!b_no_temp) {
    PadCopyTransposeMatrix(queue_, device_, db_, nullptr,
                           b_one, b_two, b_ld, b_offset, b_buffer,
                           b_one_i, b_two_i, b_one_i,
                           temp_buffer.offset() + b_temp_offset, temp_buffer,
                           ConstantOne<T>(), program_,
                           true, b_do_transpose, b_conjugate);
  }

  // As above, but now for matrix C. This is only necessary if C is used both as input and output.
  if (!c_no_temp && beta != static_cast<T>(0)) {
    PadCopyTransposeMatrix(queue_, device_, db_, nullptr,
                           c_one, c_two, c_ld, c_offset, c_buffer,
                           c_one_i, c_two_i, c_one_i,
                           temp_buffer.offset() + c_temp_offset, temp_buffer,
                           ConstantOne<T>(), program_,
                           true, c_do_transpose, false);
  }

  // Retrieves the Xgemm kernel from the compiled binary
  auto kernel = program_.getKernel("Xgemm");

  // Sets the kernel arguments
  kernel.setArguments(
    static_cast<int>(m_ceiled), static_cast<int>(n_ceiled), static_cast<int>(k_ceiled),
    GetRealArg(alpha), GetRealArg(beta),
    a_no_temp ? a_buffer : temp_buffer,
    static_cast<int>(a_no_temp ? a_offset : temp_buffer.offset()),
    b_no_temp ? b_buffer : temp_buffer,
    static_cast<int>(b_no_temp ? b_offset : temp_buffer.offset() + b_temp_offset),
    c_no_temp ? c_buffer : temp_buffer,
    static_cast<int>(c_no_temp ? c_offset : temp_buffer.offset() + c_temp_offset));

  // Computes the global and local thread sizes
  const auto global_divider_one = c_want_rotated_(db_["GEMMK"]) ? db_["NWG"] : db_["MWG"];
  const auto global_divider_two = c_want_rotated_(db_["GEMMK"]) ? db_["MWG"] : db_["NWG"];
  const auto global = std::vector<size_t>{
    (c_one_i * db_["MDIMC"]) / global_divider_one,
    (c_two_i * db_["NDIMC"]) / global_divider_two
  };
  const auto local = std::vector<size_t>{db_["MDIMC"], db_["NDIMC"]};

  // Launches the kernel
  auto eventPointer = !c_no_temp ? nullptr : event_;
  RunKernel(kernel, queue_, device_, global, local, eventPointer);

  // Runs the post-processing kernel if needed
  if (!c_no_temp) {
    PadCopyTransposeMatrix(queue_, device_, db_, event_,
                           c_one_i, c_two_i, c_one_i,
                           temp_buffer.offset() + c_temp_offset, temp_buffer,
                           c_one, c_two, c_ld, c_offset, c_buffer,
                           ConstantOne<T>(), program_,
                           false, c_do_transpose, false);
  }
}


// =================================================================================================

// The direct version of GEMM, requiring just one kernel, no pre or post-processing kernels.
template <typename T>
void Xgemm<T>::GemmDirect(const size_t m, const size_t n, const size_t k,
                          const T alpha,
                          const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                          const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
                          const T beta,
                          Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld,
                          const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                          const bool a_conjugate, const bool b_conjugate) {

  // Retrieves the proper XgemmDirect kernel from the compiled binary
  const auto name = (a_do_transpose) ? (b_do_transpose ? "XgemmDirectTT" : "XgemmDirectTN") :
                                       (b_do_transpose ? "XgemmDirectNT" : "XgemmDirectNN");
  auto kernel = program_.getKernel(name);

  // Sets the kernel arguments
  kernel.setArgument(0, static_cast<int>(m));
  kernel.setArgument(1, static_cast<int>(n));
  kernel.setArgument(2, static_cast<int>(k));
  kernel.setArgument(3, GetRealArg(alpha));
  kernel.setArgument(4, GetRealArg(beta));
  kernel.setArgument(5, a_buffer);
  kernel.setArgument(6, static_cast<int>(a_offset));
  kernel.setArgument(7, static_cast<int>(a_ld));
  kernel.setArgument(8, b_buffer);
  kernel.setArgument(9, static_cast<int>(b_offset));
  kernel.setArgument(10, static_cast<int>(b_ld));
  kernel.setArgument(11, c_buffer);
  kernel.setArgument(12, static_cast<int>(c_offset));
  kernel.setArgument(13, static_cast<int>(c_ld));
  kernel.setArgument(14, static_cast<int>(c_do_transpose));
  kernel.setArgument(15, static_cast<int>(a_conjugate));
  kernel.setArgument(16, static_cast<int>(b_conjugate));

  // Computes the global and local thread sizes
  const auto m_ceiled = Ceil(m, db_["WGD"]);
  const auto n_ceiled = Ceil(n, db_["WGD"]);
  const auto global = std::vector<size_t>{
  //  CeilDiv(m * db_["MDIMCD"], db_["WGD"]),
  //  CeilDiv(n * db_["NDIMCD"], db_["WGD"])
      (m_ceiled * db_["MDIMCD"]) / db_["WGD"],
      (n_ceiled * db_["NDIMCD"]) / db_["WGD"]
  };
  const auto local = std::vector<size_t>{db_["MDIMCD"], db_["NDIMCD"]};

  // Launches the kernel
  RunKernel(kernel, queue_, device_, global, local, event_);
}

// =================================================================================================

// Compiles the templated class
template class Xgemm<half>;
template class Xgemm<float>;
template class Xgemm<double>;
template class Xgemm<float2>;
template class Xgemm<double2>;
template class Xgemm<int32_t>;
template class Xgemm<int64_t>;

// =================================================================================================
}} // namespace gpgpu::blas
