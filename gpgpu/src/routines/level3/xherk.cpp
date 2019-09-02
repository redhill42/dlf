
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xherk class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level3/xherk.hpp"
#include "routines/level3/xgemm.hpp"

#include <string>
#include <vector>

namespace gpgpu { namespace blas {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T, typename U>
Xherk<T,U>::Xherk(const Queue &queue, Event* event, const std::string &name):
    Routine(queue, event, name, {"Copy","Pad","Transpose","Padtranspose","Xgemm"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level3/level3.cl"
    #include "../../kernels/level3/copy_fast.cl"
    #include "../../kernels/level3/copy_pad.cl"
    #include "../../kernels/level3/transpose_fast.cl"
    #include "../../kernels/level3/transpose_pad.cl"
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
template <typename T, typename U>
void Xherk<T,U>::DoHerk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
                              const size_t n, const size_t k,
                              const U alpha,
                              const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                              const U beta,
                              Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld) {
  const auto b_transpose = (a_transpose != Transpose::NoTrans) ? Transpose::NoTrans : Transpose::Trans;
  const auto b_buffer = a_buffer;
  const auto b_offset = a_offset;
  const auto b_ld = a_ld;
  const auto complex_alpha = T{alpha, static_cast<U>(0.0)};
  const auto complex_beta = T{beta, static_cast<U>(0.0)};
  HerkAB(layout, triangle, a_transpose, b_transpose, n, k, complex_alpha,
         a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, complex_beta, c_buffer, c_offset, c_ld,
         event_, true);
}

template <typename T, typename U>
void Xherk<T,U>::HerkAB(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t n, const size_t k,
                        const T complex_alpha,
                        const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                        const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
                        const T complex_beta,
                        Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld,
                        Event* final_event, const bool diagonal_to_zero) {

  // Computes the transpose/conjugate options and sets the a/b/c sizes based on that
  bool a_do_transpose, b_do_transpose, c_do_transpose, dummy1, dummy2;
  size_t a_one, a_two, b_one, b_two, c_one, c_two;
  Xgemm<T>::ProcessArguments(layout, a_transpose, b_transpose, n, n, k,
                             a_one, a_two, b_one, b_two, c_one, c_two,
                             a_do_transpose, b_do_transpose, c_do_transpose, dummy1, dummy2,
                             db_["GEMMK"]);

  // Determines whether to apply the conjugate transpose to matrix B (argument: no transpose) or
  // to matrix A (argument: conjugate transpose)
  auto a_conjugate = (a_transpose != Transpose::NoTrans);
  auto b_conjugate = (b_transpose != Transpose::NoTrans);

  // Tests the two matrices (A, C) for validity, first from a perspective of the OpenCL buffers and
  // their sizes, and then from a perspective of parameter values (e.g. n, k). Tests whether the
  // OpenCL buffers are valid and non-zero and whether the OpenCL buffers have sufficient storage
  // space. Also tests that the leading dimensions of:
  //    matrix A cannot be less than N when rotated, or less than K when not-rotated
  //    matrix C cannot be less than N
  TestMatrixA(a_one, a_two, a_buffer, a_offset, a_ld);
  TestMatrixB(b_one, b_two, b_buffer, b_offset, b_ld);
  TestMatrixC(n, n, c_buffer, c_offset, c_ld);

  // Calculates the ceiled versions of n and k
  auto n_ceiled = Ceil(Ceil(n, db_["MWG"]), db_["NWG"]);
  auto k_ceiled = Ceil(k, db_["KWG"] * db_["KREG"]);

  // Computes the first and second "internal" (ceiled) dimensions of the 3 matrices taking into account
  // whether the matrices need to be rotated or not for the kernel.
  const auto a_one_i = (Xgemm<T>::a_want_rotated_(db_["GEMMK"])) ? k_ceiled : n_ceiled;
  const auto a_two_i = (Xgemm<T>::a_want_rotated_(db_["GEMMK"])) ? n_ceiled : k_ceiled;
  const auto b_one_i = (!Xgemm<T>::b_want_rotated_(db_["GEMMK"])) ? k_ceiled : n_ceiled;
  const auto b_two_i = (!Xgemm<T>::b_want_rotated_(db_["GEMMK"])) ? n_ceiled : k_ceiled;

  // Decides which kernel to run: the upper-triangular or lower-triangular version
  auto kernel_name = (triangle == Triangle::Upper) ? "XgemmUpper" : "XgemmLower";

  // Determines whether or not temporary matrices are needed
  const auto a_no_temp = Xgemm<T>::NoTempBuffer(a_one, a_one_i, a_two, a_two_i, a_ld, a_offset, a_do_transpose, a_conjugate);
  const auto b_no_temp = Xgemm<T>::NoTempBuffer(b_one, b_one_i, b_two, b_two_i, b_ld, b_offset, b_do_transpose, b_conjugate);

  // Creates the temporary matrices
  TemporaryBuffer<T> a_temp, b_temp, c_temp;

  // Runs the pre-processing kernel for matrix A. This transposes the matrix, but also pads zeros
  // to fill it up until it reaches a certain multiple of size (kernel parameter dependent). In
  // case nothing has to be done, these kernels can be skipped. Two copies are created.
  if (!a_no_temp) {
    a_temp = context_.getTemporaryBuffer<T>(a_one_i * a_two_i);
    PadCopyTransposeMatrix(queue_, device_, db_, nullptr,
                           a_one, a_two, a_ld, a_offset, a_buffer,
                           a_one_i, a_two_i, a_one_i, a_temp.offset(), a_temp,
                           ConstantOne<T>(), program_,
                           true, a_do_transpose, a_conjugate);
  }
  
  if (!b_no_temp) {
    b_temp = context_.getTemporaryBuffer<T>(b_one_i * b_two_i);
    PadCopyTransposeMatrix(queue_, device_, db_, nullptr,
                           b_one, b_two, b_ld, b_offset, b_buffer,
                           b_one_i, b_two_i, b_one_i, b_temp.offset(), b_temp,
                           ConstantOne<T>(), program_,
                           true, b_do_transpose, b_conjugate);
  }

  // Furthermore, also creates a (possibly padded) copy of matrix C, since it is not allowed to
  // modify the other triangle.
  c_temp = context_.getTemporaryBuffer<T>(n_ceiled * n_ceiled);
  PadCopyTransposeMatrix(queue_, device_, db_, nullptr,
                         n, n, c_ld, c_offset, c_buffer,
                         n_ceiled, n_ceiled, n_ceiled, c_temp.offset(), c_temp,
                         ConstantOne<T>(), program_,
                         true, c_do_transpose, false);

  // Retrieves the XgemmUpper or XgemmLower kernel from the compiled binary
  auto kernel = program_.getKernel(kernel_name);

  // Sets the kernel arguments
  kernel.setArguments(static_cast<int>(n_ceiled),
                      static_cast<int>(k_ceiled),
                      GetRealArg(complex_alpha),
                      GetRealArg(complex_beta),
                      a_no_temp ? a_buffer : a_temp,
                      static_cast<int>(a_no_temp ? a_offset : a_temp.offset()),
                      b_no_temp ? b_buffer : b_temp,
                      static_cast<int>(b_no_temp ? b_offset : b_temp.offset()),
                      c_temp,
                      static_cast<int>(c_temp.offset()));

  // Computes the global and local thread sizes
  auto global = std::vector<size_t>{
    (n_ceiled * db_["MDIMC"]) / db_["MWG"],
    (n_ceiled * db_["NDIMC"]) / db_["NWG"]
  };
  auto local = std::vector<size_t>{db_["MDIMC"], db_["NDIMC"]};

  // Launches the kernel
  RunKernel(kernel, queue_, device_, global, local, nullptr);

  // Runs the post-processing kernel
  const auto upper = Xgemm<T>::c_want_rotated_(db_["GEMMK"]) ? (triangle == Triangle::Lower) :
                     (triangle == Triangle::Upper);
  const auto lower = !upper;
  PadCopyTransposeMatrix(queue_, device_, db_, final_event,
                         n_ceiled, n_ceiled, n_ceiled, c_temp.offset(), c_temp,
                         n, n, c_ld, c_offset, c_buffer,
                         ConstantOne<T>(), program_,
                         false, c_do_transpose, false, upper, lower, diagonal_to_zero);
}

// =================================================================================================

// Compiles the templated class
template class Xherk<float2,float>;
template class Xherk<double2,double>;

// =================================================================================================
}} // namespace gpgpu::blas
