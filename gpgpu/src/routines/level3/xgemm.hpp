
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgemm routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef GPGPU_BLAS_ROUTINES_XGEMM_H_
#define GPGPU_BLAS_ROUTINES_XGEMM_H_

#include "routine.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xgemm: public Routine {
public:
  // Constructor
  Xgemm(const Queue &queue, Event* event, const std::string &name = "GEMM");

  // Templated-precision implementation of the routine
  void DoGemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
              const size_t m, const size_t n, const size_t k,
              const T alpha,
              const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
              const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
              const T beta,
              Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld);

  // Defines the assumptions of the GEMM kernels
  static bool a_want_rotated_(const size_t gemm_kernel_id) { return gemm_kernel_id == 1; }
  static bool b_want_rotated_(const size_t) { return true; }
  static bool c_want_rotated_(const size_t gemm_kernel_id) { return gemm_kernel_id == 1; }

  // Selects which version of GEMM to run
  static bool UseDirectKernel(const size_t m, const size_t n, const size_t k,
                              const size_t min_indirect_size) {
    const auto m_n_k = static_cast<unsigned long long>(m) * static_cast<unsigned long long>(n) *
                       static_cast<unsigned long long>(k);
    const auto min_indirect_size_ll = static_cast<unsigned long long>(min_indirect_size);
    const auto min_indirect_size_e3 = min_indirect_size_ll * min_indirect_size_ll * min_indirect_size_ll;
    return (m_n_k < min_indirect_size_e3);
  }

  // Process the user-arguments, computes secondary parameters
  static void ProcessArguments(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                               const size_t m, const size_t n, const size_t k,
                               size_t& a_one, size_t& a_two, size_t& b_one,
                               size_t& b_two, size_t& c_one, size_t& c_two,
                               bool& a_do_transpose, bool& b_do_transpose, bool& c_do_transpose,
                               bool& a_conjugate, bool& b_conjugate,
                               const size_t gemm_kernel_id) {

    // Makes sure all dimensions are larger than zero
    if ((m == 0) || (n == 0) || (k == 0)) { throw BLASError(StatusCode::kInvalidDimension); }

    // Computes whether or not the matrices are transposed in memory. This is based on their layout
    // (row or column-major) and whether or not they are requested to be pre-transposed. Note
    // that the Xgemm kernel expects either matrices A and C (in case of row-major) or B (in case of
    // col-major) to be transformed, so transposing requirements are not the same as whether or not
    // the matrix is actually transposed in memory.
    const auto a_rotated = (layout == Layout::ColMajor && a_transpose != Transpose::NoTrans) ||
                           (layout == Layout::RowMajor && a_transpose == Transpose::NoTrans);
    const auto b_rotated = (layout == Layout::ColMajor && b_transpose != Transpose::NoTrans) ||
                           (layout == Layout::RowMajor && b_transpose == Transpose::NoTrans);
    const auto c_rotated = (layout == Layout::RowMajor);
    a_do_transpose = a_rotated != a_want_rotated_(gemm_kernel_id);
    b_do_transpose = b_rotated != b_want_rotated_(gemm_kernel_id);
    c_do_transpose = c_rotated != c_want_rotated_(gemm_kernel_id);

    // In case of complex data-types, the transpose can also become a conjugate transpose
    a_conjugate = (a_transpose == Transpose::ConjTrans);
    b_conjugate = (b_transpose == Transpose::ConjTrans);

    // Computes the first and second dimensions of the 3 matrices taking into account whether the
    // matrices are rotated or not
    a_one = (a_rotated) ? k : m;
    a_two = (a_rotated) ? m : k;
    b_one = (b_rotated) ? n : k;
    b_two = (b_rotated) ? k : n;
    c_one = (c_rotated) ? n : m;
    c_two = (c_rotated) ? m : n;
  }

  // Computes the sizes and offsets for (optional) temporary buffers for the 3 matrices
  static size_t ComputeTempSize(const bool a_no_temp, const bool b_no_temp, const bool c_no_temp,
                                const size_t a_size, const size_t b_size, const size_t c_size,
                                size_t &b_temp_offset, size_t &c_temp_offset) {
    auto temp_size = size_t{0};
    if (!a_no_temp) { temp_size += a_size; }
    if (!b_no_temp) { b_temp_offset = temp_size; temp_size += b_size; }
    if (!c_no_temp) { c_temp_offset = temp_size; temp_size += c_size; }
    return temp_size;
  }

  // Determines whether or not temporary matrices are needed
  static bool NoTempBuffer(const size_t one, const size_t one_i, const size_t two, const size_t two_i,
                           const size_t ld, const size_t offset,
                           const bool do_transpose, const bool conjugate) {
    return one == one_i && two == two_i && ld == one && offset == 0 && !do_transpose && !conjugate;
  }


  // Computes the first and second "internal" (ceiled) dimensions of the 3 matrices taking into account
  // whether the matrices need to be rotated or not for the kernel.
  static void CalculateInternalDimensions(const size_t m, const size_t n, const size_t k,
                                          const size_t mwg, const size_t nwg, const size_t kwg,
                                          size_t& a_one_i, size_t& a_two_i, size_t& b_one_i,
                                          size_t& b_two_i, size_t& c_one_i, size_t& c_two_i,
                                          const size_t gemm_kernel_id) {
    const auto m_ceiled = Ceil(m, mwg);
    const auto n_ceiled = Ceil(n, nwg);
    const auto k_ceiled = Ceil(k, kwg);
    a_one_i = (a_want_rotated_(gemm_kernel_id)) ? k_ceiled : m_ceiled;
    a_two_i = (a_want_rotated_(gemm_kernel_id)) ? m_ceiled : k_ceiled;
    b_one_i = (b_want_rotated_(gemm_kernel_id)) ? n_ceiled : k_ceiled;
    b_two_i = (b_want_rotated_(gemm_kernel_id)) ? k_ceiled : n_ceiled;
    c_one_i = (c_want_rotated_(gemm_kernel_id)) ? n_ceiled : m_ceiled;
    c_two_i = (c_want_rotated_(gemm_kernel_id)) ? m_ceiled : n_ceiled;
  }

  // Indirect version of GEMM (with pre and post-processing kernels)
  void GemmIndirect(const size_t m, const size_t n, const size_t k,
                    const T alpha,
                    const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                    const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
                    const T beta,
                    Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld,
                    const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                    const bool a_conjugate, const bool b_conjugate,
                    const size_t a_one, const size_t a_two,
                    const size_t b_one, const size_t b_two,
                    const size_t c_one, const size_t c_two);

  // Direct version of GEMM (no pre and post-processing kernels)
  void GemmDirect(const size_t m, const size_t n, const size_t k,
                  const T alpha,
                  const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                  const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
                  const T beta,
                  Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld,
                  const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
                  const bool a_conjugate, const bool b_conjugate);
};

// =================================================================================================
}} // namespace gpgpu::blas

// GPGPU_BLAS_ROUTINES_XGEMM_H_
#endif
