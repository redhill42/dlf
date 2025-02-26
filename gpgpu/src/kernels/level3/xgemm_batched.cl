
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the batched version of the non-direct GEMM kernel. See part 1 for information
// about the non-batched version of the kernel.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
#if defined(ROUTINE_GEMMBATCHED)

__kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
void XgemmBatched(const int kSizeM, const int kSizeN, const int kSizeK,
                  const __constant real_arg* arg_alphas,
                  const __constant real_arg* arg_betas,
                  const __global realM* restrict agm, const int a_offset, const int a_one, const int a_two,
                  const __global realN* restrict bgm, const int b_offset, const int b_one, const int b_two,
                  __global realM* cgm, const int c_offset, const int c_one, const int c_two) {
  const int batch = get_group_id(2);
  const real alpha = GetRealArg(arg_alphas[batch]);
  const real beta = GetRealArg(arg_betas[batch]);

  // Sets the offsets
  const int a_batch_offset = a_offset + batch * a_one * a_two;
  const int b_batch_offset = b_offset + batch * b_one * b_two;
  const int c_batch_offset = c_offset + batch * c_one * c_two;
  agm = (const __global realM*)((const __global real*)agm + a_batch_offset);
  bgm = (const __global realN*)((const __global real*)bgm + b_batch_offset);
  cgm = (__global realM*)((__global real*)cgm + c_batch_offset);

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realN blm[KWG * NWG/VWN];
  #endif

  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm, blm);
  #elif SA == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm);
  #elif SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, blm);
  #else
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta);
  #endif
}

#endif
// =================================================================================================
#if defined(ROUTINE_GEMMSTRIDEDBATCHED)

__kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
void XgemmStridedBatched(const int kSizeM, const int kSizeN, const int kSizeK,
                         const real_arg arg_alpha, const real_arg arg_beta,
                         const __global realM* restrict agm, const int a_offset, const int a_one, const int a_two,
                         const __global realN* restrict bgm, const int b_offset, const int b_one, const int b_two,
                         __global realM* cgm, const int c_offset, const int c_one, const int c_two) {
  const int batch = get_group_id(2);
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Sets the offsets
  const int a_batch_offset = a_offset + batch * a_one * a_two;
  const int b_batch_offset = b_offset + batch * b_one * b_two;
  const int c_batch_offset = c_offset + batch * c_one * c_two;
  agm = (const __global realM*)((const __global real*)agm + a_batch_offset);
  bgm = (const __global realN*)((const __global real*)bgm + b_batch_offset);
  cgm = (__global realM*)((const __global real*)cgm + c_batch_offset);

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realN blm[KWG * NWG/VWN];
  #endif

  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm, blm);
  #elif SA == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm);
  #elif SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, blm);
  #else
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta);
  #endif
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
