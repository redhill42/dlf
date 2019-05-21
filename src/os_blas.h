#ifndef KNERON_OS_BLAS_H
#define KNERON_OS_BLAS_H

#if HAS_MKL
#include <mkl.h>

#elif defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#define cblas_saxpby catlas_saxpby
#define cblas_daxpby catlas_daxpby
#define cblas_caxpby catlas_caxpby
#define cblas_zaxpby catlas_zaxpby

#else
#include <cblas.h>
#endif

#endif //KNERON_OS_BLAS_H
