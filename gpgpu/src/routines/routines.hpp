
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains all the includes of all the routines in CLBlast.
//
// =================================================================================================

#ifndef GPGPU_BLAS_ROUTINES_ROUTINES_H_
#define GPGPU_BLAS_ROUTINES_ROUTINES_H_

// BLAS level-1 includes
#include "routines/level1/xswap.hpp"
#include "routines/level1/xscal.hpp"
#include "routines/level1/xcopy.hpp"
#include "routines/level1/xaxpy.hpp"
#include "routines/level1/xdot.hpp"
#include "routines/level1/xnrm2.hpp"
#include "routines/level1/xasum.hpp"
#include "routines/level1/xsum.hpp" // non-BLAS routine
#include "routines/level1/xamax.hpp"
#include "routines/level1/xamin.hpp" // non-BLAS routine
#include "routines/level1/xmax.hpp" // non-BLAS routine
#include "routines/level1/xmin.hpp" // non-BLAS routine

// BLAS level-2 includes
#include "routines/level2/xgemv.hpp"
#include "routines/level2/xgbmv.hpp"
#include "routines/level2/xhemv.hpp"
#include "routines/level2/xhbmv.hpp"
#include "routines/level2/xhpmv.hpp"
#include "routines/level2/xsymv.hpp"
#include "routines/level2/xsbmv.hpp"
#include "routines/level2/xspmv.hpp"
#include "routines/level2/xtrmv.hpp"
#include "routines/level2/xtbmv.hpp"
#include "routines/level2/xtpmv.hpp"
#include "routines/level2/xtrsv.hpp"
#include "routines/level2/xger.hpp"
#include "routines/level2/xgerc.hpp"
#include "routines/level2/xher.hpp"
#include "routines/level2/xhpr.hpp"
#include "routines/level2/xher2.hpp"
#include "routines/level2/xhpr2.hpp"
#include "routines/level2/xsyr.hpp"
#include "routines/level2/xspr.hpp"
#include "routines/level2/xsyr2.hpp"
#include "routines/level2/xspr2.hpp"

// BLAS level-3 includes
#include "routines/level3/xgemm.hpp"
#include "routines/level3/xsymm.hpp"
#include "routines/level3/xhemm.hpp"
#include "routines/level3/xsyrk.hpp"
#include "routines/level3/xherk.hpp"
#include "routines/level3/xsyr2k.hpp"
#include "routines/level3/xher2k.hpp"
#include "routines/level3/xtrmm.hpp"
#include "routines/level3/xtrsm.hpp"

// Level-x includes (non-BLAS)
#include "routines/levelx/xhad.hpp"
#include "routines/levelx/xomatcopy.hpp"
#include "routines/levelx/xim2col.hpp"
#include "routines/levelx/xcol2im.hpp"
#include "routines/levelx/xconvgemm.hpp"
#include "routines/levelx/xaxpybatched.hpp"
#include "routines/levelx/xgemmbatched.hpp"
#include "routines/levelx/xgemmstridedbatched.hpp"
#include "routines/levelx/xgetrf.hpp"

// DNN includes
#include "routines/dnn/xcopy.hpp"
#include "routines/dnn/xfill.hpp"
#include "routines/dnn/xreverse.hpp"
#include "routines/dnn/xrange.hpp"
#include "routines/dnn/xrandom.hpp"
#include "routines/dnn/xtransform.hpp"
#include "routines/dnn/xtransform_b.hpp"
#include "routines/dnn/xtransform_p.hpp"
#include "routines/dnn/xreduce.hpp"
#include "routines/dnn/xargreduce.hpp"
#include "routines/dnn/xscan.hpp"
#include "routines/dnn/xmerge.hpp"
#include "routines/dnn/xsort.hpp"
#include "routines/dnn/xargsort.hpp"
#include "routines/dnn/xtopk.hpp"
#include "routines/dnn/xnorm.hpp"
#include "routines/dnn/xpool.hpp"
#include "routines/dnn/xsoftmax.hpp"
#include "routines/dnn/xhardmax.hpp"
#include "routines/dnn/xwhere.hpp"
#include "routines/dnn/xonehot.hpp"
#include "routines/dnn/xgather.hpp"
#include "routines/dnn/xresize.hpp"

#endif // GPGPU_BLAS_ROUTINES_ROUTINES_H_
