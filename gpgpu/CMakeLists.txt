
# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
# width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# ==================================================================================================

cmake_minimum_required(VERSION 2.8.11)

# Overrides for MSVC static runtime
option(OVERRIDE_MSVC_FLAGS_TO_MT "Override compiler flags for MSVC to build with a static runtime (/MT instead of /MD)" ON)
if(OVERRIDE_MSVC_FLAGS_TO_MT)
  set(CMAKE_USER_MAKE_RULES_OVERRIDE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/c_flag_overrides.cmake)
  set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cxx_flag_overrides.cmake)
endif()

# CMake project details
project("gpgpu" C CXX)
set(CMAKE_CXX_STANDARD 14)

# Options and their default values
option(BUILD_SHARED_LIBS "Build a shared (ON) or static library (OFF)" ON)

# Workarounds for bugs
option(AMD_SI_EMPTY_KERNEL_WORKAROUND "Enables workaround for bug in AMD Southern Island GPUs" OFF)
if(AMD_SI_EMPTY_KERNEL_WORKAROUND)
  add_definitions(-DAMD_SI_EMPTY_KERNEL_WORKAROUND)
endif()

# Compile in verbose mode with additional diagnostic messages
option(VERBOSE "Compile in verbose mode for additional diagnostic messages" OFF)
if(VERBOSE)
  message("-- Building in verbose mode")
  add_definitions(-DVERBOSE)
endif()

# ==================================================================================================

# RPATH settings
set(CMAKE_MACOSX_RPATH 1)

# ==================================================================================================

# Compiler-version check (requires at least CMake 2.8.10)
if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
    message(FATAL_ERROR "GCC version must be at least 4.7")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.3)
    message(FATAL_ERROR "Clang version must be at least 3.3")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "AppleClang version must be at least 5.0")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.0)
    message(FATAL_ERROR "ICC version must be at least 14.0")
  endif()
elseif(MSVC)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
    message(FATAL_ERROR "MS Visual Studio version must be at least 18.0")
  endif()
endif()

# DLL Settings
if(MSVC)
  if(BUILD_SHARED_LIBS)
    add_definitions(" /DGPGPU_DLL")
  endif()
endif(MSVC)

# C++ compiler settings
if(MSVC)
  set(FLAGS "/Ot")
  set(FLAGS "${FLAGS} /wd4715 /D_CRT_SECURE_NO_WARNINGS")
else()
  set(FLAGS "-std=c++11")
  if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    set(FLAGS "${FLAGS} -Wall -Wno-comment -Wno-return-type -Wno-switch -Wno-missing-noreturn")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9.0)
      set(FLAGS "${FLAGS} -Wno-attributes -Wno-unused-variable")
    endif()
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0.0)
      # GCC does not support attributes on template arguments
      # in particular we hit this with the alignment attributes on cl_XXX types
      # which are then used to instantiate various templates in GPGPU
      set(FLAGS "${FLAGS} -Wno-ignored-attributes")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
    set(FLAGS "${FLAGS} -Wextra -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded")
    set(FLAGS "${FLAGS} -Wno-missing-prototypes -Wno-float-equal -Wno-switch-enum -Wno-switch")
    set(FLAGS "${FLAGS} -Wno-exit-time-destructors -Wno-global-constructors -Wno-missing-noreturn")
    set(FLAGS "${FLAGS} -Wno-deprecated-declarations")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 3.9.0)  # clang 4.0 or higher
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0)  # but not for AppleClang
          set(FLAGS "${FLAGS} -Wno-undefined-var-template")
        endif()
    endif()
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")

# C compiler settings (for the sample)
if(MSVC)
  set(CFLAGS "/Ot")
else()
  set(CFLAGS "-O2 -std=c99")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CFLAGS}")

# ==================================================================================================

# Package scripts location
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gpgpu_SOURCE_DIR}/cmake/Modules/")

# OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

# CUDA
find_package(CUDA)
if(CUDA_FOUND)
  set(HAS_CUDA YES)
  add_definitions(-DHAS_CUDA=1)
  find_cuda_helper_libs("cuda")
  find_cuda_helper_libs("nvrtc")
  find_cuda_helper_libs("cublas")
  find_cuda_helper_libs("cudnn")
  set(CUDA_LIBRARIES ${CUDA_cuda_LIBRARY} ${CUDA_nvrtc_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_cudnn_LIBRARY})
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

if(HAS_CUDA)
  set(API_LIBRARIES ${OpenCL_LIBRARY} ${CUDA_LIBRARIES})
  set(API_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
else()
  set(API_LIBRARIES ${OpenCL_LIBRARY})
  set(API_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
endif()

# ==================================================================================================

# Sets the supported routines and the used kernels. New routines and kernels should be added here.
set(KERNELS copy_fast copy_pad transpose_fast transpose_pad xaxpy xdot xger
            xgemm xgemm_direct xgemv invert)
set(DATABASES copy pad padtranspose transpose xaxpy xdot
              xgemm xgemm_direct xgemv xgemv_fast xgemv_fast_rot xger invert
              gemm_routine trsv_routine)
set(ROUTINE_TUNERS xgemm xtrsv)
set(LEVEL1_ROUTINES xswap xscal xcopy xaxpy xdot xdotu xdotc xnrm2 xasum xamax)
set(LEVEL2_ROUTINES xgemv xgbmv xhemv xhbmv xhpmv xsymv xsbmv xspmv xtrmv xtbmv xtpmv xtrsv
                    xger xgeru xgerc xher xhpr xher2 xhpr2 xsyr xspr xsyr2 xspr2)
set(LEVEL3_ROUTINES xgemm xsymm xhemm xsyrk xherk xsyr2k xher2k xtrmm xtrsm)
set(LEVELX_ROUTINES xhad xomatcopy xim2col xcol2im xconvgemm xaxpybatched xgemmbatched xgemmstridedbatched)
set(DNN_ROUTINES xcopy xfill xtransform xtransform_p xtransform_b xreduce xscan
                 xargreduce xnorm xpool xsoftmax xwhere xgather)
set(ROUTINES ${LEVEL1_ROUTINES} ${LEVEL2_ROUTINES} ${LEVEL3_ROUTINES} ${LEVELX_ROUTINES})
set(PRECISIONS 32 64 3232 6464 16)

# ==================================================================================================

# Gathers all source-files (required for the compiler) and header-files (for IDEs only)
set(SOURCES
  src/gpgpu.cpp
  src/gpgpu_cl.cpp
  src/gpgpu_cu.cpp
  src/gblas.cpp
  src/gdnn.cpp
  src/database/database.cpp
  src/routines/common.cpp
  src/utilities/compile.cpp
  src/utilities/exceptions.cpp
  src/utilities/utilities.cpp
  src/api_common.cpp
  src/cache.cpp
  src/preprocessor.cpp
  src/routine.cpp
  src/routines/levelx/xinvert.cpp  # only source, don't include it as a test
)
set(PUBLIC_HEADERS
  include/gpgpu.h
  include/gblas_half.h
  include/gblas.h
  include/gdnn.h
)
set(HEADERS  # such that they can be discovered by IDEs such as CLion and Visual Studio
  src/database/apple_cpu_fallback.hpp
  src/database/database.hpp
  src/database/database_structure.hpp
  src/routines/level1/xamin.hpp
  src/routines/level1/xmax.hpp
  src/routines/level1/xmin.hpp
  src/routines/level1/xsum.hpp
  src/routines/common.hpp
  src/routines/routines.hpp
  src/utilities/buffer_test.hpp
  src/utilities/compile.hpp
  src/utilities/exceptions.hpp
  src/utilities/device_mapping.hpp
  src/utilities/utilities.hpp
  src/cache.hpp
  src/preprocessor.hpp
  src/routine.hpp
  src/cudnn.hpp
)
foreach(ROUTINE ${LEVEL1_ROUTINES})
  set(SOURCES ${SOURCES} src/routines/level1/${ROUTINE}.cpp)
  set(HEADERS ${HEADERS} src/routines/level1/${ROUTINE}.hpp)
endforeach()
foreach(ROUTINE ${LEVEL2_ROUTINES})
  set(SOURCES ${SOURCES} src/routines/level2/${ROUTINE}.cpp)
  set(HEADERS ${HEADERS} src/routines/level2/${ROUTINE}.hpp)
endforeach()
foreach(ROUTINE ${LEVEL3_ROUTINES})
  set(SOURCES ${SOURCES} src/routines/level3/${ROUTINE}.cpp)
  set(HEADERS ${HEADERS} src/routines/level3/${ROUTINE}.hpp)
endforeach()
foreach(ROUTINE ${LEVELX_ROUTINES})
  set(SOURCES ${SOURCES} src/routines/levelx/${ROUTINE}.cpp)
  set(HEADERS ${HEADERS} src/routines/levelx/${ROUTINE}.hpp)
endforeach()
foreach(ROUTINE ${DNN_ROUTINES})
  set(SOURCES ${SOURCES} src/routines/dnn/${ROUTINE}.cpp)
  set(HEADERS ${HEADERS} src/routines/dnn/${ROUTINE}.hpp)
endforeach()
foreach(DATABASE ${DATABASES})
  set(SOURCES ${SOURCES} src/database/kernels/${DATABASE}/${DATABASE}.cpp)
  set(HEADERS ${HEADERS} src/database/kernels/${DATABASE}/${DATABASE}.hpp)
  set(HEADERS ${HEADERS} src/database/kernels/${DATABASE}/${DATABASE}_16.hpp)
  set(HEADERS ${HEADERS} src/database/kernels/${DATABASE}/${DATABASE}_32.hpp)
  set(HEADERS ${HEADERS} src/database/kernels/${DATABASE}/${DATABASE}_64.hpp)
  set(HEADERS ${HEADERS} src/database/kernels/${DATABASE}/${DATABASE}_3232.hpp)
  set(HEADERS ${HEADERS} src/database/kernels/${DATABASE}/${DATABASE}_6464.hpp)
endforeach()

# Creates and links the library
if(BUILD_SHARED_LIBS)
  add_library(gpgpu SHARED ${SOURCES} ${HEADERS})
else(BUILD_SHARED_LIBS)
  add_library(gpgpu STATIC ${SOURCES} ${HEADERS})
endif()

target_link_libraries(gpgpu ${API_LIBRARIES})

# Includes directories:
target_include_directories(
  gpgpu PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/src
    ${API_INCLUDE_DIRS})

# Sets the proper __declspec(dllexport) keyword for Visual Studio when the library is built
if(MSVC)
  if(BUILD_SHARED_LIBS)
    target_compile_definitions(gpgpu PRIVATE COMPILING_DLL=1) # requires at least CMake 2.8.11
  endif()
endif()

# ==================================================================================================
