cmake_minimum_required(VERSION 3.1)

project(DLF C CXX)
if("$ENV{CUDA_VERSION}" STREQUAL "8.0")
  set(CMAKE_CXX_STANDARD 14)
else()
  set(CMAKE_CXX_STANDARD 17)
endif()

# place binaries and libraries according to GNU standards
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

set(CMAKE_MACOSX_RPATH 1)

set(CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/cmake
  ${CMAKE_MODULE_PATH}
)

include(cmake/tbb.cmake)
fetch_tbb(${PROJECT_SOURCE_DIR}/cmake ${PROJECT_BINARY_DIR}/tbb)

if(NOT(DEFINED USE_MKL))
  if(APPLE)
    set(USE_MKL NO)
  else()
    set(USE_MKL YES)
  endif()
endif()
if(USE_MKL)
  find_package(MKL)
  set(USE_MKL ${MKL_FOUND})
endif()
if(USE_MKL)
  add_definitions(-DHAS_MKL=1)
  include_directories(${MKL_INCLUDE_DIRS})
endif()

# OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

# CUDA
find_package(CUDA REQUIRED)
find_cuda_helper_libs("cuda")
find_cuda_helper_libs("nvrtc")
find_cuda_helper_libs("cublas")
find_cuda_helper_libs("cudnn")
set(CUDA_LIBRARIES ${CUDA_cuda_LIBRARY} ${CUDA_nvrtc_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_cudnn_LIBRARY})
include_directories(${CUDA_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV CONFIG)
if(OpenCV_FOUND)
  set(HAS_OPENCV YES)
  add_definitions(-DHAS_OPENCV=1)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# add sub-directories that we want CMake to scan
add_subdirectory(gpgpu)
include_directories(${gpgpu_SOURCE_DIR}/include ${gpgpu_SOURCE_DIR}/src)
add_subdirectory(dlf)

add_executable(demo main.cpp)
target_link_libraries(demo gpgpu dlf ${TBB_LIBRARIES})

if (USE_MKL)
  target_link_libraries(demo ${MKL_LIBRARIES})
elseif (APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework Accelerate")
endif()

if (HAS_OPENCV)
  add_executable(imagenet imagenet.cpp)
  target_link_libraries(imagenet gpgpu dlf ${TBB_LIBRARIES} ${OpenCV_LIBS})
  if (USE_MKL)
    target_link_libraries(imagenet ${MKL_LIBRARIES})
  endif()
endif()
