cmake_minimum_required(VERSION 3.1)

project(DLF C CXX)
if("$ENV{CUDA_VERSION}" STREQUAL "8.0" OR "$ENV{CUDA_VERSION}" STREQUAL "9.0")
  set(CMAKE_CXX_STANDARD 14)
else()
  set(CMAKE_CXX_STANDARD 17)
endif()

# Options and their default values
option(BUILD_SHARED_LIBS "Build a shared (ON) or static library (OFF)" ON)

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

# C++ compiler settings
if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  set(FLAGS "${FLAGS} -Wall")
  set(FLAGS "${FLAGS} -Wno-unknown-pragmas")
  set(FLAGS "${FLAGS} -Wno-unused-parameter -Wno-sign-compare")
elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
  set(FLAGS "${FLAGS} -Wextra -Wno-c++98-compat -Wno-c++98-compat-pedantic")
  set(FLAGS "${FLAGS} -Wno-unused-parameter -Wno-sign-compare")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")

include(cmake/tbb.cmake)
fetch_tbb(${PROJECT_SOURCE_DIR}/cmake ${PROJECT_BINARY_DIR}/tbb)

include(cmake/pstl.cmake)
fetch_pstl(${PROJECT_SOURCE_DIR}/cmake ${PROJECT_BINARY_DIR}/pstl)

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
  set(BLAS_LIBRARY ${MKL_LIBRARIES})
elseif(APPLE)
  find_library(BLAS_LIBRARY Accelerate)
endif()

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
target_link_libraries(demo gpgpu dlf ${TBB_LIBRARIES} ${BLAS_LIBRARY})

if (HAS_OPENCV)
  add_executable(imagenet imagenet.cpp)
  target_link_libraries(imagenet gpgpu dlf ${TBB_LIBRARIES} ${BLAS_LIBRARY} ${OpenCV_LIBS})
endif()

if (HAS_OPENCV)
  add_executable(mnist mnist.cpp)
  target_link_libraries(mnist gpgpu dlf ${TBB_LIBRARIES} ${BLAS_LIBRARY} ${OpenCV_LIBS})
endif()

add_executable(optimize optimize.cpp)
target_link_libraries(optimize gpgpu dlf ${TBB_LIBRARIES} ${BLAS_LIBRARY})
