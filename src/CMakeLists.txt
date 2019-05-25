add_library(dlfl "")

find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "onnx.proto")

set(LLVM_DIR $ENV{LLVM_DIR})
find_package(LLVM CONFIG)
if (LLVM_FOUND)
  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  include_directories(${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})
  llvm_map_components_to_libnames(LLVM_LIBS
    Analysis
    Core
    ExecutionEngine
    InstCombine
    Object
    OrcJIT
    RuntimeDyld
    ScalarOpts
    Support
    native)
endif()

file(GLOB_RECURSE DLFL_SRCS "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
file(GLOB_RECURSE DLFL_HDRS "${CMAKE_CURRENT_LIST_DIR}/*.h")
file(GLOB_RECURSE DLFL_PRIVATE_HDRS "${CMAKE_CURRENT_LIST_DIR}/*.hpp")

target_sources(
  dlfl
  PRIVATE
    ${DLFL_SRCS}
    ${DLFL_PRIVATE_HDRS}
    ${PROTO_SRCS}
  PUBLIC
    ${DLFL_HDRS}
)

target_include_directories(
  dlfl
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Protobuf_INCLUDE_DIR}
)

target_link_libraries(
  dlfl
  PRIVATE
    ${Protobuf_LIBRARIES}
    ${LLVM_LIBS}
    ${TBB_LIBS}
    ${MKL_LIBRARIES}
    ${OpenCL_LIBRARY}
    ${CUDA_LIBRARIES}
    clblast
)
