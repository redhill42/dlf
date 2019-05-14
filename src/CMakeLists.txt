add_library(dlfl "")

include(FindProtobuf)
find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "onnx.proto")

set(LLVM_DIR "/usr/local/opt/llvm/lib/cmake/llvm")
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(llvm_libs
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

file(GLOB_RECURSE DLFL_SRCS "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
file(GLOB_RECURSE DLFL_HDRS "${CMAKE_CURRENT_LIST_DIR}/*.h")

target_sources(
  dlfl
  PRIVATE
    ${DLFL_SRCS}
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
    ${llvm_libs}
)
