add_library(dlfl "")

find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "onnx.proto")

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
    ${TBB_LIBS}
    ${MKL_LIBRARIES}
    ${OpenCL_LIBRARY}
    ${CUDA_LIBRARIES}
    gpgpu
)
