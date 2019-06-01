add_library(dlf "")

find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "onnx.proto")

file(GLOB_RECURSE DLF_SRCS "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
file(GLOB_RECURSE DLF_HDRS "${CMAKE_CURRENT_LIST_DIR}/*.h")
file(GLOB_RECURSE DLF_PRIVATE_HDRS "${CMAKE_CURRENT_LIST_DIR}/*.hpp")

target_sources(
  dlf
  PRIVATE
    ${DLF_SRCS}
    ${DLF_PRIVATE_HDRS}
    ${PROTO_SRCS}
  PUBLIC
    ${DLF_HDRS}
)

target_include_directories(
  dlf
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Protobuf_INCLUDE_DIR}
)

target_link_libraries(
  dlf
  PRIVATE
    ${Protobuf_LIBRARIES}
    ${TBB_LIBRARIES}
    ${MKL_LIBRARIES}
    gpgpu
)
