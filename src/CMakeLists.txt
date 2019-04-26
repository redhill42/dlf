add_library(example "")

include(FindProtobuf)
find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "onnx.proto")

file(GLOB_RECURSE EXAMPLE_SRCS "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
file(GLOB_RECURSE EXAMPLE_HDRS "${CMAKE_CURRENT_LIST_DIR}/*.h")

target_sources(
  example
  PRIVATE
    ${EXAMPLE_SRCS}
    ${PROTO_SRCS}
  PUBLIC
    ${EXAMPLE_HDRS}
)

target_include_directories(
  example
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Protobuf_INCLUDE_DIR}
)

target_link_libraries(
  example
  PRIVATE
    ${Protobuf_LIBRARIES}
)