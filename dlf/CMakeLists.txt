add_library(dlf "")

find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "src/onnx.proto")

file(GLOB_RECURSE DLF_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
file(GLOB_RECURSE DLF_HDRS "${CMAKE_CURRENT_LIST_DIR}/include/*.h")
file(GLOB_RECURSE DLF_PRIVATE_HDRS "${CMAKE_CURRENT_LIST_DIR}/src/*.hpp")

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
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
)

target_link_libraries(
  dlf
  PRIVATE
    ${PROTOBUF_LIBRARIES}
    ${TBB_LIBRARIES}
    ${MKL_LIBRARIES}
    gpgpu
)

if(NOT DEFINED ENV{SKIP_TESTS})
  include(${PROJECT_SOURCE_DIR}/cmake/googletest.cmake)
  fetch_googletest(${PROJECT_SOURCE_DIR}/cmake ${PROJECT_BINARY_DIR}/googletest)
  enable_testing()
  add_subdirectory(test)
endif()
