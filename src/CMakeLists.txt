add_library(example "")

file(GLOB_RECURSE EXAMPLE_SRCS "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
file(GLOB_RECURSE EXAMPLE_HDRS "${CMAKE_CURRENT_LIST_DIR}/*.h")

target_sources(
  example
  PRIVATE
    ${EXAMPLE_SRCS}
  PUBLIC
    ${EXAMPLE_HDRS}
)

target_include_directories(
  example
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)
