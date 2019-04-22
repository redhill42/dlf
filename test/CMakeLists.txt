file(GLOB_RECURSE EXAMPLE_TESTS "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_executable(unit_tests ${EXAMPLE_TESTS})
target_link_libraries(unit_tests gtest gmock example)

add_test(
  NAME
    unit
  COMMAND
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/unit_tests
)
