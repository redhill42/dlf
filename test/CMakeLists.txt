file(GLOB_RECURSE DLFL_TESTS "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
add_executable(unit_tests ${DLFL_TESTS})
target_link_libraries(unit_tests gtest gmock dlfl ${TBB_LIBS})

if (MKL_FOUND)
  target_link_libraries(unit_tests ${MKL_LIBRARIES})
elseif (APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework Accelerate")
endif()

find_package(GMP)
if (GMP_FOUND)
  add_definitions(-DHAS_GMP=1)
  include_directories(${GMP_INCLUDES})
  target_link_libraries(unit_tests ${GMP_LIBRARIES})
endif()

add_test(
  NAME
    unit
  COMMAND
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/unit_tests
)
