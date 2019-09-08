file(GLOB_RECURSE DLF_TESTS "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
add_executable(unit_tests ${DLF_TESTS})
target_link_libraries(unit_tests gmock_main gpgpu dlf ${TBB_LIBRARIES})

if (USE_MKL)
  target_link_libraries(unit_tests ${MKL_LIBRARIES})
elseif (APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework Accelerate")
endif()

find_package(GMP)
if (GMP_FOUND)
  add_definitions(-DHAS_GMP=1)
  include_directories(${GMP_INCLUDES})
  target_link_libraries(unit_tests ${GMP_LIB} ${GMPXX_LIB})
endif()

set(TEST_FILE ${CMAKE_BINARY_DIR}/data/resnet18v1.onnx)
if(NOT EXISTS ${TEST_FILE})
  file(DOWNLOAD
    https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.onnx
    ${TEST_FILE}
    SHOW_PROGRESS
  )
endif()

add_test(
  NAME
    unit
  COMMAND
    ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/unit_tests
)
