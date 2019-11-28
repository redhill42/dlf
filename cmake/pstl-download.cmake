cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(pstl-download NONE)
include(ExternalProject)

ExternalProject_Add(
  pstl
  SOURCE_DIR "@PSTL_SOURCE_DIR@"
  GIT_REPOSITORY
    https://github.com/intel/parallelstl.git
  GIT_TAG 20190522
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)
