cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(tbb-download NONE)
include(ExternalProject)

ExternalProject_Add(
  tbb
  SOURCE_DIR "@TBB_SOURCE_DIR@"
  GIT_REPOSITORY
    https://github.com/intel/tbb.git
  GIT_TAG
    2019_U6
  CONFIGURE_COMMAND ""
  BUILD_COMMAND
    make tbb_build_prefix=tbb
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)