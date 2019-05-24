cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(clblast-download NONE)
include(ExternalProject)

ExternalProject_Add(
  clblast
  SOURCE_DIR "@CLBLAST_DOWNLOAD_ROOT@/clblast-src"
  BINARY_DIR "@CLBLAST_DOWNLOAD_ROOT@/clblast-build"
  GIT_REPOSITORY
    https://github.com/CNugteren/CLBlast.git
  GIT_TAG
    1.5.0
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND ""
)