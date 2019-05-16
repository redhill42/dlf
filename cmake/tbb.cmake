macro(fetch_tbb _download_module_path _download_root)
  set(TBB_SOURCE_DIR ${_download_root}/tbb-src)

  configure_file(
    ${_download_module_path}/tbb-download.cmake
    ${_download_root}/CMakeLists.txt
    @ONLY
  )

  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY
      ${_download_root}
  )
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY
      ${_download_root}
  )

  set(TBB_INCLUDE_DIRS ${TBB_SOURCE_DIR}/include)
  include_directories(${TBB_INCLUDE_DIRS})

  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(TBB_LIBRARY_DIRS ${TBB_SOURCE_DIR}/build/tbb_debug)
    set(TBB_LIBS tbb_debug tbbmalloc_debug)
  else (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(TBB_LIBRARY_DIRS ${TBB_SOURCE_DIR}/build/tbb_release)
    set(TBB_LIBS tbb tbbmalloc)
  endif (CMAKE_BUILD_TYPE STREQUAL "Debug")
  link_directories(${TBB_LIBRARY_DIRS})

endmacro()
