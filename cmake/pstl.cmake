macro(fetch_pstl _download_module_path _download_root)
  set(PSTL_SOURCE_DIR ${_download_root}/pstl-src)

  configure_file(
    ${_download_module_path}/pstl-download.cmake
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

  set(PSTL_INCLUDE_DIRS ${PSTL_SOURCE_DIR}/include)
  include_directories(${PSTL_INCLUDE_DIRS})
endmacro()
