macro(fetch_clblast _download_module_path _download_root)
  set(CLBLAST_DOWNLOAD_ROOT ${_download_root})
  configure_file(
    ${_download_module_path}/clblast-download.cmake
    ${_download_root}/CMakeLists.txt
    @ONLY
  )
  unset(CLBLAST_DOWNLOAD_ROOT)

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

  set(TUNERS OFF)
  add_subdirectory(${_download_root}/clblast-src)

  include_directories("${clblast_SOURCE_DIR}/include")
endmacro()
