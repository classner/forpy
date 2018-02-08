if (WITH_PYTHON)
  set (CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  file (GLOB_RECURSE Forpy_CPP "*.cpp")
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR
      "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    # There are problems when linking with THIN_LTO with Clang. :-/
    pybind11_add_module(forpy MODULE NO_EXTRAS ${Forpy_CPP})
  else ()
    pybind11_add_module(forpy MODULE THIN_LTO ${Forpy_CPP})
  endif()
  target_link_libraries(forpy PRIVATE forpy_core)
  set_target_properties(forpy PROPERTIES PREFIX "")
  set_target_properties(forpy PROPERTIES
    COTIRE_PREFIX_HEADER_INCLUDE_PATH
    "${CMAKE_SOURCE_DIR}/build_support/external")
  if (NOT PCH_ENABLED)
      set_target_properties(forpy PROPERTIES COTIRE_ENABLE_PRECOMPILED_HEADER FALSE)
  endif()
  cotire (forpy)
  add_custom_command (
    OUTPUT "forpy_CXX_cotire.cmake"
    COMMENT "Noop proxy for ninja about the cotire generated PCHs.")

  if (APPLE)
    set_target_properties (forpy PROPERTIES SUFFIX .so)
    set_target_properties (forpy PROPERTIES INSTALL_RPATH "@loader_path/")
  endif()
  if (WIN32)
    set_target_properties (forpy PROPERTIES SUFFIX .pyd)
  endif()
  if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set_target_properties (forpy PROPERTIES INSTALL_RPATH "$ORIGIN/:$$ORIGIN")
    set_target_properties (forpy PROPERTIES SUFFIX .so)
  endif()
  install (TARGETS forpy DESTINATION bindings/python/)
  add_custom_command (TARGET forpy POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy
       "$<TARGET_FILE:forpy>"
       "${PROJECT_SOURCE_DIR}/$<TARGET_FILE_NAME:forpy>"
    COMMENT "Copying to root directory")
endif()
