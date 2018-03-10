file (GLOB_RECURSE Common_CPP "*.cpp")
add_library (forpy_core SHARED ${Common_CPP})
add_dependencies(forpy_core glog)
target_compile_features (forpy_core PRIVATE ${REQ_CPP11_FEATURES})
set_property(TARGET forpy_core PROPERTY CXX_STANDARD 14)
target_compile_definitions (forpy_core PRIVATE -D__BUILD_FORPY_LIBRARY)
target_link_libraries (forpy_core ${GLOG_BINARY})
if (WITHGPERFTOOLS)
  target_link_libraries (forpy_core profiler)
endif()
if (APPLE)
  set_target_properties (forpy_core PROPERTIES INSTALL_RPATH "@rpath")
endif()
set_target_properties(forpy_core PROPERTIES
  COTIRE_PREFIX_HEADER_INCLUDE_PATH
  "${CMAKE_SOURCE_DIR}/build_support/external")
if (NOT PCH_ENABLED)
  set_target_properties(forpy_core PROPERTIES COTIRE_ENABLE_PRECOMPILED_HEADER FALSE)
endif()
cotire (forpy_core)
add_custom_command (
  OUTPUT "forpy_core_CXX_cotire.cmake" PRE_BUILD
  COMMENT "Noop proxy for ninja about the cotire generated PCHs.")

install (TARGETS forpy_core DESTINATION lib)
if (WITH_PYTHON)
  add_custom_command (TARGET forpy_core POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy
       "$<TARGET_FILE:forpy_core>"
       "${CMAKE_BINARY_DIR}/bindings/python/$<TARGET_FILE_NAME:forpy_core>"
    COMMENT "Copying to python directory")
  install (TARGETS forpy_core DESTINATION bindings/python/)
  add_custom_command (TARGET forpy_core POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy
    "$<TARGET_FILE:forpy_core>"
    "${PROJECT_SOURCE_DIR}/$<TARGET_FILE_NAME:forpy_core>"
    COMMENT "Copying to root directory")
endif()
add_custom_command (TARGET forpy_core POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy
     "$<TARGET_FILE:forpy_core>"
     "${CMAKE_BINARY_DIR}/test/$<TARGET_FILE_NAME:forpy_core>"
  COMMENT "Copying to test directory")
install (TARGETS forpy_core DESTINATION tests/)
add_custom_command (TARGET forpy_core POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy
     "$<TARGET_FILE:forpy_core>"
     "${CMAKE_BINARY_DIR}/examples/c++/$<TARGET_FILE_NAME:forpy_core>"
  COMMENT "Copying to C++ examples directory")
install (TARGETS forpy_core DESTINATION examples/c++)
# install(TARGETS forpy_core LIBRARY DESTINATION .)
