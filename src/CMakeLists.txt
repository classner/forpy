file (GLOB_RECURSE Common_CPP "*.cpp")
add_library (forpy_core SHARED ${Common_CPP})
add_dependencies(forpy_core glog)
target_compile_features (forpy_core PRIVATE ${REQ_CPP11_FEATURES})
set_property(TARGET forpy_core PROPERTY CXX_STANDARD 14)
target_compile_definitions (forpy_core PRIVATE -D__BUILD_FORPY_LIBRARY)
target_link_libraries (forpy_core ${GLOG_BINARY})
if (APPLE)
  set_target_properties (forpy_core PROPERTIES INSTALL_RPATH "@rpath")
endif()

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
