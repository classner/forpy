FILE(GLOB_RECURSE Test_CPP "cpp/*.cpp")
foreach (OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
  string (TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_CURRENT_BINARY_DIR}/)
endforeach (OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)

add_executable (forpy_tests ${Test_CPP})
target_compile_features(forpy_tests PRIVATE ${REQ_CPP11_FEATURES})

if (APPLE)
  set_target_properties (forpy_tests PROPERTIES INSTALL_RPATH "@loader_path/")
else()
  set_target_properties (forpy_tests PROPERTIES INSTALL_RPATH "$ORIGIN/:$$ORIGIN")
endif()
target_link_libraries(forpy_tests gtest_main forpy_core ${OpenCV_COMPATIBLE_LIBRARIES})
set_target_properties(forpy_tests PROPERTIES
  COTIRE_PREFIX_HEADER_INCLUDE_PATH
  "${CMAKE_SOURCE_DIR}/build_support/external")
cotire (forpy_tests)

# Installation.
install (TARGETS forpy_tests DESTINATION tests)

# Add the tests to the test suite.
add_test (NAME CPPTestRun
  COMMAND forpy_tests --gtest_filter=-*Speed*
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME PyTest_Reg
  COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/python/regression.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)
add_test(NAME PyTest_RegDemo
  COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/python/regression_example.py --test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)
add_test(NAME PyTest_ThreshOpt
  COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/python/threshopt.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)
add_test(NAME PyTest_DProv
  COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/python/data_providers.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)
add_test(NAME PyTest_Leafs
  COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/python/leafs.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)
add_test(NAME PyTest_Deciders
  COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/python/deciders.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)
add_test(NAME PyTest_Pickle
  COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/python/pickle_.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python)

enable_testing()

find_program(VALGRIND "valgrind")
if(VALGRIND)
    add_custom_target(valgrind
      COMMAND "${VALGRIND}" --tool=memcheck --leak-check=yes
      --show-reachable=yes --num-callers=20 --track-fds=yes --
      $<TARGET_FILE:forpy_tests> --gtest_filter=*-*.Speed*)
endif()
