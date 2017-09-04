cmake_minimum_required (VERSION 3.2)
include(ExternalProject)

project (forpy C CXX)
list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/build_support/cmake/modules)

# Versioning.
set (forpy_VERSION_MAJOR 2)
set (forpy_VERSION_MINOR 0)
set (forpy_VERSION_PATCH 0)
set (forpy_VERSION
  "${forpy_VERSION_MAJOR}.${forpy_VERSION_MINOR}.${forpy_VERSION_PATCH}")
set (forpy_VERSION_NUMBER
  "${forpy_VERSION_MAJOR}${forpy_VERSION_MINOR}${forpy_VERSION_PATCH}")

# Set a default build type if none was specified.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message (STATUS "Setting build type to 'Release' since none was specified.")
  set (CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui.
  set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} CMBT_UPPER)
if (${CMBT_UPPER} MATCHES "DEBUG")
  message("Runtime checks enabled!")
  add_definitions (-DRUNTIME_CHECKS)
endif()

# Setup the library.
include_directories (include)
include_directories (${CMAKE_CURRENT_BINARY_DIR}/include)

# Python.
find_package (PythonInterp)
find_package (PythonLibs)
if (PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)
  option (WITH_PYTHON "Build the Python interface" ON)
else()
  option (WITH_PYTHON "Build the Python interface" OFF)
endif()

# Configure.
if (WITH_PYTHON)
  if (NOT PYTHONINTERP_FOUND OR NOT PYTHONLIBS_FOUND)
    message (FATAL_ERROR "You specified WITH_PYTHON, but interpreter or libs \
                          weren't found!")
  endif ()
  add_definitions (-DPYTHON_ENABLED)
  include_directories (${PYTHON_INCLUDE_PATH})
  add_subdirectory(build_support/external/pybind11-master)
endif ()

# Use C++11 features.
set (REQ_CPP11_FEATURES
     cxx_strong_enums
     cxx_auto_type)

# Check for OpenMP support.
find_package (OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# Fix clang warnings.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-tautological-compare -Wno-logical-op-parentheses -Wno-deprecated-register -Wno-comment")
endif()

# Configure the dependencies.
if (WITH_PYTHON)
  find_package (NumPy REQUIRED)
  include_directories (${NUMPY_INCLUDE_DIRS})
  include_directories ("build_support/external/pybind11-v2.2/include")
endif()
include_directories ("build_support/external/eigen3-v3.3")
add_definitions (-DEIGEN_MPL2_ONLY)
include_directories ("build_support/external/cereal-v1.2.2/include")
include_directories ("build_support/external/variant-v1.1.4/include")

# Targets.
add_subdirectory ("build_support/external/googletest-v1.8.0")
add_subdirectory ("build_support/external/glog")
include_directories (${GLOG_INCLUDE})
include_directories ("${CMAKE_CURRENT_BINARY_DIR}/glog/glog/include")

add_subdirectory ("include")
add_subdirectory ("src")
add_subdirectory ("bindings/python")
add_subdirectory ("test")

enable_testing()
# Create the examples.
#add_subdirectory (examples/c++)

# Create the documentation.
#add_subdirectory (documentation)

# Summarize.
message ("--------------------------------------------------------------------")
message ("Configuration summary")
message ("--------------------------------------------------------------------")
message ("")
message ("Library options:")
if (WITH_PYTHON)
  message ("  Building Python bindings for version " ${PYTHONLIBS_VERSION_STRING})
  message ("  Python interpreter:\t" ${PYTHON_EXECUTABLE})
  message ("  Python include path:\t" ${PYTHON_INCLUDE_DIRS})
  message ("  Python library:\t" ${PYTHON_LIBRARIES})
else ()
  message ("  Not building Python bindings")
endif ()
message ("")
message ("Compiler options:")
message ("  Build type: " ${CMAKE_BUILD_TYPE})
message ("  Compiler flags: " ${CMAKE_CXX_COMPILE_FLAGS})
message ("  Compiler cxx debug flags: " ${CMAKE_CXX_FLAGS_DEBUG})
message ("  Compiler cxx release flags: " ${CMAKE_CXX_FLAGS_RELEASE})
message ("  Compiler cxx min size flags: " ${CMAKE_CXX_FLAGS_MINSIZEREL})
message ("  Compiler cxx flags: " ${CMAKE_CXX_FLAGS})
message ("")
message ("Installation options:")
message ("  Installation prefix path: " ${CMAKE_INSTALL_PREFIX})
message ("")
message ("--------------------------------------------------------------------")
