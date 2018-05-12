cmake_minimum_required (VERSION 3.2)
include(ExternalProject)

project (forpy C CXX)
list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/build_support/cmake/modules)
list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/build_support/cmake/cotire/CMake/)
include(cotire)  # Automatic precompiled headers for fast builds.

# Versioning.
set (forpy_VERSION_MAJOR 2)
set (forpy_VERSION_MINOR 0)
set (forpy_VERSION_PATCH 5)
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
if (${CMBT_UPPER} MATCHES "DEBUG" OR ASSERTIONS)
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
  find_package (NumPy REQUIRED)
  include_directories (${NUMPY_INCLUDE_DIRS})
  add_subdirectory("build_support/external/pybind11-master")
endif ()

# Use C++11 features.
set (REQ_CPP11_FEATURES
     cxx_strong_enums
     cxx_auto_type)
   # Use optimizations.
if (NOT WIN32)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -ffast-math")
else ()
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest")
endif ()

# Fix clang warnings.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-tautological-compare -Wno-logical-op-parentheses -Wno-deprecated-register -Wno-comment")
endif()

# SIMD tools.
if (NOT FORCE_WITHOUT_SIMD)
  CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_OPT_ARCH_AVX_SUPPORTED)
  if(COMPILER_OPT_ARCH_AVX_SUPPORTED)
    set (SIMD_ENABLED On CACHE STRING "Enable SIMD instructions." FORCE)
  endif()
endif()
#if (SIMD_ENABLED)
#  add_definitions (-DSIMD_ENABLED)
#  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
#endif()

if (NOT PCH)
  set (PCH_ENABLED Off CACHE STRING "Enable precompiled headers." FORCE)
endif()
if (NOT SKLEARN_COMPAT)
  set (SKLEARN_COMPAT_MODE Off CACHE STRING "Enable ScikitLearn compatibility." FORCE)
else ()
  set (SKLEARN_COMPAT_MODE On CACHE STRING "Enable ScikitLearn compatibility." FORCE)
endif()
if (SKLEARN_COMPAT_MODE)
  add_definitions (-DFORPY_SKLEARN_COMPAT)
endif()

# Configure the dependencies.
include_directories ("build_support/external/eigen3-v3.3")
add_definitions (-DEIGEN_MPL2_ONLY)
include_directories ("build_support/external/cereal-v1.2.2/include")
include_directories ("build_support/external/variant-v1.1.4/include")
include_directories ("build_support/external/simd-master/include")
include_directories ("build_support/external/cpptask-master/include")
include_directories ("build_support/external/skasort")

# Targets.
add_subdirectory ("build_support/external/googletest-v1.8.0")
add_subdirectory ("build_support/external/glog")
include_directories (${GLOG_INCLUDE})
include_directories ("${CMAKE_CURRENT_BINARY_DIR}/glog/glog/include")

# Profiling tools.
if (WITHGPERFTOOLS)
  set (CMAKE_BUILD_TYPE RelWithDebInfo)
  #set (CMAKE_BUILD_TYPE Debug)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITHGPERFTOOLS")
endif()

add_subdirectory ("include")
add_subdirectory ("src")
add_subdirectory ("bindings/python")
add_subdirectory ("test")

enable_testing()

# Create the documentation.
add_subdirectory (docs)
get_directory_property(DOXYGEN_FOUND DIRECTORY docs DEFINITION DOXYGEN_FOUND)

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
if (SKLEARN_COMPAT_MODE)
  message ("ScikitLearn compatibility mode enabled.")
  message ("")
endif()
message ("Compiler options:")
message ("  Build type: " ${CMAKE_BUILD_TYPE})
if (${PCH_ENABLED})
  message ("  Pre-compiled headers enabled.")
else ()
  message ("  Pre-compiled headers disabled.")
endif()
if (${CMBT_UPPER} MATCHES "DEBUG" OR ASSERTIONS)
  message ("  Assertions enabled.")
else ()
  message ("  Assertions disabled.")
endif ()
if (SIMD_ENABLED)
  message ("  SIMD instructions enabled.")
else ()
  message ("  SIMD instructions disabled.")
endif()
message ("  Profiling tools: " ${WITHGPERFTOOLS})
message ("  Compiler flags: " ${CMAKE_CXX_COMPILE_FLAGS})
message ("  Compiler cxx debug flags: " ${CMAKE_CXX_FLAGS_DEBUG})
message ("  Compiler cxx release flags: " ${CMAKE_CXX_FLAGS_RELEASE})
message ("  Compiler cxx min size flags: " ${CMAKE_CXX_FLAGS_MINSIZEREL})
message ("  Compiler cxx flags: " ${CMAKE_CXX_FLAGS})
message ("")
message ("Installation options:")
message ("  Installation prefix path: " ${CMAKE_INSTALL_PREFIX})
message ("")
message ("Documentation:")
if (DOXYGEN_FOUND)
  message ("  Doxygen found. Building doc.")
else ()
  message ("  Doxygen not found.")
endif ()
message ("--------------------------------------------------------------------")
