/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_GLOBAL_H_
#define FORPY_GLOBAL_H_

#include <glog/logging.h>
#include <iomanip>  // std::setprecision
#include <iostream>
#include <thread>
#include "./version.h"
#ifdef WITHGPERFTOOLS
#include <gperftools/profiler.h>
#endif

#if !(defined NDEBUG) && defined(_MSC_VER)
// Solve a MSVC specific name clash between WinDef.h and <algorithm> :(
// See http://www.suodenjoki.dk/us/archive/2010/min-max.htm.
#define NOMINMAX
#include <Windows.h>
#endif

#include <string>
inline bool ends_with(std::string const& value, std::string const& ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

// A define that makes it easier to find forbidden calls to pure virtual
// functions in debugging mode.
#if defined(NDEBUG) || !defined(_MSC_VER)
#define VIRTUAL(type) = 0
#define VIRTUAL_VOID = 0
#define VIRTUAL_PTR = 0
#else
#define VIRTUAL(type) \
  {                   \
    DebugBreak();     \
    return type();    \
  }
#define VIRTUAL_VOID \
  { DebugBreak(); }
#define VIRTUAL_PTR \
  {                 \
    DebugBreak();   \
    return nullptr; \
  }
#endif

/**
 * A macro to disallow the copy constructor and operator= functions
 * This should be used in the private: declarations for a class
 * see http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml.
 */
#define NOASSIGN_BUT_MOVE(TypeName)              \
  TypeName(const TypeName&) = delete;            \
  TypeName& operator=(const TypeName&) = delete; \
  TypeName(TypeName&&) = default;                \
  TypeName& operator=(TypeName&&) = default;

#define MOVE_ASSIGN(TypeName)     \
  TypeName(TypeName&&) = default; \
  TypeName& operator=(TypeName&&) = default;

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&);               \
  TypeName& operator=(const TypeName&);

namespace forpy {

inline void init() {
  google::InitGoogleLogging("");
  FLAGS_logtostderr = 1;
  LOG(INFO) << "forpy version " << std::setprecision(2) << std::fixed
            << static_cast<float>(FORPY_LIB_VERSION()) / 100.f
            << " initialized." << std::defaultfloat;
  LOG(INFO) << "Detected support for " << std::thread::hardware_concurrency()
            << " hardware threads.";
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
const static bool SKLEARN_COMPAT =
#ifdef FORPY_SKLEARN_COMPAT
    true;
#else
    false;
#endif
#pragma clang diagnostic pop

// This library's exception type.
class ForpyException : public std::exception {
 public:
  explicit ForpyException(const std::string& what) : whatstr(what) {}

  virtual const char* what() const throw() { return whatstr.c_str(); };

  virtual ~ForpyException() throw(){};

 private:
  const std::string whatstr;
};

class EmptyException : public ForpyException {
 public:
  EmptyException() : ForpyException("Tried to access an empty variant."){};
};
}  // namespace forpy

// Debugging support.
#if defined _MSC_VER
#define FBREAKP DebugBreak()
#else
#include <csignal>
#define FBREAKP std::raise(SIGINT)
#endif

#if defined RUNTIME_CHECKS
#define FASSERT(condition)             \
  if (!(condition)) {                  \
    LOG(ERROR) << "Assertion failed!"; \
    FBREAKP;                           \
  }
#else
#define FASSERT(condition)
#endif

// Library exports.
#if !defined(_MSC_VER)
#define DllExport
#elif __BUILD_FORPY_LIBRARY
#define DllExport __declspec(dllexport)
#else
#define DllExport __declspec(dllimport)
#endif

#ifdef __BUILD_FORPY_LIBRARY
#define TemplateExport template class
#define TemplateFuncExport template
#define ExportVar
#else
#define TemplateExport extern template class
#define TemplateFuncExport extern template
#define ExportVar extern
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#ifdef WITH_OPENCV
static bool FORPY_OPENCV_AVAILABLE() { return true; }
#else
static bool FORPY_OPENCV_AVAILABLE() { return false; }
#endif
#pragma clang diagnostic pop
#endif  // FORPY_GLOBAL_H_
