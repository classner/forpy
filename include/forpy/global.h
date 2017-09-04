/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_GLOBAL_H_
#define FORPY_GLOBAL_H_

#include "./version.h"
#include <iostream>
#include <iomanip>      // std::setprecision
#include <glog/logging.h>


#if ! (defined NDEBUG) && defined(_MSC_VER)
// Solve a MSVC specific name clash between WinDef.h and <algorithm> :(
// See http://www.suodenjoki.dk/us/archive/2010/min-max.htm.
#define NOMINMAX
#include <Windows.h>
#endif

#include <string>

// Use 64-bit aligned memory for faster computations.
#define mmalloc(size) mkl_malloc(size, 64)
#define ffree(space) mkl_free(space)

// A define that makes it easier to find forbidden calls to pure virtual
// functions in debugging mode.
#if defined(NDEBUG) || !defined(_MSC_VER)
  #define VIRTUAL(type) =0
  #define VIRTUAL_VOID =0
  #define VIRTUAL_PTR =0
#else
  #define VIRTUAL(type) { DebugBreak(); return type();}
  #define VIRTUAL_VOID { DebugBreak(); }
  #define VIRTUAL_PTR { DebugBreak(); return nullptr; }
#endif

/**
 * A macro to disallow the copy constructor and operator= functions
 * This should be used in the private: declarations for a class
 * see http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml.
 */
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&);               \
  void operator=(const TypeName&)


namespace forpy {

  inline void init() {
    google::InitGoogleLogging("");
    FLAGS_logtostderr = 1;
    LOG(INFO) << "forpy version " << std::setprecision(2) << std::fixed <<
      static_cast<float>(FORPY_LIB_VERSION()) / 100.f << " initialized." <<
      std::defaultfloat;
  }

  // This library's exception type.
  class Forpy_Exception: public std::exception {
   public:
    explicit Forpy_Exception(const std::string &what)
      : whatstr(what) {}

    virtual const char* what() const throw() {
      return whatstr.c_str();
    };

    virtual ~Forpy_Exception() throw() {};

   private:
    const std::string whatstr;
  };

  class EmptyException: public Forpy_Exception {
   public:
    EmptyException() : Forpy_Exception("Tried to access an empty variant.") {};
  };
}  // namespace forpy

// Debugging support.
#if defined _MSC_VER
#define FBREAKP DebugBreak();
#else
#include <csignal>
#define FBREAKP std::raise(SIGINT);
#endif

#if defined RUNTIME_CHECKS
  #define FASSERT(condition)                                                   \
    if (!(condition))                                                          \
      FBREAKP
#else
  #define FASSERT(condition) 
#endif

// Library exports.
#if !defined(_MSC_VER)
#define DllExport
#elif __BUILD_FORPY_LIBRARY
#define DllExport __declspec( dllexport )
#else
#define DllExport __declspec( dllimport )
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
