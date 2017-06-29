#pragma once
#ifndef BINDINGS_PYTHON_MACROS_H_
#define BINDINGS_PYTHON_MACROS_H_

#include <forpy/util/macros.h>

// Python export macros.
#define FORPY_EXPCLASS_EQ(CLS, NAME)                                \
  py::class_<CLS, std::shared_ptr<CLS>> NAME(m, #CLS);                  \
  NAME.def("__eq__", [](const CLS &a, const CLS &b) {return a == b;}, py::is_operator()); \
  NAME.def("__ne__", [](const CLS &a, const CLS &b) {return ! (a == b);}, py::is_operator());

#define FORPY_EXPCLASS(CLS, NAME)                   \
  py::class_<CLS, std::shared_ptr<CLS>> NAME(m, #CLS);

#define FORPY_EXPCLASS_PARENT(CLS, NAME, PARENT)               \
  py::class_<CLS, std::shared_ptr<CLS>> NAME(m, #CLS, PARENT);

#define FORPY_EXPFUNC(OBJ, CLS, FUNC)       \
  OBJ .def(#FUNC, &CLS::FUNC);

// Stages for EXPVFUNC.
#define FORPY_EXPVFUNC__(TMPLF, OBJ, FNAME, CNAME, PARAMTF, PARAMF, ARGSPEC) \
  FORPY_UNPACK(TMPLF)                                 \
  void exp_##FNAME(py::class_<CNAME,std::shared_ptr<CNAME>> &OBJ()) {   \
    OBJ().def(#FNAME, [] PARAMTF { return self.FNAME PARAMF ; } ARGSPEC); \
  };
#define FORPY_EXPVFUNC_(TMPLF, OBJ, FNAME, CNAME, PARAMTF, PARAMF) \
  FORPY_EXPVFUNC__(FORPY_GEN_TMPLMARK_##TMPLF, OBJ, FNAME, CNAME, PARAMTF, PARAMF, )
#define FORPY_EXPVFUNC_ARGS(TMPLF, OBJ, FNAME, CNAME, PARAMTF, PARAMF, ARGSPEC) \
  FORPY_EXPVFUNC__(FORPY_GEN_TMPLMARK_##TMPLF, OBJ, FNAME, CNAME, PARAMTF, PARAMF, FORPY_UNPACK(, FORPY_UNPACK ARGSPEC))
#define FORPY_EXPVFUNC(FID, TMPL, CNAME, OBJ)                           \
  FORPY_EXPVFUNC_(TMPL,                                                 \
                  OBJ,                                                  \
                  FID##_NAME,                                           \
                  CNAME,                                                \
                  (CNAME &self, FORPY_INSERT_TPARMS_##TMPL(FID##_PARAMTYPESNNAMES)), \
                  (FID##_PARAMNAMES))
#define FORPY_EXPVFUNC_DEF(FID, TMPL, CNAME, OBJ, ARGSPEC)                   \
  FORPY_EXPVFUNC_ARGS(TMPL,                                        \
                      OBJ,                                              \
                      FID##_NAME,                                       \
                      CNAME,                                            \
                      (CNAME &self, FORPY_INSERT_TPARMS_##TMPL(FID##_PARAMTYPESNNAMES)), \
                      (FID##_PARAMNAMES),                               \
                      ARGSPEC)

// Stages for EXPVFUNC_CALL.
#define FORPY_EXPVFUNC_CALL__(FNAME) exp_##FNAME
#define FORPY_EXPVFUNC_CALL_(FNAME) FORPY_EXPVFUNC_CALL__(FNAME)
#define FORPY_EXPVFUNC_CALL(FID, TMPL, OBJ)       \
  FORPY_GEN_CALLER(FORPY_GEN_FUNCHEAD_##TMPL,           \
                   ,                                                    \
                   FORPY_EXPVFUNC_CALL_(FID##_NAME),                                 \
                   FORPY_GEN_TMPLINST_##TMPL,                                   \
                   ,                                                    \
                   EXPORTER,                                            \
                   ,                                                    \
                   ;)

// Pickle support.
#define FORPY_DEFAULT_PICKLE(CNAME, OBJ) \
OBJ .def("__getstate__", [](const CNAME &p) { \
    std::stringstream ss; \
    { \
      cereal::JSONOutputArchive oarchive(ss); \
      oarchive(p); \
    } \
    return py::make_tuple(ss.str(), ""); \
  }); \
OBJ .def("__setstate__", [](CNAME &p, py::tuple t) { \
    if (t.size() != 2) \
      throw std::runtime_error("Invalid state!"); \
    new (&p) CNAME (); \
    std::stringstream ss(t[0].cast<std::string>()); \
    { \
      cereal::JSONInputArchive iarchive(ss); \
      iarchive(p); \
    } \
  });

#define FORPY_DEFAULT_REPR(INST, CNAME) \
  INST .def("__repr__", [](const CNAME &self) {std::stringstream ss; ss<<self; return ss.str(); });

#endif // BINDINGS_PYTHON_MACROS_H_
