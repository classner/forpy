#pragma once
#ifndef BINDINGS_PYTHON_MACROS_H_
#define BINDINGS_PYTHON_MACROS_H_

// Python export macros.
#define FORPY_EXPCLASS_EQ(CLS, NAME)                                       \
  py::class_<CLS, std::shared_ptr<CLS>> NAME(m, #CLS);                     \
  NAME.def("__eq__", [](const CLS &a, const CLS &b) { return a == b; },    \
           py::is_operator());                                             \
  NAME.def("__ne__", [](const CLS &a, const CLS &b) { return !(a == b); }, \
           py::is_operator());

#define FORPY_EXPCLASS(CLS, NAME) \
  py::class_<CLS, std::shared_ptr<CLS>> NAME(m, #CLS);

#define FORPY_EXPCLASS_PARENT(CLS, NAME, PARENT) \
  py::class_<CLS, std::shared_ptr<CLS>> NAME(m, #CLS, PARENT);

#define FORPY_EXPFUNC(OBJ, CLS, FUNC) OBJ.def(#FUNC, &CLS::FUNC);

// Pickle support.
#define FORPY_DEFAULT_PICKLE(CNAME, OBJ)                           \
  OBJ.def("__getstate__", [](const CNAME &p) {                     \
    std::stringstream ss;                                          \
    {                                                              \
      cereal::JSONOutputArchive oarchive(ss);                      \
      oarchive(p);                                                 \
    }                                                              \
    return py::make_tuple(ss.str(), "");                           \
  });                                                              \
  OBJ.def("__setstate__", [](CNAME &p, py::tuple t) {              \
    if (t.size() != 2) throw std::runtime_error("Invalid state!"); \
    new (&p) CNAME();                                              \
    std::stringstream ss(t[0].cast<std::string>());                \
    {                                                              \
      cereal::JSONInputArchive iarchive(ss);                       \
      iarchive(p);                                                 \
    }                                                              \
  });

#define FORPY_DEFAULT_REPR(INST, CNAME)        \
  INST.def("__repr__", [](const CNAME &self) { \
    std::stringstream ss;                      \
    ss << self;                                \
    return ss.str();                           \
  });

#endif  // BINDINGS_PYTHON_MACROS_H_
