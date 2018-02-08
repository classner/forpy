#include <forpy/impurities/impurities.h>
#include <Eigen/Dense>
#include "./conversion.h"
#include "./forpy_exporters.h"

namespace py = pybind11;

namespace forpy {

void export_impurities(py::module &m) {
  FORPY_EXPCLASS_EQ(IEntropyFunction, ieff);
  ieff.def("__call__",
           [](const IEntropyFunction &ef, const std::vector<float> &a) {
             return ef(a);
           },
           py::is_operator());

  FORPY_EXPCLASS_PARENT(ShannonEntropy, se, ieff);
  se.def(py::init<>());
  FORPY_DEFAULT_REPR(se, ShannonEntropy);

  FORPY_EXPCLASS_PARENT(ClassificationError, ce, ieff);
  ce.def(py::init<>());
  FORPY_DEFAULT_REPR(ce, ClassificationError);

  FORPY_EXPCLASS_PARENT(InducedEntropy, ie, ieff);
  ie.def(py::init<float>());
  ie.def_property_readonly("get_p", &InducedEntropy::get_p);
  FORPY_DEFAULT_REPR(ie, InducedEntropy);

  FORPY_EXPCLASS_PARENT(TsallisEntropy, te, ieff);
  te.def(py::init<float>());
  te.def_property_readonly("get_q", &TsallisEntropy::get_q);
  FORPY_DEFAULT_REPR(te, TsallisEntropy);

  FORPY_EXPCLASS_PARENT(RenyiEntropy, re, ieff);
  re.def(py::init<float>());
  re.def_property_readonly("get_alpha", &RenyiEntropy::get_alpha);
  FORPY_DEFAULT_REPR(re, RenyiEntropy);
};

}  // namespace forpy
