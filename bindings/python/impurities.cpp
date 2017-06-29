#include "./forpy_exporters.h"
#include <Eigen/Dense>
#include <forpy/impurities/impurities.h>
#include "./conversion.h"

namespace py = pybind11;

namespace forpy {

void export_impurities_impl(py::module &m, const std::string &suff) {
  py::class_<IEntropyFunction,
             std::shared_ptr<IEntropyFunction>> ieff(
     m, ("IEntropyFunction" + suff).c_str());
  ieff.def("__eq__", [](const IEntropyFunction &a,
                        const IEntropyFunction &b) { return a == b; },
           py::is_operator());
  ieff.def("__ne__", [](const IEntropyFunction &a,
                        const IEntropyFunction &b) { return !(a == b); },
           py::is_operator());
  ieff.def("__call__", [](const IEntropyFunction &ef,
                          const std::vector<float> &a) { return ef(a); },
           py::is_operator());
  ieff.def("differential_normal",
           [](const IEntropyFunction &ef,
              const MatCRef<float>& in) {
             return ef.differential_normal(in);
           });
    
  py::class_<ShannonEntropy, std::shared_ptr<ShannonEntropy>> se(
      m,
      ("ShannonEntropy" + suff).c_str(),
      ieff);
  se.def(py::init<>());

  py::class_<ClassificationError, std::shared_ptr<ClassificationError>>
             ce(m,
                ("ClassificationError" + suff).c_str(),
                ieff);
  ce.def(py::init<>());

  py::class_<InducedEntropy, std::shared_ptr<InducedEntropy>>
             ie(m,
                ("InducedEntropy" + suff).c_str(),
                ieff);
  ie.def(py::init<float>());
  ie.def_property_readonly("get_p", &InducedEntropy::get_p);

  py::class_<TsallisEntropy, std::shared_ptr<TsallisEntropy>>
             te(m,
                ("TsallisEntropy" + suff).c_str(),
                ieff);
  te.def(py::init<float>());
  te.def_property_readonly("get_q", &TsallisEntropy::get_q);

  py::class_<RenyiEntropy, std::shared_ptr<RenyiEntropy>>
             re(m,
                ("RenyiEntropy" + suff).c_str(),
                ieff);
  re.def(py::init<float>());
  re.def_property_readonly("get_alpha", &RenyiEntropy::get_alpha); 
};
}  // namespace forpy


void forpy::export_impurities(py::module &m) {
  export_impurities_impl(m, "");
};
