#include "./forpy_exporters.h"
#include <forpy/gains/gains.h>
#include "./conversion.h"

namespace py = pybind11;

namespace forpy {
void export_gains_impl(py::module &m, const std::string &suff) {
  py::class_<IGainCalculator, std::shared_ptr<IGainCalculator>>
    igc(m, ("IGainCalculator" + suff).c_str());
  igc.def("__eq__", [](const IGainCalculator &a,
                       const IGainCalculator &b) { return a == b; },
          py::is_operator());
  igc.def("__ne__", [](const IGainCalculator &a,
                       const IGainCalculator &b) { return !(a == b); },
          py::is_operator());
  igc.def("__call__", [](IGainCalculator &gc,
                         const std::vector<float> &left,
                         const std::vector<float> &right) {
            return gc(left, right); },
            py::is_operator());
  igc.def("approx", [](IGainCalculator &gc,
                       const std::vector<float> &left,
                       const std::vector<float> &right) {
            return gc.approx(left, right); });
    
  py::class_<EntropyGain, std::shared_ptr<EntropyGain>>
    pg(m,
       ("EntropyGain" + suff).c_str(),
       igc);
  pg.def(py::init<std::shared_ptr<IEntropyFunction>>());
  pg.def_property_readonly("entropy_function",
                           &EntropyGain::getEntropy_function);
};
}  // namespace forpy

void forpy::export_gains(py::module &m) {
  export_gains_impl(m, "");
};
