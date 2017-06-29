#include "./forpy_exporters.h"
#include <forpy/util/sampling.h>
#include "./conversion.h"

namespace py = pybind11;

namespace forpy {

  void export_util(py::module &m) {
    py::class_<std::mt19937, std::shared_ptr<std::mt19937>> sre(m, "RandomEngine");
    sre.def(py::init<unsigned int>());

    py::class_<SamplingWithoutReplacement<size_t>, std::shared_ptr<SamplingWithoutReplacement<size_t>>>
      swr(m, "SamplingWithoutReplacement");
    swr.def(py::init<size_t, size_t, std::shared_ptr<std::mt19937>>());
    FORPY_EXPFUNC(swr, SamplingWithoutReplacement<size_t>, sample_available);
    FORPY_EXPFUNC(swr, SamplingWithoutReplacement<size_t>, get_next);

    FORPY_EXPCLASS(Empty, ety);
    ety.def(py::init<>());
  }

} // namespace forpy
