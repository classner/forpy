#include <forpy/global.h>
#include <forpy/impurities/impurities.h>
#include "./forpy_exporters.h"

namespace py = pybind11;

namespace forpy {

void export_global(py::module &m) {
  m.attr("__version__") = FORPY_LIB_VERSION();
  m.attr("_OpenCV_available") = FORPY_OPENCV_AVAILABLE();
  m.attr("_entropy_eps") = ENTROPY_EPS;
  m.attr("_sklearn_compatibility_mode") = SKLEARN_COMPAT;
};

}  // namespace forpy
