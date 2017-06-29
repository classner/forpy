#include "./forpy_exporters.h"
#include <forpy/global.h>
#include <forpy/impurities/impurities.h>


namespace py = pybind11;

namespace forpy {

  void export_global(py::module &m) {
    m.attr("__version__") = FORPY_LIB_VERSION();
    m.attr("_OpenCV_available") = FORPY_OPENCV_AVAILABLE();
    m.attr("_diffentropy_eps") = DIFFENTROPY_EPS;
  };

} // namespace forpy
