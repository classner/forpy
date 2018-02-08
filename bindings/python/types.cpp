#include <forpy/types.h>
#include "./conversion.h"
#include "./forpy_exporters.h"

namespace py = pybind11;

namespace forpy {
void export_types(py::module &m) {
  py::enum_<ECompletionLevel>(m, "ECompletionLevel")
      .value("Node", ECompletionLevel::Node)
      .value("Level", ECompletionLevel::Level)
      .value("Complete", ECompletionLevel::Complete);

  py::enum_<ESearchType>(m, "ESearchType")
      .value("DFS", ESearchType::DFS)
      .value("BFS", ESearchType::BFS);

  py::class_<SplitOptRes<float>> sorf(m, "SplitOptRes_f");
  sorf.def_readwrite("split_idx", &SplitOptRes<float>::split_idx)
      .def_readwrite("thresh", &SplitOptRes<float>::thresh)
      .def_readwrite("gain", &SplitOptRes<float>::gain)
      .def_readwrite("valid", &SplitOptRes<float>::valid);
  FORPY_DEFAULT_REPR(sorf, SplitOptRes<float>);
  sorf.def("__eq__",
           [](const SplitOptRes<float> &a, const SplitOptRes<float> &b) {
             return a == b;
           },
           py::is_operator());
  sorf.def("__ne__",
           [](const SplitOptRes<float> &a, const SplitOptRes<float> &b) {
             return !(a == b);
           },
           py::is_operator());

  py::class_<SplitOptRes<double>> sord(m, "SplitOptRes_d");
  sord.def_readwrite("split_idx", &SplitOptRes<double>::split_idx)
      .def_readwrite("thresh", &SplitOptRes<double>::thresh)
      .def_readwrite("gain", &SplitOptRes<double>::gain)
      .def_readwrite("valid", &SplitOptRes<double>::valid);
  FORPY_DEFAULT_REPR(sord, SplitOptRes<double>);
  sord.def("__eq__",
           [](const SplitOptRes<double> &a, const SplitOptRes<double> &b) {
             return a == b;
           },
           py::is_operator());
  sord.def("__ne__",
           [](const SplitOptRes<double> &a, const SplitOptRes<double> &b) {
             return !(a == b);
           },
           py::is_operator());

  py::class_<SplitOptRes<uint>> sorui(m, "SplitOptRes_uint");
  sorui.def_readwrite("split_idx", &SplitOptRes<uint>::split_idx)
      .def_readwrite("thresh", &SplitOptRes<uint>::thresh)
      .def_readwrite("gain", &SplitOptRes<uint>::gain)
      .def_readwrite("valid", &SplitOptRes<uint>::valid);
  FORPY_DEFAULT_REPR(sorui, SplitOptRes<uint>);
  sorui.def("__eq__",
            [](const SplitOptRes<uint> &a, const SplitOptRes<uint> &b) {
              return a == b;
            },
            py::is_operator());
  sorui.def("__ne__",
            [](const SplitOptRes<uint> &a, const SplitOptRes<uint> &b) {
              return !(a == b);
            },
            py::is_operator());

  py::class_<SplitOptRes<uint8_t>> sorui8(m, "SplitOptRes_uint8");
  sorui8.def_readwrite("split_idx", &SplitOptRes<uint8_t>::split_idx)
      .def_readwrite("thresh", &SplitOptRes<uint8_t>::thresh)
      .def_readwrite("gain", &SplitOptRes<uint8_t>::gain)
      .def_readwrite("valid", &SplitOptRes<uint8_t>::valid);
  FORPY_DEFAULT_REPR(sorui8, SplitOptRes<uint8_t>);
  sorui8.def("__eq__",
             [](const SplitOptRes<uint8_t> &a, const SplitOptRes<uint8_t> &b) {
               return a == b;
             },
             py::is_operator());
  sorui8.def("__ne__",
             [](const SplitOptRes<uint8_t> &a, const SplitOptRes<uint8_t> &b) {
               return !(a == b);
             },
             py::is_operator());
};
}  // namespace forpy
