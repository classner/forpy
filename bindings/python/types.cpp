#include "./forpy_exporters.h"
#include <forpy/types.h>
#include "./conversion.h"

namespace py = pybind11;

namespace forpy {
  void export_types(py::module &m) {
    py::enum_<ECompletionLevel>(m, "ECompletionLevel")
      .value("Node", ECompletionLevel::Node)
      .value("Level", ECompletionLevel::Level)
      .value("Complete", ECompletionLevel::Complete);

    py::enum_<EThresholdSelection>(m, "EThresholdSelection")
      .value("LessOnly", EThresholdSelection::LessOnly)
      .value("GreaterOnly", EThresholdSelection::GreaterOnly)
      .value("Both", EThresholdSelection::Both);

    py::enum_<ESampleAction>(m, "ESampleAction")
      .value("AddToTraining", ESampleAction::AddToTraining)
      .value("RemoveFromTraining", ESampleAction::RemoveFromTraining)
      .value("AddToValidation", ESampleAction::AddToValidation)
      .value("RemoveFromValidation", ESampleAction::RemoveFromValidation);

    py::enum_<ESearchType>(m, "ESearchType")
      .value("DFS", ESearchType::DFS)
      .value("BFS", ESearchType::BFS);
  };
}
