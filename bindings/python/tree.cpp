#include <forpy/tree.h>
#include "./macros.h"
#include "./conversion.h"

namespace py = pybind11;

namespace forpy {

  void export_tree(py::module &m) {
    FORPY_EXPCLASS_EQ(Tree, t);
    t.def(py::init<uint, uint, uint, std::shared_ptr<IDecider>,
          std::shared_ptr<ILeaf>>());
    t.def(py::init<std::string>());
    t.def_property_readonly("depth", &Tree::get_depth);
    t.def_property_readonly("initialized", &Tree::is_initialized);
    t.def_property_readonly("n_nodes", &Tree::get_n_nodes);
    t.def_property("weight", &Tree::get_weight, &Tree::set_weight);
    t.def_property_readonly("n_samples_stored", &Tree::get_samples_stored);
    FORPY_EXPFUNC(t, Tree, get_input_data_dimensions);
    FORPY_EXPFUNC(t, Tree, get_decider);
    FORPY_EXPFUNC(t, Tree, get_leaf_manager);
    t.def("fit", &Tree::fit,
          py::arg("data").noconvert(),
          py::arg("annotations").noconvert(),
          py::arg("complete_dfs")=true,
          py::call_guard<py::gil_scoped_release>());
    t.def("predict", &Tree::predict,
          py::arg("data").noconvert(),
          py::arg("num_threads")=1,
          py::call_guard<py::gil_scoped_release>());
    FORPY_EXPFUNC(t, Tree, save);
    FORPY_DEFAULT_REPR(t, Tree);
  };

} // namespace forpy
