#include <forpy/data_providers/idataprovider.h>
#include <forpy/leafs/leafs.h>
#include "./conversion.h"
#include "./macros.h"

namespace py = pybind11;

namespace forpy {

void export_leafs(py::module &m) {
  FORPY_EXPCLASS_EQ(ILeaf, il);
  il.def("is_compatible_with", [](ILeaf &self, const IDataProvider &dp) {
    return self.is_compatible_with(dp);
  });
  il.def("is_compatible_with", [](ILeaf &self, const IThreshOpt &to) {
    return self.is_compatible_with(to);
  });
  il.def("get_result_columns", &ILeaf::get_result_columns,
         py::arg("n_trees") = 1, py::arg("predict_proba") = false,
         py::arg("for_forest") = false);
  il.def("get_result",
         [](const ILeaf &self, const id_t &node_id, const bool &predict_proba,
            const bool &for_forest) {
           return self.get_result(node_id, predict_proba, for_forest);
         },
         py::arg("node_id"), py::arg("predict_proba") = false,
         py::arg("for_forest") = false);
  il.def("get_result",
         [](const ILeaf &self, const std::vector<Data<Mat>> &leaf_results,
            const Vec<float> &weights, const bool &predict_proba) {
           return self.get_result(leaf_results, weights, predict_proba);
         },
         py::arg("leaf_results"), py::arg("weights") = Vec<float>(),
         py::arg("predict_proba") = false);

  FORPY_EXPCLASS_PARENT(ClassificationLeaf, cl, il);
  cl.def(py::init<uint>(), py::arg("n_classes") = 0);
  FORPY_DEFAULT_REPR(cl, ClassificationLeaf);
  FORPY_EXPCLASS_PARENT(RegressionLeaf, regl, il);
  regl.def(py::init<bool, bool>(), py::arg("store_variance") = false,
           py::arg("summarize") = false);
  FORPY_DEFAULT_REPR(regl, RegressionLeaf);
};

}  // namespace forpy
