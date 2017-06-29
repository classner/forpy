#include <forpy/leafs/leafs.h>
#include <forpy/data_providers/idataprovider.h>
#include <forpy/util/storage.h>
#include <forpy/util/regression/regression.h>
#include "./conversion.h"
#include "./macros.h"

namespace py = pybind11;

namespace forpy {

  void export_leafs(py::module &m) {
    FORPY_EXPCLASS_EQ(ILeaf, il);
    il.def("is_compatible_with", [](ILeaf &self,
                                    const IDataProvider &dp) {
             return self.is_compatible_with(dp);
           });
    il.def("is_compatible_with", [](ILeaf &self,
                                    const IThresholdOptimizer &to) {
             return self.is_compatible_with(to);
           });
    FORPY_EXPFUNC(il, ILeaf, make_leaf);
    FORPY_EXPFUNC(il, ILeaf, needs_data);
    il.def("get_result_columns",
           &ILeaf::get_result_columns,
           py::arg("n_trees")=1);
    il.def("get_result",
           [](const ILeaf &self,
              const node_id_t &node_id,
              const Data<MatCRef> &data) {
             return self.get_result(node_id, data);
           },
           py::arg("node_id"),
           py::arg("data")=Empty());
    il.def("get_result",
           [](const ILeaf &self,
              const std::vector<Data<Mat>> &leaf_results,
              const Vec<float> &weights) {
             return self.get_result(leaf_results, weights);
           },
           py::arg("leaf_results"),
           py::arg("weights")=Vec<float>());

    FORPY_EXPCLASS_PARENT(ClassificationLeaf, cl, il);
    cl.def(py::init<uint>(),
           py::arg("n_classes")=0);
    FORPY_DEFAULT_REPR(cl, ClassificationLeaf);

    FORPY_EXPCLASS_PARENT(RegressionLeaf, rl, il);
    rl.def(py::init<std::shared_ptr<IRegressor>,
                    uint, size_t, size_t, int, uint>(),
           py::arg("regressor_template")=std::make_shared<LinearRegressor>(),
           py::arg("summary_mode")=0,
           py::arg("regression_input_dim")=0,
           py::arg("selections_to_try")=0,
           py::arg("num_threads")=1,
           py::arg("random_seed")=1);
    FORPY_DEFAULT_REPR(rl, RegressionLeaf);
  };

} // namespace forpy
