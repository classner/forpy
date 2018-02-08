#include <forpy/deciders/fastdecider.h>
#include <forpy/leafs/classificationleaf.h>
#include <forpy/leafs/regressionleaf.h>
#include <forpy/threshold_optimizers/fastclassopt.h>
#include <forpy/threshold_optimizers/regression_opt.h>
#include <forpy/tree.h>
#include "./conversion.h"
#include "./macros.h"

namespace py = pybind11;

namespace forpy {

void export_tree(py::module &m) {
  FORPY_EXPCLASS_EQ(Tree, t);
  t.def(py::init<uint, uint, uint, std::shared_ptr<IDecider>,
                 std::shared_ptr<ILeaf>, uint>(),
        py::arg("max_depth") = std::numeric_limits<uint>::max(),
        py::arg("min_samples_leaf") = 1, py::arg("min_samples_node") = 2,
        py::arg("decider_template") = nullptr,
        py::arg("leaf_template") = nullptr, py::arg("random_seed") = 1);
  t.def(py::init<std::string>(), py::arg("filename"));
  FORPY_DEFAULT_PICKLE(Tree, t);
  t.def_property_readonly("depth", &Tree::get_depth);
  t.def_property_readonly("initialized", &Tree::is_initialized);
  t.def_property_readonly("n_nodes", &Tree::get_n_nodes);
  t.def_property("weight", &Tree::get_weight, &Tree::set_weight);
  t.def_property_readonly("n_samples_stored", &Tree::get_samples_stored);
  FORPY_EXPFUNC(t, Tree, get_input_data_dimensions);
  FORPY_EXPFUNC(t, Tree, get_decider);
  FORPY_EXPFUNC(t, Tree, get_leaf_manager);
  t.def_property_readonly("tree", &Tree::get_tree);
  t.def("fit", &Tree::fit, py::arg("data").noconvert(),
        py::arg("annotations").noconvert(), py::arg("n_threads") = 0,
        py::arg("complete_dfs") = true,
        py::arg("weights") = std::vector<float>(),
        py::call_guard<py::gil_scoped_release>(),
        py::return_value_policy::reference_internal);
  t.def("fit_dprov", &Tree::fit_dprov, py::arg("data_provider"),
        py::arg("complete_dfs") = true,
        py::call_guard<py::gil_scoped_release>(),
        py::return_value_policy::reference_internal);
  t.def("predict", &Tree::predict, py::arg("data").noconvert(),
        py::arg("num_threads") = 1,
        py::arg("use_fast_prediction_if_available") = true,
        py::arg("predict_proba") = false, py::arg("for_forest") = false,
        py::call_guard<py::gil_scoped_release>());
  t.def("predict_proba", &Tree::predict_proba, py::arg("data").noconvert(),
        py::arg("num_threads") = 1,
        py::arg("use_fast_prediction_if_available") = true,
        py::call_guard<py::gil_scoped_release>());
  FORPY_EXPFUNC(t, Tree, enable_fast_prediction);
  FORPY_EXPFUNC(t, Tree, disable_fast_prediction);
  FORPY_EXPFUNC(t, Tree, save);
  FORPY_DEFAULT_REPR(t, Tree);

  FORPY_EXPCLASS_PARENT(ClassificationTree, ct, t);
  ct.def(py::init<uint, uint, uint, uint, bool, uint, size_t, float>(),
         py::arg("max_depth") = std::numeric_limits<uint>::max(),
         py::arg("min_samples_at_leaf") = 1, py::arg("min_samples_at_node") = 2,
         py::arg("n_valid_features_to_use") = 0,
         py::arg("autoscale_valid_features") = false,
         py::arg("random_seed") = 1, py::arg("n_thresholds") = 0,
         py::arg("gain_threshold") = 1E-7f);
  ct.def(py::init<std::string>(), py::arg("filename"));
  FORPY_DEFAULT_PICKLE(ClassificationTree, ct);
  ct.def("get_params", &ClassificationTree::get_params,
         py::arg("deep") = false);
  ct.def("set_params", [](const std::shared_ptr<ClassificationTree> &self,
                          py::kwargs kwargs) {
    std::unordered_map<std::string, mu::variant<uint, size_t, float, bool>>
        params;
    if (kwargs) {
      for (auto item : kwargs) {
        auto key = std::string(py::str(item.first));
        if (key == "gain_threshold")
          params[key] = item.second.cast<py::float_>();
        else if (key == "autoscale_valid_features")
          params[key] = static_cast<bool>(item.second.cast<py::bool_>());
        else
          params[key] = static_cast<uint>(item.second.cast<py::int_>());
      }
    }
    return self->set_params(params);
  });
  FORPY_DEFAULT_REPR(ct, ClassificationTree);

  FORPY_EXPCLASS_PARENT(RegressionTree, rt, t);
  rt.def(
      py::init<uint, uint, uint, uint, bool, uint, size_t, float, bool, bool>(),
      py::arg("max_depth") = std::numeric_limits<uint>::max(),
      py::arg("min_samples_at_leaf") = 1, py::arg("min_samples_at_node") = 2,
      py::arg("n_valid_features_to_use") = 0,
      py::arg("autoscale_valid_features") = false, py::arg("random_seed") = 1,
      py::arg("n_thresholds") = 0, py::arg("gain_threshold") = 1E-7f,
      py::arg("store_variance") = false, py::arg("summarize") = false);
  rt.def(py::init<std::string>(), py::arg("filename"));
  FORPY_DEFAULT_PICKLE(RegressionTree, rt);
  rt.def("get_params", &RegressionTree::get_params, py::arg("deep") = false);
  rt.def("set_params", [](const std::shared_ptr<RegressionTree> &self,
                          py::kwargs kwargs) {
    std::unordered_map<std::string, mu::variant<uint, size_t, float, bool>>
        params;
    if (kwargs) {
      for (auto item : kwargs) {
        auto key = std::string(py::str(item.first));
        if (key == "gain_threshold")
          params[key] = item.second.cast<py::float_>();
        else if (key == "store_variance" || key == "summarize" ||
                 key == "autoscale_valid_features")
          params[key] = static_cast<bool>(item.second.cast<py::bool_>());
        else
          params[key] = static_cast<uint>(item.second.cast<py::int_>());
      }
    }
    return self->set_params(params);
  });
  FORPY_DEFAULT_REPR(rt, RegressionTree);
};

}  // namespace forpy
