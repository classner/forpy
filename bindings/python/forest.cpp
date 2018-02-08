#include <forpy/forest.h>
#include <forpy/leafs/regressionleaf.h>
#include <forpy/threshold_optimizers/regression_opt.h>
#include "./conversion.h"
#include "./macros.h"

namespace py = pybind11;

namespace forpy {

void export_forest(py::module &m) {
  FORPY_EXPCLASS_EQ(Forest, f);
  f.def(py::init<uint, uint, uint, uint, std::shared_ptr<IDecider>,
                 std::shared_ptr<ILeaf>>(),
        py::arg("n_trees") = 10,
        py::arg("max_depth") = std::numeric_limits<uint>::max(),
        py::arg("min_samples_leaf") = 1, py::arg("min_samples_node") = 2,
        py::arg("decider_template") = std::shared_ptr<IDecider>(),
        py::arg("leaf_template") = std::shared_ptr<ILeaf>());
  f.def(py::init<std::string>());
  f.def(py::init<std::vector<std::shared_ptr<Tree>> &>());
  FORPY_DEFAULT_PICKLE(Forest, f);
  f.def_property_readonly("depths", &Forest::get_depths);
  f.def_property("tree_weights", &Forest::get_tree_weights,
                 &Forest::set_tree_weights);
  f.def_property_readonly("trees", &Forest::get_trees);
  FORPY_EXPFUNC(f, Forest, get_input_data_dimensions);
  FORPY_EXPFUNC(f, Forest, get_decider);
  FORPY_EXPFUNC(f, Forest, get_leaf_manager);
  f.def("fit", &Forest::fit, py::arg("data").noconvert(),
        py::arg("annotations").noconvert(), py::arg("n_threads") = 1,
        py::arg("bootstrap") = true, py::arg("weights") = std::vector<float>(),
        py::call_guard<py::gil_scoped_release>(),
        py::return_value_policy::reference_internal);
  f.def("fit_dprov", &Forest::fit_dprov, py::arg("data_provider"),
        py::arg("bootstrap") = true, py::call_guard<py::gil_scoped_release>(),
        py::return_value_policy::reference_internal);
  f.def("predict", &Forest::predict, py::arg("data").noconvert(),
        py::arg("num_threads") = 1,
        py::arg("use_fast_prediction_if_available") = true,
        py::arg("predict_proba") = false,
        py::call_guard<py::gil_scoped_release>());
  f.def("predict_proba", &Forest::predict_proba, py::arg("data").noconvert(),
        py::arg("num_threads") = 1,
        py::arg("use_fast_prediction_if_available") = true,
        py::call_guard<py::gil_scoped_release>());
  f.def("enable_fast_prediction", &Forest::enable_fast_prediction);
  f.def("disable_fast_prediction", &Forest::disable_fast_prediction);
  FORPY_EXPFUNC(f, Forest, save);
  FORPY_DEFAULT_REPR(f, Forest);

  FORPY_EXPCLASS_PARENT(ClassificationForest, ct, f);
  ct.def(py::init<size_t, uint, uint, uint, uint, bool, uint, size_t, float>(),
         py::arg("n_trees") = 10,
         py::arg("max_depth") = std::numeric_limits<uint>::max(),
         py::arg("min_samples_at_leaf") = 1, py::arg("min_samples_at_node") = 2,
         py::arg("n_valid_features_to_use") = 0,
         py::arg("autoscale_valid_features") = true, py::arg("random_seed") = 1,
         py::arg("n_thresholds") = 0, py::arg("gain_threshold") = 1E-7f);
  ct.def(py::init<std::string>(), py::arg("filename"));
  FORPY_DEFAULT_PICKLE(ClassificationForest, ct);
  ct.def("get_params", &ClassificationForest::get_params,
         py::arg("deep") = false);
  ct.def("set_params", [](const std::shared_ptr<ClassificationForest> &self,
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
  FORPY_DEFAULT_REPR(ct, ClassificationForest);

  FORPY_EXPCLASS_PARENT(RegressionForest, rt, f);
  rt.def(py::init<size_t, uint, uint, uint, uint, bool, uint, size_t, float,
                  bool, bool>(),
         py::arg("n_trees") = 10,
         py::arg("max_depth") = std::numeric_limits<uint>::max(),
         py::arg("min_samples_at_leaf") = 1, py::arg("min_samples_at_node") = 2,
         py::arg("n_valid_features_to_use") = 0,
         py::arg("autoscale_valid_features") = false,
         py::arg("random_seed") = 1, py::arg("n_thresholds") = 0,
         py::arg("gain_threshold") = 1E-7f, py::arg("store_variance") = false,
         py::arg("summarize") = false);
  rt.def(py::init<std::string>(), py::arg("filename"));
  FORPY_DEFAULT_PICKLE(RegressionForest, rt);
  rt.def("get_params", &RegressionForest::get_params, py::arg("deep") = false);
  rt.def("set_params", [](const std::shared_ptr<RegressionForest> &self,
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
  FORPY_DEFAULT_REPR(rt, RegressionForest);
};

}  // namespace forpy
