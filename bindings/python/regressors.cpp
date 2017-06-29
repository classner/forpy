#include "./forpy_exporters.h"
#include <forpy/util/regression/regression.h>
#include <forpy/types.h>
#include "./conversion.h"

namespace py = pybind11;

namespace forpy {

  void export_regressors(py::module &m) {
    FORPY_EXPCLASS_EQ(IRegressor, ir);
    ir.def("needs_input_data",
           &IRegressor::needs_input_data);
    ir.def("has_constant_prediction_covariance",
           &IRegressor::has_constant_prediction_covariance);
    ir.def("initialize", [](const std::shared_ptr<IRegressor> &self,
                            const MatCRef<double> &sample_mat,
                            const MatCRef<double> &annotation_mat,
                            regint_t index_interval) {
             auto sp = std::make_shared<const Mat<double>>(sample_mat);
             auto ap = std::make_shared<const Mat<double>>(annotation_mat);
             self -> initialize(sp, ap,
                                index_interval);
           },
           py::arg("sample_mat"),
           py::arg("annotation_mat"),
           py::arg("index_interval")=FORPY_FULL_INTERVAL);
    ir.def_property("index_interval",
                    &IRegressor::get_index_interval,
                    &IRegressor::set_index_interval);
    ir.def("has_solution", &IRegressor::has_solution);
    ir.def("get_residual_error", &IRegressor::get_residual_error);
    ir.def("get_kernel_dimension", &IRegressor::get_kernel_dimension);
    ir.def("predict", [](const IRegressor &self){
        if (self.needs_input_data()) {
          throw Forpy_Exception("This regressor needs input data!");
        }
        Vec<double> tmpin = Vec<double>::Zero(0);
        Vec<double> out = Vec<double>::Zero(self.get_annotation_dimension());
        self.predict(tmpin, out);
        return out;
      });
    ir.def("predict", [](const IRegressor &self,
                         const VecCRef<double> &input) {
             Vec<double> out = Vec<double>::Zero(self.get_annotation_dimension());
             self.predict(input.transpose(), out);
             return out;
           });
    ir.def("predict_covar", [](const IRegressor &self) {
        if (self.needs_input_data()) {
          throw Forpy_Exception("This regressor needs input data!");
        }
        Vec<double> tmpin = Vec<double>::Zero(0);
        Vec<double> out = Vec<double>::Zero(self.get_annotation_dimension());
        Mat<double> outcov = Mat<double>::Zero(self.get_annotation_dimension(),
                                                self.get_annotation_dimension());
        self.predict_covar(tmpin.transpose(), out, outcov);
        return std::make_pair(out, outcov);
      });
    ir.def("predict_covar", [](const IRegressor &self,
                               const VecCRef<double> &input) {
             Vec<double> out = Vec<double>::Zero(self.get_annotation_dimension());
             Mat<double> outcov = Mat<double>::Zero(self.get_annotation_dimension(),
                                                     self.get_annotation_dimension());
             self.predict_covar(input.transpose(), out, outcov);
             return std::make_pair(out, outcov);
      });
    ir.def("get_constant_prediction_covariance",
           [](const IRegressor &self){
             Mat<double> outcov = Mat<double>::Zero(self.get_annotation_dimension(),
                                                     self.get_annotation_dimension());
             self.get_constant_prediction_covariance(outcov);
             return outcov;
           });
    ir.def("freeze", &IRegressor::freeze);
    ir.def_property_readonly("frozen", &IRegressor::get_frozen);
    ir.def("get_input_dimension", &IRegressor::get_input_dimension);
    ir.def("get_annotation_dimension",
           &IRegressor::get_annotation_dimension);
    ir.def_property_readonly("n_samples", &IRegressor::get_n_samples);
    ir.def("__repr__", [](const IRegressor &self) {
        std::string ret = self.get_name() + "[";
        if (! self.has_solution()) {
          ret += "No solution available!";
        } else {
        ret += std::to_string(self.get_input_dimension()) +
          "->" + std::to_string(self.get_annotation_dimension()) +
          ", kerneldim: " + std::to_string(self.get_kernel_dimension()) +
          ", msqe: " + std::to_string(self.get_residual_error());
        }
        ret += " (" + std::to_string(self.get_n_samples()) + " samples)]";
        return ret;
      });

    FORPY_EXPCLASS_PARENT(ConstantRegressor, cr, ir);
    cr.def(py::init<>());
    FORPY_DEFAULT_PICKLE(ConstantRegressor, cr);

    FORPY_EXPCLASS_PARENT(LinearRegressor, lr, ir);
    lr.def(py::init<bool, double>(),
           py::arg("force_numerical_stability")=true,
           py::arg("numerical_zero_threshold")=-1.);
    FORPY_DEFAULT_PICKLE(LinearRegressor, lr);
  }
} // namespace forpy
