#include "./forpy_exporters.h"
#include <forpy/gains/gains.h>
#include <forpy/threshold_optimizers/threshold_optimizers.h>
#include <forpy/util/regression/regression.h>
#include <forpy/impurities/impurities.h>
#include <forpy/types.h>
#include "./conversion.h"

namespace py = pybind11;

namespace forpy {
  #define EXPORTER(...) ito
  FORPY_EXPVFUNC(FORPY_ITHRESHOPT_EARLYSTOP, AT, IThresholdOptimizer, EXPORTER)
  FORPY_EXPVFUNC_DEF(FORPY_ITHRESHOPT_OPT, ITFTAT, IThresholdOptimizer, EXPORTER, \
                     (py::arg("selected_data").noconvert(),             \
                      py::arg("feature_values").noconvert(),            \
                      py::arg("annotations").noconvert(),               \
                     py::arg("node_id")=0,                              \
                     py::arg("min_samples_at_leaf")=0,                  \
                     py::arg("weights")=FORPY_ZERO_MATR,                \
                     py::arg("suggestion_index")=0))

  void export_threshold_optimizers(py::module &m) {
    FORPY_EXPCLASS_EQ(IThresholdOptimizer, ito)
    FORPY_EXPVFUNC_CALL(FORPY_ITHRESHOPT_EARLYSTOP, AT, EXPORTER)
    FORPY_EXPVFUNC_CALL(FORPY_ITHRESHOPT_OPT, ITFTAT, EXPORTER)
    FORPY_EXPFUNC(ito, IThresholdOptimizer, prepare_for_optimizing)
    FORPY_EXPFUNC(ito, IThresholdOptimizer, get_gain_threshold_for)
    FORPY_EXPFUNC(ito, IThresholdOptimizer, supports_weights)
    FORPY_EXPFUNC(ito, IThresholdOptimizer, check_annotations)

    FORPY_EXPCLASS_PARENT(ClassificationThresholdOptimizer, cto, ito);
    cto.def(py::init<size_t, std::shared_ptr<IGainCalculator>, float, bool>(),
            py::arg("n_classes")=0,
            py::arg("gain_calculator") = std::make_shared<EntropyGain>(
                std::make_shared<ShannonEntropy>()),
            py::arg("gain_threshold") = 1E-7f,
            py::arg("use_fast_search_approximation") = true);
    cto.def_property_readonly("use_fast_search_approximation",
                              &ClassificationThresholdOptimizer::
                              getUse_fast_search_approximation);
    cto.def_property_readonly("n_classes",
                              &ClassificationThresholdOptimizer::
                              getN_classes);
    cto.def_property_readonly("gain_threshold",
                              &ClassificationThresholdOptimizer::
                              getGain_threshold);
    cto.def_property_readonly("gain_calculator",
                              &ClassificationThresholdOptimizer::
                              getGain_calculator);

  FORPY_EXPCLASS_PARENT(RegressionThresholdOptimizer, rto, ito);
  rto.def(py::init<size_t,
                   std::shared_ptr<IRegressor>,
                   std::shared_ptr<IEntropyFunction>,
                   float,
                   unsigned int>(),
          py::arg("n_thresholds")=0,
          py::arg("regressor_template")=std::make_shared<ConstantRegressor>(),
          py::arg("entropy_function")=std::make_shared<ShannonEntropy>(),
          py::arg("gain_threshold")=1E-7f,
          py::arg("random_seed")=1);
  };
}
