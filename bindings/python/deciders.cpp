#include <forpy/deciders/deciders.h>
#include <forpy/threshold_optimizers/threshold_optimizers.h>
#include <forpy/features/features.h>
#include <forpy/util/storage.h>
#include "./macros.h"
#include "./conversion.h"

namespace py = pybind11;

namespace forpy {

  void export_deciders(py::module &m) {
    FORPY_EXPCLASS_EQ(IDecider, id);
    FORPY_EXPFUNC(id, IDecider, supports_weights);
    FORPY_EXPFUNC(id, IDecider, get_data_dim);
    id.def("decide", [](const IDecider &self,
                        const uint &node_id,
                        const Data<MatCRef> &data_v) {
             return self.decide(node_id, data_v, nullptr);
           });
    FORPY_EXPFUNC(id, IDecider, make_node);


    FORPY_EXPCLASS_PARENT(ThresholdDecider, td, id);
    td.def(py::init<
             std::shared_ptr<IThresholdOptimizer>,
             size_t,
             std::shared_ptr<ISurfaceCalculator>,
             int,
             uint>(),
           py::arg("threshold_optimizer"),
           py::arg("n_valid_features_to_use"),
           py::arg("surface_calculator")=std::make_shared<AlignedSurfaceCalculator>(),
           py::arg("num_threads")=1,
           py::arg("random_seed")=1);
    FORPY_DEFAULT_REPR(td, ThresholdDecider);
  };

} // namespace forpy
