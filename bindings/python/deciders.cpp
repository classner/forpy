#include <forpy/deciders/deciders.h>
#include <forpy/threshold_optimizers/threshold_optimizers.h>
#include <forpy/util/storage.h>
#include "./conversion.h"
#include "./macros.h"

namespace py = pybind11;

namespace forpy {

void export_deciders(py::module &m) {
  FORPY_EXPCLASS_EQ(IDecider, id);
  FORPY_EXPFUNC(id, IDecider, supports_weights);
  FORPY_EXPFUNC(id, IDecider, get_data_dim);
  FORPY_EXPFUNC(id, IDecider, set_data_dim);
  id.def("decide", [](const IDecider &self, const uint &node_id,
                      const Data<MatCRef> &data_v) {
    return self.decide(node_id, data_v, nullptr);
  });

  FORPY_EXPCLASS_PARENT(FastDecider, fd, id);
  fd.def(py::init<std::shared_ptr<IThreshOpt>, size_t, bool>(),
         py::arg("threshold_optimizer") = nullptr,
         py::arg("n_valid_features_to_use") = 0,
         py::arg("autoscale_valid_features") = false);
  FORPY_EXPFUNC(fd, FastDecider, get_maps);
  FORPY_DEFAULT_REPR(fd, FastDecider);
};

}  // namespace forpy
