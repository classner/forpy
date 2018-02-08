#include <forpy/data_providers/data_providers.h>
#include <forpy/util/storage.h>
#include "./conversion.h"
#include "./macros.h"

namespace py = pybind11;

namespace forpy {

void export_data_providers(py::module &m) {
  FORPY_EXPCLASS_EQ(IDataProvider, idp);
  FORPY_EXPFUNC(idp, IDataProvider, get_initial_sample_list);
  idp.def("get_feature", [](IDataProvider &self, const size_t &idx) {
    Data<Vec> ret;
    const auto &res = self.get_feature(idx);
    res.match(
        [&](const auto &data) {
          typedef typename get_core<decltype(data.data())>::type IT;
          ret.set<Vec<IT>>(data);
        },
        [](const Empty &) { throw EmptyException(); });
    return ret;
  });
  idp.def("get_annotations", [](IDataProvider &self) {
    Data<Mat> ret;
    Data<MatCRef> res = self.get_annotations();
    res.match(
        [&](const auto &data) {
          typedef typename get_core<decltype(data.data())>::type AT;
          ret.set<Mat<AT>>(data);
        },
        [](const Empty &) { throw EmptyException(); });
    return ret;
  });
  idp.def("get_weights", &IDataProvider::get_weights);
  idp.def_property_readonly("feat_vec_dim", &IDataProvider::get_feat_vec_dim);
  idp.def_property_readonly("annot_vec_dim", &IDataProvider::get_annot_vec_dim);
  idp.def(
      "create_tree_providers",
      [](std::shared_ptr<IDataProvider> &self,
         const std::vector<std::pair<std::vector<size_t>, std::vector<float>>>
             &usage_map) {
        usage_map_t us_converted;
        for (const auto &tree_vec : usage_map)
          us_converted.push_back(
              {std::make_shared<std::vector<size_t>>(tree_vec.first),
               std::make_shared<std::vector<float>>(tree_vec.second)});
        return self->create_tree_providers(us_converted);
      });

  FORPY_EXPCLASS_PARENT(FastDProv, fdp, idp);
  fdp.def("__init__",
          [](FastDProv &self, Data<MatCRef> &data, Data<MatCRef> &annot,
             std::vector<float> &weights) {
            new (&self) FastDProv(
                data, annot, std::make_shared<std::vector<float>>(weights));
          },
          py::arg("data").noconvert(), py::arg("annotations").noconvert(),
          py::arg("weights") = std::vector<float>(), py::keep_alive<1, 2>(),
          py::keep_alive<1, 3>());
  FORPY_DEFAULT_REPR(fdp, FastDProv);
};

}  // namespace forpy
