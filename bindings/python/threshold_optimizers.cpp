#include <forpy/data_providers/idataprovider.h>
#include <forpy/impurities/impurities.h>
#include <forpy/threshold_optimizers/threshold_optimizers.h>
#include <forpy/types.h>
#include <forpy/util/desk.h>
#include "./conversion.h"
#include "./forpy_exporters.h"

namespace py = pybind11;

namespace forpy {

void export_threshold_optimizers(py::module &m) {
  FORPY_EXPCLASS_EQ(IThreshOpt, ito)
  FORPY_EXPFUNC(ito, IThreshOpt, get_gain_threshold_for)
  FORPY_EXPFUNC(ito, IThreshOpt, supports_weights)
  FORPY_EXPFUNC(ito, IThreshOpt, check_annotations)
  ito.def("full_entropy",
          [](const std::shared_ptr<IThreshOpt> &self,
             const std::shared_ptr<IDataProvider> &dprov,
             std::vector<id_t> sample_ids) {
            if (sample_ids.size() == 0)
              sample_ids = dprov->get_initial_sample_list();
            Desk desk(0);
            desk.setup(nullptr, nullptr, nullptr);
            desk.d.n_samples = sample_ids.size();
            desk.d.input_dim = dprov->get_feat_vec_dim();
            desk.d.annot_dim = dprov->get_annot_vec_dim();
            desk.d.elem_id_p = &sample_ids[0];
            desk.d.node_id = 0;
            desk.d.start_id = 0;
            desk.d.end_id = sample_ids.size();
            self->full_entropy(*dprov, &desk);
            return desk.d.fullentropy;
          },
          py::arg("dprov"), py::arg("sample_ids") = std::vector<id_t>());
  ito.def(
      "optimize",
      [](const std::shared_ptr<IThreshOpt> &self,
         const std::shared_ptr<IDataProvider> &dprov, const size_t &feature_id,
         std::vector<id_t> sample_ids, const size_t &min_samples_at_leaf) {
        if (sample_ids.size() == 0)
          sample_ids = dprov->get_initial_sample_list();
        Desk desk(0);
        desk.setup(nullptr, nullptr, nullptr);
        desk.d.n_samples = sample_ids.size();
        desk.d.input_dim = dprov->get_feat_vec_dim();
        desk.d.annot_dim = dprov->get_annot_vec_dim();
        desk.d.min_samples_at_leaf = min_samples_at_leaf;
        desk.d.elem_id_p = &sample_ids[0];
        desk.d.node_id = 0;
        desk.d.start_id = 0;
        desk.d.end_id = sample_ids.size();
        self->full_entropy(*dprov, &desk);
        desk.d.best_res_v = SplitOptRes<float>{
            0, std::numeric_limits<float>::lowest(), 0.f, false};
        desk.d.opt_res_v.match([](auto &opt_res) {
          opt_res.gain = 0.f;
          opt_res.valid = false;
        });
        desk.d.need_sort = false;
        desk.d.presorted = false;
        dprov->get_feature(feature_id).match([&](const auto &feat_dta) {
          desk.d.full_feat_p_v = feat_dta.data();
        });
        self->optimize(&desk);
        return desk.d.opt_res_v;
      },
      py::arg("dprov"), py::arg("feature_id"),
      py::arg("sample_ids") = std::vector<id_t>(),
      py::arg("min_samples_at_leaf") = 1);
  FORPY_EXPCLASS_PARENT(RegressionOpt, ro, ito);
  ro.def(py::init<size_t, float>(), py::arg("n_thresholds") = 0,
         py::arg("gain_threshold") = 1E-7f);
  FORPY_DEFAULT_REPR(ro, RegressionOpt);

  FORPY_EXPCLASS_PARENT(ClassificationOpt, co, ito);
  co.def(py::init<size_t, float, std::shared_ptr<IEntropyFunction>>(),
         py::arg("n_thresholds") = 0, py::arg("gain_threshold") = 1E-7f,
         py::arg("entropy_function") = std::make_shared<InducedEntropy>(2));
  co.def_property_readonly("n_classes", &ClassificationOpt::get_n_classes);
  co.def_property_readonly("class_translation",
                           [](const std::shared_ptr<ClassificationOpt> &self) {
                             return *(self->get_class_translation());
                           });
  co.def_property_readonly("true_max_class",
                           &ClassificationOpt::get_true_max_class);
  FORPY_DEFAULT_REPR(co, ClassificationOpt);

  FORPY_EXPCLASS_PARENT(FastClassOpt, fco, co);
  fco.def(py::init<size_t, float>(), py::arg("n_thresholds") = 0,
          py::arg("gain_threshold") = 1E-7f);
  FORPY_DEFAULT_REPR(fco, FastClassOpt);
};
}  // namespace forpy
