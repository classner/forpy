#include <forpy/data_providers/data_providers.h>
#include <forpy/util/storage.h>
#include "./macros.h"
#include "./conversion.h"

namespace py = pybind11;

namespace forpy {
#define EXP_SAMPLE_DOC 
#define EXP_SAMPLE_RET(IT, AT) void
#define EXP_SAMPLE_NAME exp_sample
#define EXP_SAMPLE_PARAMNAMES it, at, m
#define EXP_SAMPLE_PARAMCALL(IT, AT) IT(), AT(), m
#define EXP_SAMPLE_PARAMTYPESNNAMES(IT, AT) IT it, AT at, py::module &m
#define EXP_SAMPLE_PARAMTYPESNNAMESNDEF(IT, AT) IT it, AT at, py::module &m
#define EXP_SAMPLE_MOD
  FORPY_DECL(EXP_SAMPLE, ITAT, , ;)
  FORPY_DECL_IMPL(EXP_SAMPLE, ITAT, ;);

  void export_data_providers(py::module &m) {
    FORPY_CALL(EXP_SAMPLE, ITAT);
    FORPY_EXPCLASS_EQ(IDataProvider, idp);
    FORPY_EXPFUNC(idp, IDataProvider, get_initial_sample_list);
    idp.def("get_samples", &IDataProvider::get_samples,
            py::return_value_policy::reference_internal);
    FORPY_EXPFUNC(idp, IDataProvider, track_child_nodes);
    idp.def_property_readonly("feat_vec_dim",
                              &IDataProvider::get_feat_vec_dim);
    idp.def_property_readonly("annot_vec_dim",
                              &IDataProvider::get_annot_vec_dim);
    FORPY_EXPFUNC(idp, IDataProvider, optimize_set_for_node);
    FORPY_EXPFUNC(idp, IDataProvider, get_decision_transf_func);
    FORPY_EXPFUNC(idp, IDataProvider, load_samples_for_leaf);
    idp.def("create_tree_providers",
            &IDataProvider::create_tree_providers,
            py::return_value_policy::reference_internal);

    FORPY_EXPCLASS_PARENT(PlainDataProvider, pdp, idp);
    pdp.def(py::init<Data<MatCRef>, Data<MatCRef>>(),
            py::arg("data").noconvert(),
            py::arg("annotations").noconvert(),
            py::keep_alive<1, 2>(),
            py::keep_alive<1, 3>());
    pdp.def("__repr__", [](const PlainDataProvider &self) {
        std::string ret = "PlainDataProvider[" +
          std::to_string(self.get_n_samples()) + ": " +
          std::to_string(self.get_feat_vec_dim()) + "->" +
          std::to_string(self.get_annot_vec_dim()) + "; " +
          std::to_string(self.get_initial_sample_list().size()) + " used]";
        return ret;
      });
  };

} // namespace forpy

FORPY_IMPL(EXP_SAMPLE, ITAT, forpy) {
  py::class_<Sample<IT, AT>, std::shared_ptr<Sample<IT, AT>>> smpl(
       m,
       (std::string("Sample_") +
        Name<IT>::value() + "_" +
        Name<AT>::value()).c_str());
  smpl.def("__init__",
           [](Sample<IT, AT> &self,
              const MatCRef<IT> &dta,
              const MatCRef<AT> &an,
              const float &weight) {
             if (dta.rows() != 1 || an.rows() != 1) {
               throw Forpy_Exception("Only one row per sample allowed!");
             }
             if (an.cols() == 0 || dta.cols() == 0) {
               throw Forpy_Exception("At least one data and label dimension "
                                     "required!");
             }
             const auto &block_tmp = an.block(0, 1, 1, 1);
             ptrdiff_t anstride = block_tmp.data() - an.data();
             const auto &dtablock_tmp = dta.block(0, 1, 1, 1);
             ptrdiff_t dtastride = dtablock_tmp.data() - dta.data();
             new (&self) Sample<IT, AT>(VecCMap<IT>(dta.data(),
                                                    1, dta.cols(),
                                                    Eigen::InnerStride<>(dtastride)),
                                        VecCMap<AT>(an.data(),
                                                    1, an.cols(),
                                                    Eigen::InnerStride<>(anstride)),
                                        weight);
           },
           py::arg("dta"),
           py::arg("annotation"),
           py::arg("weight")=1.f,
           py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>());
  smpl.def("__eq__", &Sample<IT, AT>::operator==);
  smpl.def("__ne__", [](const Sample<IT, AT> &self,
                        const Sample<IT, AT> &rhs) {
             return ! (self == rhs);
           });
  smpl.def("__repr__", [](const Sample<IT, AT> &self) {
      std::string ret =  "Sample<" + Name<IT>::value() + ", " +
        Name<AT>::value() + ">([";
      for (size_t i = 0; i < std::min<size_t>(self.data.cols(), 3); ++i) {
        ret += std::to_string(self.data[i]);
        if (static_cast<size_t>(self.data.cols()) > i+1) {
          ret += ", ";
        }
      }
      if (self.data.cols() > 3) {
        ret += "...";
      }
      ret += "], [";
      for (size_t i = 0; i < std::min<size_t>(self.annotation.cols(), 3); ++i) {
        ret += std::to_string(self.annotation[i]);
        if (static_cast<size_t>(self.annotation.cols()) > i+1) {
          ret += ", ";
        }
      }
      if (self.annotation.cols() > 3) {
        ret += "...";
      }
      ret += "], " + std::to_string(self.weight) + ")";
      return ret;
    });
  smpl.def_property_readonly("data", [](const Sample<IT, AT> &self) {
      return self.data;
    });
  smpl.def_property_readonly("annotation", [](const Sample<IT, AT> &self) {
      return self.annotation;
      });
  smpl.def_property_readonly("weight", [](const Sample<IT, AT> &self) {
      return self.weight;
    });
  smpl.def_property_readonly("parent_dt", [](const Sample<IT, AT> &self){
      if (self.parent_dt == nullptr) {
        throw Forpy_Exception("No parent data array registered!");
      } else {
        return *self.parent_dt;
      }
    });
  smpl.def_property_readonly("parent_at", [](const Sample<IT, AT> &self) {
      if (self.parent_dt == nullptr) {
        throw Forpy_Exception("No parent annotation array registered!");
      } else {
        return *self.parent_at;
      }
    });
};
