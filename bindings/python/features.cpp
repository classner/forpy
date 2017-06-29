#include <forpy/features/features.h>
#include "./macros.h"
#include "./conversion.h"

namespace py = pybind11;

namespace forpy {

#define EXP_ISURFCALC_CALC_DOC 
#define EXP_ISURFCALC_CALC_RET(IT, FT) void
#define EXP_ISURFCALC_CALC_NAME exp_isurfcalc_calc
#define EXP_ISURFCALC_CALC_PARAMNAMES it, ft, isc, m
#define EXP_ISURFCALC_CALC_PARAMCALL(IT, FT) IT(), FT(), isc, m
#define EXP_ISURFCALC_CALC_PARAMTYPESNNAMES(IT, FT) IT it, FT ft, \
    py::class_<ISurfaceCalculator, std::shared_ptr<ISurfaceCalculator>> isc, \
    py::module &m
#define EXP_ISURFCALC_CALC_PARAMTYPESNNAMESNDEF(IT, FT) IT it, FT ft,   \
    py::class_<ISurfaceCalculator, std::shared_ptr<ISurfaceCalculator>> isc, \
    py::module &m
#define EXP_ISURFCALC_CALC_MOD
  FORPY_DECL(EXP_ISURFCALC_CALC, ITFT, , ;)
  FORPY_DECL_IMPL(EXP_ISURFCALC_CALC, ITFT, ;);


  void export_feature(py::module &m) {
    FORPY_EXPCLASS_EQ(IFeatureProposer, ipg);
    FORPY_EXPFUNC(ipg, IFeatureProposer, available);
    FORPY_EXPFUNC(ipg, IFeatureProposer, max_count);
    FORPY_EXPFUNC(ipg, IFeatureProposer, get_next);
    FORPY_EXPCLASS_PARENT(FeatureProposer, pg, ipg);

    FORPY_EXPCLASS_EQ(IFeatureSelector, ifs);
    ifs.def("get_proposals", [](IFeatureSelector &self) {
        // Convert to list.
        const auto &res = self.get_proposals();
        std::vector<std::vector<size_t>> retres;
        for (const auto &prop : res) {
          retres.push_back(prop);
        }
        return retres;
      });
    FORPY_EXPFUNC(ifs, IFeatureSelector, get_proposal_generator);
    ifs.def_property_readonly("input_dimension",
                              &IFeatureSelector::get_input_dimension);
    ifs.def_property_readonly("selection_dimension",
                              &IFeatureSelector::get_selection_dimension);
    ifs.def("register_used", [](IFeatureSelector &self,
                                const std::vector<std::vector<size_t>> used) {
              proposal_set_t in;
              for (const auto &vec : used) {
                in.insert(vec);
              }
              self.register_used(in);
            });


    FORPY_EXPCLASS_PARENT(FeatureSelector, fs, ifs);
    fs.def(py::init<size_t, size_t, size_t, size_t, unsigned int>(),
           py::arg("n_selections_per_node"),
           py::arg("selection_dimension"),
           py::arg("input_dim"),
           py::arg("max_to_use")=0,
           py::arg("random_seed")=1);
    fs.def_property_readonly("max_to_use", &FeatureSelector::get_max_to_use);

    FORPY_EXPCLASS_EQ(FeatCalcParams, fcp);
    fcp.def(py::init<>());
    fcp.def_property("weights", [](const FeatCalcParams &self) {
        std::vector<float>::const_iterator it(&self.weights[0]);
        return std::vector<float>(it, it+9);
      },
      [](FeatCalcParams &self, const std::vector<float> &val) {
        if (val.size() != 9) {
          throw Forpy_Exception("A 9-element vector is required!");
        }
        for (size_t idx = 0; idx < 9; ++idx) {
          self.weights[idx] = val[idx];
        }
      });
    fcp.def_property("offsets", [](const FeatCalcParams &self) {
        std::vector<float>::const_iterator it(&self.offsets[0]);
        return std::vector<float>(it, it+2);
      },
      [](FeatCalcParams &self, const std::vector<float> &val) {
        if (val.size() != 2) {
          throw Forpy_Exception("A 2-element vector is required!");
        }
        for (size_t idx = 0; idx < 2; ++idx) {
          self.offsets[idx] = val[idx];
        }
      });
    fcp.def("__repr__", [](const FeatCalcParams &self) {
        std::string ret =  "FeatCalcParams[w: ";
        for (size_t i = 0; i < 9; ++i) {
          ret += std::to_string(self.weights[i]);
          if (i+1 < 9) {
            ret += ", ";
          }
        }
        ret += ", o: ";
        for (size_t i = 0; i < 2; ++i) {
          ret += std::to_string(self.offsets[i]);
          if (i+1 < 2) {
            ret += ", ";
          }
        }
        ret += "]";
        return ret;
      });

    FORPY_EXPCLASS_EQ(ISurfaceCalculator, isc);
    FORPY_EXPFUNC(isc, ISurfaceCalculator, propose_params);
    FORPY_EXPFUNC(isc, ISurfaceCalculator, is_compatible_to);
    isc.def_property_readonly("needs_elements_prepared",
                              &ISurfaceCalculator::needs_elements_prepared);
    isc.def_property_readonly("required_num_features",
                              &ISurfaceCalculator::required_num_features);
    FORPY_CALL(EXP_ISURFCALC_CALC, ITFT);
    FORPY_EXPCLASS_PARENT(AlignedSurfaceCalculator, asc, isc);
    asc.def(py::init<>());

    m.def("test", [](const mu::variant<std::vector<float>, std::vector<int>> &in) {
        std::cout << "hello";
      });
  };
} // namespace forpy

FORPY_IMPL(EXP_ISURFCALC_CALC, ITFT, forpy) {
  isc.def("calculate", [](const ISurfaceCalculator &self,
                          MatCM<IT> &data,
                          MatRef<FT> &out,
                          const std::vector<size_t> &feature_selection,
                          const FeatCalcParams &parameter_set,
                          const SampleVec<Sample> &samples,
                          const elem_id_vec_t &element_ids) {
            auto tmp = std::make_shared<const MatCM<IT>>(data);
            std::shared_ptr<const MatCM<FT>> tmpout;
            self.calculate(tmp, tmpout,
                           feature_selection,
                           samples,
                           element_ids,
                           parameter_set);
            if (out.rows() != tmpout->rows()) {
              throw Forpy_Exception("The output matrix must have " + 
                                    std::to_string(tmpout->rows()) + " rows!");
            }
            if (out.cols() != tmpout->cols()) {
              throw Forpy_Exception("The output matrix must have " +
                                    std::to_string(tmpout->cols()) + " cols!");
            }
            out = *tmpout;
          },
          py::arg("data"),
          py::arg("out"),
          py::arg("feature_selection")=FORPY_EMPTY_VEC,
          py::arg("parameter_set")=FeatCalcParams(),
          py::arg("samples"),//=SampleStore<Sample>(),
          py::arg("element_ids")=FORPY_EMPTY_VEC
          );
  isc.def("calculate_pred", [](const ISurfaceCalculator &self,
                               const MatCRef<IT> &data,
                               MatRef<FT> &out,
                               const std::vector<size_t> &feature_selection,
                               const FeatCalcParams &parameter_set
                               ) {
            if (out.rows() != 1) {
              throw Forpy_Exception("The output matrix must have " + 
                                    std::to_string(1) + " rows!");
            }
            if (out.cols() != 1) {
              throw Forpy_Exception("The output matrix must have " +
                                    std::to_string(1) + " cols!");
            }            
            self.calculate_pred(data, &out(0, 0),
                                feature_selection,
                                parameter_set);
            return out;
          },
          py::arg("data"),
          py::arg("out"),
          py::arg("feature_selection")=FORPY_EMPTY_VEC,
          py::arg("parameter_set")=FeatCalcParams());
};
