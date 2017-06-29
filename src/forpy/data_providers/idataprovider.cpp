#include <forpy/data_providers/idataprovider.h>

namespace forpy {
  IDataProvider::~IDataProvider() {};

  size_t IDataProvider::get_feat_vec_dim() const { return feat_vec_dim; };

  size_t IDataProvider::get_annot_vec_dim() const {return annot_vec_dim; };

  std::function<void(void*)> IDataProvider::get_decision_transf_func(const element_id_t &)
    const {
    return nullptr;
  };

  void IDataProvider::load_samples_for_leaf(const node_id_t &node_id,
                                            const node_predf &node_predictor,
                                            elem_id_vec_t *element_list) {};

  IDataProvider::IDataProvider(const size_t &feature_dimension,
                               const size_t &annotation_dimension)
    : feat_vec_dim(feature_dimension), annot_vec_dim(annotation_dimension) {};

  IDataProvider::IDataProvider() {};

} // namespace forpy
