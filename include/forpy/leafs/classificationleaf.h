/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_LEAFS_CLASSIFICATIONLEAF_H_
#define FORPY_LEAFS_CLASSIFICATIONLEAF_H_

#include "../global.h"

#include "../util/serialization/basics.h"

#include <numeric>
#include <vector>

#include "../data_providers/idataprovider.h"
#include "../types.h"
#include "../util/serialization/eigen.h"
#include "../util/serialization/serialization.h"
#include "../util/storage.h"
#include "./ileaf.h"

namespace forpy {
/**
 * \brief Stores the probability distributions for n_classes at a leaf.
 *
 * \ingroup forpyleafsGroup
 */
class ClassificationLeaf : public ILeaf {
 public:
  /**
   * \param n_classes uint
   *   The number of classes. If set to 0, they're automatically inferred.
   */
  inline explicit ClassificationLeaf(const uint &n_classes = 0)
      : n_classes(n_classes),
        class_transl_ptr(nullptr),
        true_max_class(0),
        stored_distributions(){};

  //@{
  /// Interface implementation.
  inline std::shared_ptr<ILeaf> create_duplicate() const {
    return std::make_shared<ClassificationLeaf>(n_classes);
  };
  inline bool is_compatible_with(const IDataProvider & /*data_provider*/) {
    return true;
  }
  bool is_compatible_with(const IThreshOpt &threshopt);
  inline void transfer_or_run_check(ILeaf *other, IThreshOpt *thresh_opt,
                                    IDataProvider *dprov) {
    auto *cl_ot = dynamic_cast<ClassificationLeaf *>(other);
    if (cl_ot == nullptr) {
      cl_ot->n_classes = n_classes;
      cl_ot->class_transl_ptr = class_transl_ptr;
      cl_ot->true_max_class = true_max_class;
    } else {
      other->is_compatible_with(*dprov);
      other->is_compatible_with(*thresh_opt);
    }
  }
  void make_leaf(const TodoMark &todo_info, const IDataProvider &data_provider,
                 Desk *desk) const;
  size_t get_result_columns(const size_t &n_trees = 1,
                            const bool &predict_proba = false,
                            const bool &for_forest = false) const;
  Data<Mat> get_result_type(const bool &predict_proba,
                            const bool &for_forest = false) const;
  void get_result(const id_t &node_id, Data<MatRef> &target_v,
                  const bool &predict_proba, const bool &for_forest) const;
  void get_result(const std::vector<Data<Mat>> &leaf_results,
                  Data<MatRef> &target_v,
                  const Vec<float> &weights = Vec<float>(),
                  const bool &predict_proba = false) const;
  inline void ensure_capacity(const size_t &n) {
    stored_distributions.resize(n);
  };
  inline void finalize_capacity(const size_t &n) { ensure_capacity(n); };
  //@}

  bool operator==(const ILeaf &rhs) const;
  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const ClassificationLeaf &self) {
    stream << "forpy::ClassificationLeaf[" << self.stored_distributions.size()
           << " stored]";
    return stream;
  };

  const std::vector<Vec<float>> &get_stored_dists() const;

  inline const std::vector<Mat<float>> *get_map() const { return nullptr; };

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint) {
    ar(cereal::make_nvp("base", cereal::base_class<ILeaf>(this)),
       CEREAL_NVP(n_classes), CEREAL_NVP(stored_distributions));
  };

  uint n_classes;
  std::shared_ptr<std::vector<uint>> class_transl_ptr;
  uint true_max_class;
  std::vector<Vec<float>> stored_distributions;
};
};  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::ClassificationLeaf);
#endif  // FORPY_LEAFS_CLASSIFICATIONLEAF_H_
