/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_LEAFS_REGRESSIONLEAF_H_
#define FORPY_LEAFS_REGRESSIONLEAF_H_

#include "../global.h"

#include "../util/serialization/basics.h"

#include <numeric>
#include <utility>
#include <vector>

#include "../data_providers/idataprovider.h"
#include "../impurities/ientropyfunction.h"
#include "../types.h"
#include "../util/checks.h"
#include "./ileaf.h"

namespace forpy {
/**
 * \brief Manages the leaf nodes of regression trees.
 *
 * \ingroup forpyleafsGroup
 */
class RegressionLeaf : public ILeaf {
 public:
  inline explicit RegressionLeaf(const bool &store_variance = false,
                                 const bool &summarize = false)
      : leaf_regression_map(0),
        annot_dim(0),
        store_variance(store_variance),
        summarize(summarize){};

  //@{
  /// Interface implementation.
  inline std::shared_ptr<ILeaf> create_duplicate() const {
    return std::make_shared<RegressionLeaf>(store_variance);
  }

  inline bool is_compatible_with(const IDataProvider &data_provider) {
    this->annot_dim = data_provider.get_annot_vec_dim();
    return true;
  };
  inline bool is_compatible_with(const IThreshOpt & /*threshopt*/) {
    return true;
  };
  inline void transfer_or_run_check(ILeaf *other, IThreshOpt *thresh_opt,
                                    IDataProvider *dprov) {
    auto *cl_ot = dynamic_cast<RegressionLeaf *>(other);
    if (cl_ot == nullptr) {
      cl_ot->annot_dim = annot_dim;
    } else {
      other->is_compatible_with(*dprov);
      other->is_compatible_with(*thresh_opt);
    }
  }
  void make_leaf(const TodoMark &todo_info, const IDataProvider &data_provider,
                 Desk *desk) const;
  inline size_t get_result_columns(const size_t &n_trees,
                                   const bool &predict_proba,
                                   const bool & /*for_forest*/) const {
    DLOG(INFO) << "Determining result columns. Summarize: " << summarize
               << ", predict_proba: " << predict_proba
               << ", n_trees: " << n_trees;
    if (annot_dim == 0)
      throw ForpyException("This leaf manager has not been initialized yet!");
    if (predict_proba) {
      if (!store_variance)
        throw ForpyException(
            "You called `predict_proba` but didn't enable "
            "storing the variances. Use `store_variance=True` "
            "for predictor construction!");
      if (summarize) {
        VLOG(23) << "Result columns: " << 2 * annot_dim;
        return 2 * annot_dim;
      } else {
        VLOG(23) << "Result columns: " << n_trees * 2 * annot_dim;
        return n_trees * 2 * annot_dim;
      }
    } else {
      VLOG(23) << "Result columns: " << annot_dim;
      return annot_dim;
    }
  };
  inline Data<Mat> get_result_type(const bool & /*predict_proba*/,
                                   const bool & /*for_forest*/) const {
    Data<Mat> ret_mat;
    ret_mat.set<Mat<float>>();
    return ret_mat;
  };
  void get_result(const id_t &node_id, Data<MatRef> &target,
                  const bool &predict_proba, const bool &for_forest) const;
  inline const std::vector<Mat<float>> *get_map() const {
    return &leaf_regression_map;
  };
  void get_result(const std::vector<Data<Mat>> &leaf_results,
                  Data<MatRef> &target_v,
                  const Vec<float> &weights = Vec<float>(),
                  const bool &predict_proba = false) const;
  inline void ensure_capacity(const size_t &n) {
    leaf_regression_map.resize(n);
  };
  inline void finalize_capacity(const size_t &n) { ensure_capacity(n); };
  //@}

  bool operator==(const ILeaf &rhs) const;
  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const RegressionLeaf &self) {
    stream << "forpy::RegressionLeaf[" << self.leaf_regression_map.size()
           << " stored]";
    return stream;
  };

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(cereal::make_nvp("base", cereal::base_class<ILeaf>(this)),
       CEREAL_NVP(leaf_regression_map), CEREAL_NVP(annot_dim));
  }

  std::vector<Mat<float>> leaf_regression_map;
  size_t annot_dim;
  bool store_variance;
  bool summarize;
};
};  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::RegressionLeaf);
#endif  // FORPY_LEAFS_REGRESSIONLEAF_H_
