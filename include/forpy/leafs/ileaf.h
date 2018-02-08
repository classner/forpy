/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_LEAFS_ILEAF_H_
#define FORPY_LEAFS_ILEAF_H_

#include "../global.h"

#include "../util/serialization/basics.h"

#include <vector>

#include "../data_providers/idataprovider.h"
#include "../threshold_optimizers/ithreshopt.h"
#include "../types.h"

namespace forpy {
/**
 * \brief Stores and returns leaf values, and combines them to forest
 * results.
 *
 * \ingroup forpyleafsGroup
 */
class ILeaf {
 public:
  inline virtual ~ILeaf(){};
  /** Create a similar, but empty, leaf. */
  virtual std::shared_ptr<ILeaf> create_duplicate() const VIRTUAL_PTR;

  /**
   * \brief Checks compatibility with a certain \ref IDataProvider.
   *
   * This method is guaranteed to be called at the beginning of a training.
   */
  inline virtual bool is_compatible_with(
      const IDataProvider & /*data_provider*/) {
    return true;
  };

  virtual bool is_compatible_with(const IThreshOpt &threshopt) VIRTUAL(bool);

  virtual void transfer_or_run_check(ILeaf *other, IThreshOpt *thresh_opt,
                                     IDataProvider *dprov) VIRTUAL_VOID;

  /**
   * \brief Creates a leaf with the specified node_id and data.
   */
  virtual void make_leaf(const TodoMark &todo_info,
                         const IDataProvider &data_provider,
                         Desk *desk) const VIRTUAL_VOID;

  /** Get the number of summary dimensions per sample. */
  virtual size_t get_result_columns(const size_t &n_trees = 1,
                                    const bool &predict_proba = false,
                                    const bool &for_forest = false) const
      VIRTUAL(size_t);

  /** Get the result data type (a 0x0 mat within in appropriate variant). */
  virtual Data<Mat> get_result_type(const bool &predict_proba,
                                    const bool &for_forest = false) const
      VIRTUAL(Data<Mat>);

  /**
   * \brief Get the leaf data for the leaf with the given id.
   *
   * This function allocates space for storing the result. If the memory is
   * already prepared, use other overloads.
   */
  inline virtual Data<Mat> get_result(const id_t &node_id,
                                      const bool &predict_proba = false,
                                      const bool &for_forest = false) const {
    auto res_v = get_result_type(predict_proba);
    Data<Mat> ret;
    res_v.match(
        [&](const auto &res_mt) {
          typedef typename get_core<decltype(res_mt.data()[0])>::type RT;
          ret.set<Mat<RT>>(
              Mat<RT>::Zero(1, this->get_result_columns(1, predict_proba)));
          Data<MatRef> dref = MatRef<RT>(ret.get_unchecked<Mat<RT>>());
          this->get_result(node_id, dref, predict_proba, for_forest);
        },
        [](const Empty &) { throw EmptyException(); });
    return ret;
  };

  /**
   * \brief Get the leaf data for the leaf with the given id.
   */
  virtual void get_result(const id_t &node_id, Data<MatRef> &target,
                          const bool &predict_proba,
                          const bool &for_forest) const VIRTUAL_VOID;

  /**
   * \brief Combine leaf results of several trees with weights.
   *
   * This function allocates space for the result. If the memory is already
   * allocated, use another overload of this function.
   */
  inline virtual Data<Mat> get_result(
      const std::vector<Data<Mat>> &leaf_results,
      const Vec<float> &weights = Vec<float>(),
      const bool &predict_proba = false) const {
    Data<Mat> ret;
    leaf_results[0].match(
        [&](const auto &lr0) {
          typedef typename get_core<decltype(lr0.data())>::type RT;
          ret.set<Mat<RT>>(Mat<RT>::Zero(
              lr0.rows(),
              this->get_result_columns(leaf_results.size(), predict_proba, false)));
          Data<MatRef> dref = MatRef<RT>(ret.get_unchecked<Mat<RT>>());
          this->get_result(leaf_results, dref, weights, predict_proba);
        },
        [&](const Empty &) { throw EmptyException(); });
    return ret;
  };

  /** \brief Get the fused forest result. */
  inline virtual void get_result(
      const std::vector<Data<Mat>> &leaf_results, Data<MatRef> &target_v,
      const Vec<float> &weights = Vec<float>(),
      const bool &predict_proba = false) const VIRTUAL_VOID;

  /** \brief Ensure that storage is available for at least n leafs. */
  virtual void ensure_capacity(const size_t &n) VIRTUAL_VOID;

  /** \brief Cut down capacity to exactly n leafs. */
  virtual void finalize_capacity(const size_t &n) VIRTUAL_VOID;

  /** \brief Get all leafs. */
  virtual const std::vector<Mat<float>> *get_map() const = 0;

  virtual bool operator==(const ILeaf &rhs) const VIRTUAL(bool);

 protected:
  /** For deserialization. */
  inline ILeaf(){};

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &, const uint &){};

  DISALLOW_COPY_AND_ASSIGN(ILeaf);
};
};      // namespace forpy
#endif  // FORPY_LEAFS_ILEAF_H_
