/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_DECIDERS_IDECIDER_H_
#define FORPY_DECIDERS_IDECIDER_H_

#include "../global.h"

#include "../util/serialization/basics.h"

#include <limits>
#include <utility>
#include <vector>

#include "../data_providers/idataprovider.h"
#include "../threshold_optimizers/ithreshopt.h"
#include "../types.h"
#include "../util/desk.h"

namespace forpy {
/**
 * \brief Interface for the decider. It does the
 * optimization of the deciding classifier for each node and stores the
 * parameters.
 *
 * \param input_dtype The datatype of the data to classify.
 * \param annotation_dtype The datatype of the annotations.
 * \param feature_dtype The datatype in which features are calculated.
 *
 * \ingroup forpydecidersGroup
 */
class IDecider {
 public:
  virtual ~IDecider();

  /** Create an equivalent, but empty, duplicate. */
  virtual std::shared_ptr<IDecider> create_duplicate(
      const uint &random_seed) const VIRTUAL_PTR;

  /**
   * \brief Optimizes a classifier for the given data and stores the params.
   *
   * This method must either set make_to_leaf to true or assure that at least
   * the minimum amount of samples per leaf is contained in each of
   * element_list_left and element_list_right. In the case that make_leaf
   * is true, the list pointers may even be returned uninitialized.
   *
   * If it is necessary to enforce additional growing constraints for the
   * tree, this is the right place (e.g. have a minimum number of samples
   * per node). The classifier manager can take these constraints into
   * account and may return make_to_leaf accordingly.
   */
  virtual void make_node(const TodoMark &todo_info,
                         const uint &min_samples_at_leaf,
                         const IDataProvider &data_provider,
                         Desk *d) const VIRTUAL_VOID;

  virtual bool is_compatible_with(const IDataProvider &dprov) VIRTUAL(bool);

  virtual void transfer_or_run_check(const std::shared_ptr<IDecider> &other,
                                     IDataProvider *dprov) VIRTUAL_VOID;

  virtual void ensure_capacity(const size_t &n_samples) VIRTUAL_VOID;

  virtual void finalize_capacity(const size_t &size) VIRTUAL_VOID;

  /**
   * \brief Makes a decision for a node with already optimized parameters.
   *
   * The classifier parameters must have been optimized for the node_id
   * before this method is called.
   *
   * \param node_id The node id of the node for which the decision should
   *                be made.
   * \param data The input data.
   * \param dptf Feature transformation function; currently unused.
   * \return true, if the decision goes to left, false otherwise.
   */
  virtual bool decide(const id_t &node_id, const Data<MatCRef> &data,
                      const std::function<void(void *)> &dptf = nullptr) const
      VIRTUAL(bool);

  /**
   * \brief Whether this classifier manager supports sample weights
   * during training.
   */
  virtual bool supports_weights() const VIRTUAL(bool);

  /** Gets the input dimension of the feature selection provider. */
  virtual size_t get_data_dim() const VIRTUAL(size_t);

  virtual void set_data_dim(const size_t &val) VIRTUAL_VOID;

  virtual std::shared_ptr<IThreshOpt> get_threshopt() const VIRTUAL_PTR;

  virtual bool operator==(const IDecider &rhs) const VIRTUAL(bool);

  std::pair<const std::vector<size_t> *,
            const mu::variant<std::vector<float>, std::vector<double>,
                              std::vector<uint32_t>, std::vector<uint8_t>>
                *> virtual get_maps() const = 0;

 protected:
  /**
   * \brief Empty constructor to allow inheritance though
   * DISALLOW_COPY_AND_ASSIGN is applied.
   */
  IDecider();

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &, const uint &){};

  DISALLOW_COPY_AND_ASSIGN(IDecider);
};

}  // namespace forpy
#endif  // FORPY_DECIDERS_IDECIDER_H_
