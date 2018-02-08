/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_THRESHOLD_OPTIMIZERS_ITHRESHOPT_H_
#define FORPY_THRESHOLD_OPTIMIZERS_ITHRESHOPT_H_

#include "../global.h"
#include "../util/serialization/basics.h"

#include "../data_providers/idataprovider.h"
#include "../types.h"
#include "../util/desk.h"

namespace forpy {

/**
 * \brief Find an optimal threshold.
 *
 * This class is classically used by a forpy::IDecider to optimize
 * the threshold (\f$\tau\f$).
 *
 * \ingroup forpythreshold_optimizersGroup
 */
class IThreshOpt {
 public:
  virtual inline ~IThreshOpt(){};
  virtual std::shared_ptr<IThreshOpt> create_duplicate(
      const uint &random_seed) const VIRTUAL_PTR;
  /**
   * \brief Whether this threshold optimizer can take into account weights
   * during the optimization.
   *
   * By default, return false.
   */
  inline virtual bool supports_weights() const { return false; };

  /** \brief Validate annotations for usability with this optimizer. */
  virtual void check_annotations(IDataProvider *dprov) VIRTUAL_VOID;

  virtual void transfer_or_run_check(IThreshOpt *other,
                                     IDataProvider *dprov) VIRTUAL_VOID;

  /** \brief Get the full entropy for one node before optimization.
   *
   * Setup tasks can be performed within this function. If the entropy
   * determined is below a certain threshold, the optimization is stopped (see,
   * e.g., forpy::FastDecider).
   */
  virtual void full_entropy(const IDataProvider &dprov,
                            Desk *) const VIRTUAL_VOID;

  /** \brief Optimize for one node. */
  virtual void optimize(Desk *) const VIRTUAL_VOID;

  /** \brief Get the gain threshold to use for this node. */
  virtual float get_gain_threshold_for(const size_t &node_id) VIRTUAL(float);

  virtual bool operator==(const IThreshOpt &rhs) const VIRTUAL(bool);

 protected:
  /** For deserialization. */
  inline IThreshOpt(){};

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &, const uint &) {}
};
}  // namespace forpy
#endif  // FORPY_THRESHOLD_OPTIMIZERS_ITHRESHOPT_H_
