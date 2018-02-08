/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_THRESHOLD_OPTIMIZERS_REGOPT_H_
#define FORPY_THRESHOLD_OPTIMIZERS_REGOPT_H_

#include "../global.h"
#include "../util/serialization/basics.h"

#include "../types.h"
#include "../util/desk.h"
#include "./ithreshopt.h"

namespace forpy {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
//@{
/// Variables to control debugging and log output for the forpy::RegressionOpt.
const int DLOG_ROPT_V = 1;
const size_t LOG_ROPT_NID = 12043;
const bool LOG_ROPT_ALLN = false;
//@}

/// \brief Regression epsilon.
/// No differences less than this are considered existent. This is relevant for:
///
/// * if the difference between largest and smallest feature value at a node are
///   less than this threshold, optimization is stopped (invalid),
/// * if the difference between two consecutive feature values is leq than this
///   threshold, they are considered the same,
/// * it the difference between the largest feature value and the current one is
///   leq than this value, optimization is stopped.
#ifdef FORPY_SKLEARN_COMPAT
const float REGOPT_EPS = 1E-7;
#else
const float REGOPT_EPS = 1E-7f;
#endif
#pragma clang diagnostic pop

/**
 * \brief Optimize split thresholds to optimize regression results (MSE).
 *
 * This threshold optimizer draws `n_thresholds` random values between the
 * minimum and maximum observed feature value and returns the best one, or finds
 * the perfect split if `n_thresholds == 0`. Multiple annotations (and therefore
 * multiple output regression) are allowed.
 *
 * The optimizer is robust w.r.t. scaling of the features up to a certain
 * extent. It is important that the least noticable difference is larger than
 * 1E-7 (forpy::REGOPT_EPS).
 *
 * \ingroup forpythreshold_optimizersGroup
 */
class RegressionOpt : public IThreshOpt {
 public:
  /**
   * \param n_thresholds size_t>=0
   *   Number of randomly drawn threshold values that are assessed. If set to 0,
   *   the perfect split is determined. Default: 0.
   * \param gain_threshold float >=0.f
   *   The minimum information gain a split has to achieve. Default: 1E-7f.
   */
  RegressionOpt(const size_t &n_thresholds = 0,
                const float &gain_threshold = 1E-7f);

  //@{
  /// Interface implementation.
  virtual std::shared_ptr<IThreshOpt> create_duplicate(
      const uint & /*random_seed*/) const {
    return std::make_shared<RegressionOpt>(n_thresholds, gain_threshold);
  }
  void check_annotations(IDataProvider *dprov);
  inline void transfer_or_run_check(IThreshOpt *other, IDataProvider *dprov) {
    auto *ot_ropt = dynamic_cast<RegressionOpt *>(other);
    if (ot_ropt == nullptr) ot_ropt->check_annotations(dprov);
  };
  void full_entropy(const IDataProvider &dprov, Desk *) const;
  void optimize(Desk *) const;
  float get_gain_threshold_for(const size_t & /*node_id*/) {
    return gain_threshold;
  };
  //@}

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const RegressionOpt & /*self*/) {
    stream << "forpy::RegressionOpt";
    return stream;
  };
  bool operator==(const IThreshOpt &rhs) const;

 private:
  inline SplitOptRes<float> &optimize__setup(DeciderDesk &d) const;
  inline void optimize__sort(DeciderDesk &d) const;
  inline std::unique_ptr<std::vector<float>> optimize__thresholds(
      Desk *d) const;
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(cereal::make_nvp("base", cereal::base_class<IThreshOpt>(this)),
       CEREAL_NVP(n_thresholds), CEREAL_NVP(gain_threshold));
  }

  size_t n_thresholds;
  float gain_threshold;

  DISALLOW_COPY_AND_ASSIGN(RegressionOpt);
};
}  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::RegressionOpt);
#endif  // FORPY_THRESHOLD_OPTIMIZERS_REGOPT_H_
