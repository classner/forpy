/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_THRESHOLD_OPTIMIZERS_FASTCLASSOPT_H_
#define FORPY_THRESHOLD_OPTIMIZERS_FASTCLASSOPT_H_

#include "../global.h"
#include "../util/serialization/basics.h"

#include "../impurities/ientropyfunction.h"
#include "../impurities/shannonentropy.h"
#include "../types.h"
#include "../util/desk.h"
#include "./classification_opt.h"
#include "./ithreshopt.h"

namespace forpy {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
//@{
/// Variables to control debugging and log output for the forpy::RegressionOpt.
const int DLOG_FCOPT_V = 100;
const size_t LOG_FCOPT_NID = 3;
const bool LOG_FCOPT_ALLN = true;
//@}

/**
 * \brief Optimize split thresholds to optimize classification results.
 *
 * This threshold optimizer draws `n_thresholds` random values between the
 * minimum and maximum observed feature value and returns the best one, or finds
 * the perfect split if `n_thresholds == 0`.
 *
 * The optimizer is robust w.r.t. scaling of the features up to a certain
 * extent. It is important that the least noticable difference is larger than
 * 1E-7 (forpy::CLASSOPT_EPS).
 *
 * \ingroup forpythreshold_optimizersGroup
 */
class FastClassOpt : public ClassificationOpt {
 public:
  /**
   * \param n_thresholds size_t>=0
   *   Number of randomly drawn threshold values that are assessed. If set to 0,
   *   the perfect split is determined. Default: 0.
   * \param gain_threshold float >=0.f
   *   The minimum information gain a split has to achieve. Default: 1E-7f.
   */
  FastClassOpt(const size_t &n_thresholds = 0,
               const float &gain_threshold = 1E-7f);

  //@{
  /// Interface implementation.
  virtual std::shared_ptr<IThreshOpt> create_duplicate(
      const uint & /*random_seed*/) const {
    return std::make_shared<FastClassOpt>(n_thresholds, gain_threshold);
  }
  void full_entropy(const IDataProvider &dprov, Desk *) const;
  void optimize(Desk *) const;
  //@}

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const FastClassOpt & /*self*/) {
    stream << "forpy::FastClassOpt";
    return stream;
  };
  bool operator==(const IThreshOpt &rhs) const;

 private:
  template <typename IT>
  inline SplitOptRes<IT> &optimize__setup(DeciderDesk &d) const;
  template <typename IT>
  inline void optimize__sort(DeciderDesk &d) const;
  template <typename IT>
  inline std::unique_ptr<std::vector<IT>> optimize__thresholds(Desk *d) const;
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(cereal::make_nvp("base", cereal::base_class<ClassificationOpt>(this)));
  }

  using ClassificationOpt::class_transl_ptr;
  using ClassificationOpt::gain_threshold;
  using ClassificationOpt::n_classes;
  using ClassificationOpt::n_thresholds;
  using ClassificationOpt::true_max;
  DISALLOW_COPY_AND_ASSIGN(FastClassOpt);
};
}  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::FastClassOpt);
#endif  // FORPY_THRESHOLD_OPTIMIZERS_FASTCLASSOPT_H_
