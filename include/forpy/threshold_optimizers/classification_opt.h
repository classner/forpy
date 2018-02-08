/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_THRESHOLD_OPTIMIZERS_CLASSIFICATION_OPT_H_
#define FORPY_THRESHOLD_OPTIMIZERS_CLASSIFICATION_OPT_H_

#include "../global.h"
#include "../util/serialization/basics.h"

#include "../impurities/ientropyfunction.h"
#include "../impurities/inducedentropy.h"
#include "../types.h"
#include "../util/desk.h"
#include "./ithreshopt.h"

namespace forpy {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
//@{
/// Variables to control debugging and log output for the forpy::RegressionOpt.
const int DLOG_COPT_V = 100;
const size_t LOG_COPT_NID = 0;
const bool LOG_COPT_ALLN = false;
//@}

/// \brief Classification epsilon.
/// No differences less than this are considered existent. This is relevant for:
///
/// * if the difference between largest and smallest feature value at a node are
///   less than this threshold, optimization is stopped (invalid),
/// * if the difference between two consecutive feature values is leq than this
///   threshold, they are considered the same,
/// * it the difference between the largest feature value and the current one is
///   leq than this value, optimization is stopped.
#ifdef FORPY_SKLEARN_COMPAT
const float CLASSOPT_EPS = 1E-7;
#else
const float CLASSOPT_EPS = 1E-7f;
#endif
#pragma clang diagnostic pop

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
class ClassificationOpt : public IThreshOpt {
 public:
  /**
   * \param n_thresholds size_t>=0
   *   Number of randomly drawn threshold values that are assessed. If set to 0,
   *   the perfect split is determined. Default: 0.
   * \param gain_threshold float >=0.f
   *   The minimum information gain a split has to achieve. Default: 1E-7f.
   * \param entropy_function
   *   The entropy function to use for gain calculation during the optimization.
   */
  ClassificationOpt(const size_t &n_thresholds = 0,
                    const float &gain_threshold = 1E-7f,
                    const std::shared_ptr<IEntropyFunction> &entropy_function =
                        std::make_shared<InducedEntropy>(2));

  //@{
  /// Interface implementation.
  virtual std::shared_ptr<IThreshOpt> create_duplicate(
      const uint & /*random_seed*/) const {
    return std::make_shared<ClassificationOpt>(n_thresholds, gain_threshold,
                                               entropy_func);
  }
  virtual void check_annotations(IDataProvider *dprov);
  inline void transfer_or_run_check(IThreshOpt *other, IDataProvider *dprov) {
    auto *copt_ot = dynamic_cast<ClassificationOpt *>(other);
    if (copt_ot != nullptr) {
      copt_ot->n_classes = n_classes;
      copt_ot->true_max = true_max;
      copt_ot->class_transl_ptr = class_transl_ptr;
    } else
      other->check_annotations(dprov);
  };
  virtual void full_entropy(const IDataProvider &dprov, Desk *) const;
  virtual void optimize(Desk *) const;
  float get_gain_threshold_for(const size_t & /*node_id*/) {
    return gain_threshold;
  };
  //@}

  /** \brief Get the determined number of classes. */
  inline size_t get_n_classes() const { return n_classes; };

  inline std::shared_ptr<std::vector<uint>> get_class_translation() const {
    return class_transl_ptr;
  };

  inline uint get_true_max_class() const { return true_max; };

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const ClassificationOpt & /*self*/) {
    stream << "forpy::ClassificationOpt";
    return stream;
  };
  bool operator==(const IThreshOpt &rhs) const;

 protected:
  size_t n_thresholds;
  size_t n_classes;
  float gain_threshold;
  std::shared_ptr<IEntropyFunction> entropy_func;
  std::shared_ptr<std::vector<uint>> class_transl_ptr;
  int true_max;

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
    ar(cereal::make_nvp("base", cereal::base_class<IThreshOpt>(this)),
       CEREAL_NVP(n_thresholds), CEREAL_NVP(n_classes),
       CEREAL_NVP(gain_threshold), CEREAL_NVP(entropy_func),
       CEREAL_NVP(class_transl_ptr), CEREAL_NVP(true_max));
  }

  DISALLOW_COPY_AND_ASSIGN(ClassificationOpt);
};
}  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::ClassificationOpt);
#endif  // FORPY_THRESHOLD_OPTIMIZERS_CLASSIFICATION_OPT_H_
