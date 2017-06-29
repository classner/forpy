/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_THRESHOLD_OPTIMIZERS_CLASSIFICATIONTHRESHOLDOPTIMIZER_H_
#define FORPY_THRESHOLD_OPTIMIZERS_CLASSIFICATIONTHRESHOLDOPTIMIZER_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/access.hpp>

#include <type_traits>
#include <vector>
#include <limits>
#include <random>

#include "../global.h"
#include "../types.h"
#include "../impurities/shannonentropy.h"
#include "../gains/gains.h"
#include "../util/checks.h"
#include "../util/sampling.h"
#include "../util/argsort.h"
#include "./ithresholdoptimizer.h"
#include "./feature_value_selection.h"

namespace forpy {
  /**
   * \brief Optimizes one threshold very efficiently.
   *
   * Supports only classification annotations (unsigned int) with annotation
   * values ranging in [0; n_classes - 1]. Searches the perfect threshold to
   * split the data.
   *
   * \ingroup forpythreshold_optimizersGroup
   */
  class ClassificationThresholdOptimizer
    : public IThresholdOptimizer {
   public:
    /**
     * \brief Standard constructor.
     *
     * use_fast_search_approximation is an interesting option to speed up the
     * optimization process. In general, the elements are considered in sorted
     * feature order. If use_fast_search_approximation is set to true, the
     * gain is only calculated at positions, where the currently considered
     * element is from a different class than the last one AND if the
     * feature value changed.
     *
     * This is a true approximation (i.e. the optimal gain can be at a
     * position where the current element is from the same class than the
     * last), but this hardly ever occurs for the usual gain calculation
     * functions.
     *
     * A necessary, but not sufficient criterion for the approximation to
     * be equal to the optimal value is the following:
     * Assuming the (weighted) histogram values at position \f$k\f$ are
     * \f$k_{li}\f$ for the left hand-side histogram and \f$k_{ri}\f$ for the
     * right hand-side histogram, \f$i\in[0,n\_classes-1]\f$. Then the gain
     * function \f$g(.)\f$ must have the property
     * \f[\forall j\forall k_{li},k_{ri}: g(\{k_{li}\},\{k_{ri}\})<
     * g(\{k_{li}\}_{i\backslash j}\cup\{k_{lj}+1\},
     *   \{k_{ri}\}_{i\backslash j}\cup\{k_{rj}-1\}) \vee
     * g(\{k_{li}\}_{i\backslash j}\cup\{k_{lj}-1\},
     *   \{k_{ri}\}_{i\backslash j}\cup\{k_{rj}+1\})\f].
     *
     * This does not hold in general, but for the standard information gain
     * based measures, cases where it doesn't hold occur very rarely and even
     * if so, the found positions aren't a lot worse than the theoretical
     * optimum.
     *
     * \param n_classes size_t >1
     *     The number of classes. All annotations must be in [0, ..., n_classes[.
     * \param gain_calculator \ref IGainCalculator
     *     The gain calculator to estimate the gain at each split.
     * \param gain_threshold float>=0f
     *     The minimum gain that must be reached to continue splitting.
     *     Default: 1E-7f.
     * \param use_fast_search_approximation bool
     *     Whether to use the approximation described above or not.
     */
    ClassificationThresholdOptimizer(
      const size_t &n_classes=0,
      const std::shared_ptr<IGainCalculator> &gain_calculator=
          std::make_shared<EntropyGain>(std::make_shared<ShannonEntropy>()),
      const float &gain_threshold=1E-7f,
      const bool &use_fast_search_approximation=true);

    /** Returns true. */
    bool supports_weights() const;

    void check_annotations(const IDataProvider &dprov);

    FORPY_DECL(FORPY_ITHRESHOPT_EARLYSTOP, AT, , ;)

    /**
     *\brief See \ref IThresholdOptimizer.
     *
     * Additionally, this method has the following constraints:
     *   - The sample weights must be positive.
     *   - The sum of weights of all selected samples must be < FLOAT_MAX
     *   - The annotations must be in the interval [0, n_classes[.
     *   - Any constraints that the selected gain calculator enforces.
     */
    FORPY_DECL(FORPY_ITHRESHOPT_OPT, ITFTAT, , ;)

    /** Returns the gain threshold specified in the constructor. */
    float get_gain_threshold_for(const size_t &node_id);

    bool operator==(const IThresholdOptimizer &rhs) const;

    bool getUse_fast_search_approximation() const;

    size_t getN_classes() const;

    float getGain_threshold() const;

    std::shared_ptr<IGainCalculator> getGain_calculator() const;

   private:
    FORPY_DECL_IMPL(FORPY_ITHRESHOPT_EARLYSTOP, AT, );

    FORPY_DECL_IMPL(FORPY_ITHRESHOPT_OPT, ITFTAT, );

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar, const uint &) {
      ar(cereal::make_nvp("base",
                          cereal::base_class<IThresholdOptimizer>(this)),
         CEREAL_NVP(use_fast_search_approximation),
         CEREAL_NVP(n_classes),
         CEREAL_NVP(gain_calculator),
         CEREAL_NVP(gain_threshold));
    }

    bool use_fast_search_approximation;
    size_t n_classes;
    float gain_threshold;
    std::shared_ptr<IGainCalculator> gain_calculator;

    DISALLOW_COPY_AND_ASSIGN(ClassificationThresholdOptimizer);
  };
}  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::ClassificationThresholdOptimizer);
#endif  // FORPY_THRESHOLD_OPTIMIZERS_CLASSIFICATIONTHRESHOLDOPTIMIZER_H_
