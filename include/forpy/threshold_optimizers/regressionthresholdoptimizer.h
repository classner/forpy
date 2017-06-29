/* Author: Moritz Einfalt, Christoph Lassner. */
#pragma once
#ifndef FORPY_THRESHOLD_OPTIMIZERS_REGRESSIONTHRESHOLDOPTIMIZER_H_
#define FORPY_THRESHOLD_OPTIMIZERS_REGRESSIONTHRESHOLDOPTIMIZER_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/access.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <type_traits>
#include <vector>
#include <limits>
#include <random>
#include <set>

#include "../global.h"
#include "../types.h"
#include "../util/checks.h"
#include "../util/sampling.h"
#include "../util/argsort.h"
#include "../util/regression/regression.h"
#include "../util/serialization/stl/random.h"
#include "../impurities/impurities.h"
#include "./ithresholdoptimizer.h"
#include "./feature_value_selection.h"

namespace forpy {

  /**
   * \brief Optimizes the threshold for splitting a dataset, to ensure optimal
   * regression results on the splits.
   *
   * This threshold optimizer draws n_thresholds random values between the
   * minimum and maximum observed feature value and returns the best one.
   * Multiple annotations (and therefore multiple output regression) are
   * allowed. The splits are evaluated using a provided regression calculator.
   *
   * \ingroup forpythreshold_optimizersGroup
   */
  class RegressionThresholdOptimizer : public IThresholdOptimizer {
   public:
    /**
     * \param n_thresholds size_t>0
     *   Number of randomly drawn threshold values that are asessed.
     * \param regression_calculator shared(IRegressor)
     *   The regression calculator used to evaluate possible splits.
     * \param entropy_function shared(IEntropyFunction)
     *   The entropy function used on the regression results.
     * \param gain_threshold float >=0.f
     *   The minimum information gain a split has to achieve.
     * \param random_seed uint >0
     *   The random seed.
     * \returns A new RegressionThresholdOptimizer.
     */
    RegressionThresholdOptimizer(
      const size_t & n_thresholds=0,
      const std::shared_ptr<IRegressor> &regressor_template=
          std::make_shared<ConstantRegressor>(),
      const std::shared_ptr<IEntropyFunction> &entropy_function=
          std::make_shared<ShannonEntropy>(),
      const float &gain_threshold=1E-7f,
      const unsigned int  &random_seed=1);

    /** Returns false! */
    bool supports_weights() const;

    /** Initializes the random engines for parallel processing. */
    void prepare_for_optimizing(const size_t &node_id,
                                const int &num_threads);

    FORPY_DECL(FORPY_ITHRESHOPT_OPT, ITFTATR, , ;)

    FORPY_DECL(FORPY_ITHRESHOPT_EARLYSTOP, AT, , ;)

    void check_annotations(const IDataProvider &dprov);

    /** Returns the gain threshold specified in the constructor. */
    float get_gain_threshold_for(const size_t &node_id);

    std::shared_ptr<IRegressor> getRegressorTemplate() const;

    bool operator==(const IThresholdOptimizer &rhs) const;

   private:
    FORPY_DECL_IMPL(FORPY_ITHRESHOPT_OPT, ITFTATR, );

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive &ar, const uint &) {
      ar(cereal::make_nvp("base",
                          cereal::base_class<IThresholdOptimizer>(this)),
         CEREAL_NVP(n_thresholds),
         CEREAL_NVP(entropy_function),
         CEREAL_NVP(reg_calc_template),
         CEREAL_NVP(gain_threshold),
         CEREAL_NVP(random_engine),
         CEREAL_NVP(thread_engines),
         CEREAL_NVP(main_seed),
         CEREAL_NVP(seed_dist));
    }

    size_t n_thresholds;
    std::shared_ptr<IEntropyFunction> entropy_function;
    std::shared_ptr<IRegressor> reg_calc_template;
    float gain_threshold;
    std::shared_ptr<std::mt19937> random_engine;
    std::vector<std::unique_ptr<std::mt19937>> thread_engines;
    unsigned int main_seed;
    std::uniform_int_distribution<unsigned int> seed_dist;

    DISALLOW_COPY_AND_ASSIGN(RegressionThresholdOptimizer);
  };
}  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::RegressionThresholdOptimizer);
#endif  // FORPY_THRESHOLD_OPTIMIZERS_REGRESSIONTHRESHOLDOPTIMIZER_H_
