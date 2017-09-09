/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_DECIDERS_THRESHOLDDECIDER_H_
#define FORPY_DECIDERS_THRESHOLDDECIDER_H_

#include <cereal/access.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>

#include <unordered_map>
#include <mutex>
#include <utility>
#include <vector>
#include <limits>
#include <typeinfo>

#include "../global.h"
#include "../types.h"
#include "../util/storage.h"
#include "../features/isurfacecalculator.h"
#include "../features/alignedsurfacecalculator.h"
#include "../threshold_optimizers/ithresholdoptimizer.h"
#include "../threshold_optimizers/classificationthresholdoptimizer.h"
#include "./idecider.h"

namespace forpy {
  /**
   * \brief A classifier manager for weak classifiers with a filter function,
   * a feature calculation function and a thresholding.
   *
   * The classifier design is heavily inspired by "Decision Forests for
   * Classification, Regression, Density Estimation, Manifold Learning and
   * Semi-Supervised Learning" (Criminisi, Shotton and Konukoglu, 2011).
   * With their definition, node classifier parameters \f$\theta\f$ can
   * be split into three parts:
   *  - \f$\phi\f$: a filter function that selects relevant features,
   *  - \f$\psi\f$: parameters of a function that combines the feature values
   *                to a single scalar,
   *  - \f$\tau\f$: thresholding parameters for the calculated scalar.
   *
   * With this model, a decision can be made at each node based on whether the
   * calculated scalar lies withing the thresholding bounds.
   *
   * \ingroup forpydecidersGroup
   */
  class ThresholdDecider : public IDecider {
   public:
    /**
     * \param feature_calculator shared(ISurfaceCalculator)
     *   The feature calculation function. Its
     *   parameters are \f$\psi\f$, and it combines the data
     *   dimensions to a single scalar feature.
     * \param threshold_optimizer shared(IThresholdOptimizer)
     *   Optimizes \f$\tau\f$.
     * \param n_valid_features_to_use size_t
     *   The threshold optimizer may hint that
     *   a selected feature may be completely inappropriate for the
     *   currently searched split. If the feature selection provider
     *   does provide sufficiently many features, the classifier may
     *   use the next one and "not count" the inappropriate one.
     *   This is the maximum number of "valid" features that are
     *   used per split. If 0, ignore the flag returned by the
     *   optimizer and always use all suggested feature combinations
     *   provided by the feature selection provider. Default: 0.
     * \param num_threads size_t>0
     *   The number of threads to use for threshold optimization.
     *   Independent of the number of threads, the result is
     *   guaranteed to be the same. Default: 1.
     */
    ThresholdDecider(                     
      const std::shared_ptr<IThresholdOptimizer> &threshold_optimizer,
      const size_t &n_valid_features_to_use,
      const std::shared_ptr<ISurfaceCalculator> &feature_calculator=
          std::make_shared<AlignedSurfaceCalculator>(),
      const int &num_threads = 1,
      const uint &random_seed=1);

    /**
     * \brief Optimizes \f$\theta=(\phi,\psi,\tau)\f$ by a (non-exhaustive)
     * random search over the parameter space. \f$\tau\f$ is usually optimized
     * perfectly.
     *
     * See the description of the virtual interface function
     * \ref IClassifierManager::optimize_and_set_for_node for more
     * information on the general purpose and parameter descriptions.
     */
    std::tuple<bool, elem_id_vec_t, elem_id_vec_t> make_node(
      const node_id_t &node_id,
      const uint &node_depth,
      const uint &min_samples_at_leaf,
      const elem_id_vec_t &element_id_list,
      const IDataProvider &data_provider);

    /**
     * \brief Decides whether a sample should go left or right at a node.
     *
     * \return true if a sample goes left, false otherwise.
     */
    bool decide(const node_id_t &node_id,
                const Data<MatCRef> &data_v,
                const std::function<void(void*)> &decision_param_transf = nullptr)
     const;

    bool supports_weights() const;

    size_t get_data_dim() const;

    std::shared_ptr<IThresholdOptimizer> get_threshopt() const;

    std::shared_ptr<ISurfaceCalculator> get_featcalc() const;

    bool operator==(const IDecider &rhs) const;

    inline friend std::ostream &operator<<(std::ostream &stream,
                                           const ThresholdDecider &self) {
      stream << "forpy::ThresholdDecider[" << self.node_to_featsel.size() <<
        " stored]";
      return stream;
    };

    void set_data_dim(const size_t &val);

    std::pair<const std::unordered_map<node_id_t, std::vector<size_t>> *,
              const mu::variant<std::unordered_map<node_id_t, float>,
                                std::unordered_map<node_id_t, double>,
                                std::unordered_map<node_id_t, uint32_t>,
                                std::unordered_map<node_id_t, uint8_t>> *>
        get_maps() const;

   private:
    ThresholdDecider();

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar, const uint &) {
      ar(cereal::make_nvp("base",
                          cereal::base_class<IDecider>(this)),
         CEREAL_NVP(feature_calculator),
         CEREAL_NVP(threshold_optimizer),
         CEREAL_NVP(n_valids_to_use),
         CEREAL_NVP(node_to_featsel),
         CEREAL_NVP(node_to_thresh_v),
         CEREAL_NVP(num_threads),
         CEREAL_NVP(compat_SurfCalc_DProv_checked),
         CEREAL_NVP(random_seed),
         CEREAL_NVP(data_dim));
    }

    std::shared_ptr<ISurfaceCalculator> feature_calculator;
    std::shared_ptr<IThresholdOptimizer> threshold_optimizer;
    size_t n_valids_to_use;
    std::unordered_map<node_id_t, std::vector<size_t>> node_to_featsel;
    mu::variant<std::unordered_map<node_id_t, float>,
                std::unordered_map<node_id_t, double>,
                std::unordered_map<node_id_t, uint32_t>,
                std::unordered_map<node_id_t, uint8_t>> node_to_thresh_v;
    int num_threads;
    bool compat_SurfCalc_DProv_checked;
    uint random_seed;
    size_t data_dim;
  };
};  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::ThresholdDecider);
#endif  // FORPY_DECIDERS_THRESHOLDDECIDER_H_
