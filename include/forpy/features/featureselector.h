/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_FEATURES_STANDARDFEATURESELECTIONPROVIDER_H_
#define FORPY_FEATURES_STANDARDFEATURESELECTIONPROVIDER_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/access.hpp>

#include <unordered_set>
#include <random>
#include <vector>
#include <algorithm>

#include "../global.h"
#include "../types.h"
#include "../util/sampling.h"
#include "../util/serialization/serialization.h"
#include "./featureproposer.h"
#include "./ifeatureselector.h"

namespace forpy {
  /**
   * \brief This selection provider generates random selection combinations.
   *
   * It may be seeded for reproducible results. It can be configured to only
   * use a part of the available data dimensions. It only uses then the first
   * that are registered as used.
   *
   *
   * \ingroup forpyfeaturesGroup
   */
  class FeatureSelector : public IFeatureSelector {
   public:
    /**
     * \brief Standard constructor.
     *
     * Additional constraints on the methods arguments apply to guarantee good
     * random selection speed:
     * \f[{{how\_many\_available}\choose{selection\_dimension}}\ge
     * n\_selections\_per\_node\cdot 2,\f]
     * \f[{{max\_to\_use}\choose{selection\_dimension}}\ge
     * n\_selections\_per\_node\cdot 2.\f]
     *
     * \param n_selections_per_node size_t>0
     *   How many selection proposals are created for each node.
     * \param selection_dimension size_t>0
     *   How many data dimensions are selected per
     *   proposal. Must be > 0 and < how_many_available.
     * \param how_many_available size_t>0
     *   How many data dimensions are available.
     * \param max_to_use size_t
     *   How many data dimensions may be used. If set to zero, use how_many_available.
     *   Default: 0.
     * \param random_seed uint>0
     *   A random seed for the random number generator. Must
     *   be greater than 0. Default: 1.
     */
    FeatureSelector(
      const size_t &n_selections_per_node,
      const size_t &selection_dimension,
      const size_t &how_many_available,
      size_t max_to_use=0,
      const uint &random_seed=1);

    /** Returns how_many_available as specified in the constructor. */
    size_t get_input_dimension() const;

    /** \brief See \ref IFeatureSelectionProvider::get_selection_dimension. */
    size_t get_selection_dimension() const;

    /** Gets the associated \ref StandardFeatureSelectionGenerator. */
    std::shared_ptr<IFeatureProposer> get_proposal_generator();

    /**
     * \brief Generate a set of proposals.
     *
     * The generation is done by randomly creating new proposal sets. Each
     * proposed selection is unique in its set.
     */
    proposal_set_t get_proposals();

    /**
     * \brief Registers the given proposals as used.
     */
    void register_used(const proposal_set_t &proposals);

    size_t get_max_to_use() const;

    bool operator==(const IFeatureSelector &rhs)
      const;

   protected:
    // cppcheck-suppress uninitVar
    FeatureSelector();

   private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar, const uint &) {
      ar(cereal::make_nvp("base",
                          cereal::base_class<IFeatureSelector>(this)),
         CEREAL_NVP(dimension),
         CEREAL_NVP(how_many_per_node),
         CEREAL_NVP(how_many_available),
         CEREAL_NVP(max_to_use),
         CEREAL_NVP(used_indices),
         CEREAL_NVP(used_indices),
         CEREAL_NVP(used_index_markers),
         CEREAL_NVP(available_indices),
         CEREAL_NVP(random_engine));
    };

    size_t dimension;
    size_t how_many_per_node;
    size_t how_many_available;
    size_t max_to_use;
    std::shared_ptr<std::vector<size_t>> used_indices;
    std::vector<bool> used_index_markers;
    std::shared_ptr<std::vector<size_t>> available_indices;
    std::vector<size_t> ind_shuffle_vec;
    std::shared_ptr<std::mt19937> random_engine;

    DISALLOW_COPY_AND_ASSIGN(FeatureSelector);
  };
}  // namespace forpy
CEREAL_REGISTER_TYPE(forpy::FeatureSelector);
#endif  // FORPY_FEATURES_FEATURE_SELECTOR_H_
