/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_FEATURES_FEATUREPROPOSER_H_
#define FORPY_FEATURES_FEATUREPROPOSER_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/access.hpp>

#include <unordered_set>
#include <random>
#include <vector>
#include <algorithm>

#include "../global.h"
#include "../types.h"
#include "../util/sampling.h"
#include "../util/serialization/serialization.h"
#include "./ifeatureproposer.h"

namespace forpy {
  /** The feature generator for the \ref StandardFeatureSelectionProvider. Internal use only. */
  class FeatureProposer
    : public IFeatureProposer {
   public:
    /** Standard constructor. */
    FeatureProposer(
      const size_t &dimension,
      const size_t &index_max,
      const size_t &how_many_per_node,
      std::shared_ptr<std::vector<size_t>> used_indices,
      std::shared_ptr<std::vector<size_t>> available_indices,
      std::shared_ptr<std::mt19937> random_engine);

    /** Whether still values are available. */
    bool available() const;

    /** Returns how_many_per_node as specified in the constructor. */
    size_t max_count() const;

    /** Gets the next proposal. */
    std::vector<size_t> get_next();

    bool operator==(const IFeatureProposer &rhs) const;

   private:
    const size_t dimension;
    const size_t index_max;
    const size_t how_many_per_node;
    std::shared_ptr<const std::vector<size_t>> used_indices, available_indices;
    std::unique_ptr<SamplingWithoutReplacement<size_t>> sampler;
    std::shared_ptr<std::mt19937> random_engine;
    proposal_set_t already_used;
    size_t generated;
  };

}  // namespace forpy
#endif  // FORPY_FEATURES_FEATUREPROPOSER_H_
