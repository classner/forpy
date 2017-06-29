/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_LEAFS_CLASSIFICATIONLEAF_H_
#define FORPY_LEAFS_CLASSIFICATIONLEAF_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/access.hpp>
#include <cereal/types/unordered_map.hpp>

#include <numeric>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include "../global.h"
#include "../types.h"
#include "../util/storage.h"
#include "../data_providers/idataprovider.h"
#include "../threshold_optimizers/classificationthresholdoptimizer.h"
#include "../features/isurfacecalculator.h"
#include "./ileaf.h"

namespace forpy {
  /**
   * \brief Stores the probability distributions for n_classes at a leaf.
   *
   * \ingroup forpyleafsGroup
   */
  class ClassificationLeaf : public ILeaf {
   public:
    /**
     * \param n_classes uint
     *   The number of classes. If set to 0, they're automatically inferred.
     */
    explicit ClassificationLeaf(const uint &n_classes=0);

    bool is_compatible_with(const IDataProvider &data_provider);

    bool is_compatible_with(const IThresholdOptimizer &threshopt);

    bool needs_data() const;

    /**
     * \brief Creates and stores a probability distribution over the n_classes
     * at this leaf.
     */
    void make_leaf(
        const node_id_t &node_id,
        const elem_id_vec_t &element_list,
        const IDataProvider &data_provider);

    size_t get_result_columns(const size_t &n_trees=1) const;

    void get_result(const node_id_t &node_id,
                    Data<MatRef> &target_v,
                    const Data<MatCRef> &data=Data<MatCRef>(),
                    const std::function<void(void*)> &dptf = nullptr) const;

    /** Gets the mean of results. */
    void get_result(const std::vector<Data<Mat>> &leaf_results,
                    Data<MatRef> &target_v,
                    const Vec<float> &weights=Vec<float>()) const;

    bool operator==(const ILeaf &rhs) const;

    inline friend std::ostream &operator<<(std::ostream &stream,
                                           const ClassificationLeaf &self) {
      stream << "forpy::ClassificationLeaf[" << self.stored_distributions.size() <<
        " stored]";
      return stream;
    };

   private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar, const uint file_version) {
      ar(cereal::make_nvp("base", cereal::base_class<ILeaf>(this)),
         CEREAL_NVP(n_classes),
         CEREAL_NVP(stored_distributions));
    };

    uint n_classes;
    std::unordered_map<node_id_t, Vec<float>> stored_distributions;
  };
};  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::ClassificationLeaf);
#endif  // FORPY_LEAFS_CLASSIFICATIONLEAF_H_
