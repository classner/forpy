/* Author: Moritz Einfalt, Christoph Lassner. */
#pragma once
#ifndef FORPY_LEAFS_REGRESSIONLEAF_H_
#define FORPY_LEAFS_REGRESSIONLEAF_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/access.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/utility.hpp>

#include <numeric>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <utility>

#include "../global.h"
#include "../types.h"
#include "../data_providers/idataprovider.h"
#include "../impurities/ientropyfunction.h"
#include "../impurities/shannonentropy.h"
#include "../util/regression/iregressor.h"
#include "../util/regression/linearregressor.h"
#include "../features/ifeatureselector.h"
#include "../features/featureselector.h"
#include "../util/checks.h"
#include "./ileaf.h"

namespace forpy {
  /**
   * \brief Manages the leaf nodes of regression trees.
   *
   * This leaf manager creates leaf nodes and stores a probabilistic regression
   * model at each leaf.
   *
   * \ingroup forpyleafsGroup
   */
  class RegressionLeaf : public ILeaf {
   public:
    /**
     * \brief Constructor for a RegressionLeafManager
     *
     * Costructs a RegressionLeafManager.
     * For each leaf, a number of dimension selections used as regressors is asessed.
     * The selection resulting in the regression model with the lowest entropy is used.
     *
     * \param n_valids_to_use size_t>0
     *   How many valid selections are asessed, until the selection process is
     *   stopped.
     * \param regression_calculator IRegressionCalculator
     *   The regression calculator that is used to generate a regression model for each leaf.
     * \param entropy_function IEntropyFunction
     *   The entropy function used to evaluate the regression models.
     * \param use_fallback_constant_regression bool
     *   When no valid dimension selections can be found and this flag is set to true,
     *   a ConstantRegressionCalculator (independent from regressor selections) is used instead.
     *   Otherwise, this case results in a runtime exception. Default: false.
     * \param num_threads int>0
     *   The number of threads used when evaluating different selections.
     *   Default: 1.
     * \param summary_mode uint<3
     *   Determines the meaning of the values in the 2D prediction matrix of
     *   a forest (the output of the convenience `predict` method of a forest).
     *   Case 0: Each row contains the prediction for each regressor (the first
     *           half of its entries) and the expected variances for each
     *           regressor (second half of its entries). To estimate the joint
     *           variance, a gaussian is fitted over the multimodal distribution
     *           defined by all trees.
     *   Case 1: Each row contains the prediction for each regressor (the first
     *           half of its entries) and the mean of the expected variances of
     *           each tree. This has no direct semantic meaning, but can give
     *           better results in active learning applications.
     *   Case 2: Each row contains the prediction for each regressor and
     *           the variance estimate for each regressor for each tree, e.g.,
     *           (r11, r12, v11, v12, r21, r22, v21, v22, ...), with `r` and `v`
     *           denoting regressor prediction and variance respectively, the
     *           first index the tree and the second index the regressor index.
     * \returns A new RegressionLeafManager.
     */
    explicit RegressionLeaf(
        const std::shared_ptr<IRegressor> &regressor_template=
          std::make_shared<LinearRegressor>(),
        const uint &summary_mode=0,
        const size_t &regression_input_dim=0,
        const size_t &selections_to_try=0,
        const int &num_threads=1,
        const uint &random_seed=1);

    bool is_compatible_with(const IDataProvider &data_provider);

    bool is_compatible_with(const IThresholdOptimizer &threshopt);

    bool needs_data() const;

    void make_leaf(
        const node_id_t &node_id,
        const elem_id_vec_t &element_list,
        const IDataProvider &data_provider);

    /** Gets the number of summary dimensions per sample. */
    size_t get_result_columns(const size_t &n_trees=1) const;

    Data<Mat> get_result_type() const;

    void get_result(
        const node_id_t &node_id,
        Data<MatRef> &target,
        const Data<MatCRef> &data=Data<MatCRef>(),
        const std::function<void(void*)> &dptf = nullptr) const;

    Data<Mat> get_result(const node_id_t &node_id,
                         const Data<MatCRef> &data=Data<MatCRef>(),
                         const std::function<void(void*)> &dptf=nullptr) const;

    void get_result(const std::vector<Data<Mat>> &leaf_results,
                    Data<MatRef> &target_v,
                    const Vec<float> &weights=Vec<float>()) const;

    Data<Mat> get_result(const std::vector<Data<Mat>> &leaf_results,
                         const Vec<float> &weights=Vec<float>()) const;

    bool operator==(const ILeaf &rhs) const;

    inline friend std::ostream &operator<<(std::ostream &stream,
                                    const RegressionLeaf &self) {
      stream << "forpy::RegressionLeaf[" << self.leaf_regression_map.size() <<
        " stored]";
      return stream;
    };

   private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive &ar, const uint &) {
      ar(cereal::make_nvp("base",
                          cereal::base_class<ILeaf>(this)),
         CEREAL_NVP(reg_calc_template),
         CEREAL_NVP(summary_mode),
         CEREAL_NVP(regression_input_dim),
         CEREAL_NVP(selections_to_try),
         CEREAL_NVP(num_threads),
         CEREAL_NVP(random_seed),
         CEREAL_NVP(leaf_regression_map),
         CEREAL_NVP(max_input_dim),
         CEREAL_NVP(annot_dim),
         CEREAL_NVP(double_mode));
    }

    std::shared_ptr<IRegressor> reg_calc_template;
    size_t summary_mode;
    size_t regression_input_dim;
    size_t selections_to_try;
    int num_threads;
    uint random_seed;
    std::unordered_map<node_id_t, std::pair<std::unique_ptr<IRegressor>,
                                            std::vector<size_t>>> leaf_regression_map;
    size_t max_input_dim;
    size_t annot_dim;
    bool double_mode;
  };
};  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::RegressionLeaf);
#endif  // FORPY_LEAFS_REGRESSIONLEAF_H_
