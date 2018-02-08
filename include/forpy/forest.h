/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_FOREST_H_
#define FORPY_FOREST_H_

#include "./global.h"

#include "./util/serialization/basics.h"

#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "./data_providers/idataprovider.h"
#include "./deciders/fastdecider.h"
#include "./leafs/classificationleaf.h"
#include "./threshold_optimizers/fastclassopt.h"
#include "./tree.h"
#include "./types.h"
#include "./util/threading/ctpl.h"

namespace forpy {
/**
 * Standard forest class of the library.
 */
class Forest {
 public:
  /**
   * \param n_trees uint>1
   *     The number of trees.
   * \param max_depth uint > 0
   *     The maximum tree depth, including leafs (up to including).
   * \param min_samples_at_leaf uint > 0
   *     The minimum number of samples at a leaf (from including).
   * \param min_samples_at_node uint>=2*min_samples_at_leaf
   *     The minimum number of samples at a node (from including).
   * \param decider_template shared(IDecider)
   *     The decider configuration for the trees.
   * \param leaf_manager_template shared(ILeafManager)
   *     The leaf manager template for the trees.
   * \param random_seed
   *     The random seed to use to seed all trees.
   */
  Forest(const uint &n_trees = 10,
         const uint &max_depth = std::numeric_limits<uint>::max(),
         const uint &min_samples_at_leaf = 1,
         const uint &min_samples_at_node = 2,
         const std::shared_ptr<IDecider> &decider_template = nullptr,
         const std::shared_ptr<ILeaf> &leaf_manager_template = nullptr,
         const uint &random_seed = 1);

  /**
   * Combines TRAINED trees to a forest. !! Training is not possible any more !!
   *
   * \param trees vector(shared(Tree))
   *   The trained trees to combine.
   */
  Forest(std::vector<std::shared_ptr<Tree>> &trees);

  /**
   * Deserializing constructor to load a forest from a file.
   *
   * \param filename string
   *   The file to load the forest from.
   */
  Forest(std::string filename);

  /**
   * Fit the forest on the given data.
   *
   * \param data_v Variant of 2D array, col-major contiguous
   *   Col-wise data points.
   * \param annotation_v Variant of 2D array, row-major contiguous
   *   Row-wise annotations.
   * \param num_threads uint>0
   *   The number of threads to use for fitting.
   * \param bootstrap bool
   *   If set to true, resample the training set for each tree. Default: true.
   * \param weights vector<float>
   *   A vector with positive weights for each sample.
   */
  Forest *fit(const Data<MatCRef> &data_v, const Data<MatCRef> &annotation_v,
              const size_t &num_threads = 1, const bool &bootstrap = true,
              const std::vector<float> &weights = std::vector<float>());

  /**
   * Get the depths of all trees.
   *
   * The depth is defined to be 0 for an "empty" tree (only a leaf/root node)
   * and as the amount of edges on the longest path in the tree otherwise.
   */
  std::vector<size_t> get_depths() const {
    std::vector<size_t> result(trees.size());
    size_t tree_id = 0;
    for (const auto &tree_ptr : trees) {
      result[tree_id] = tree_ptr->get_depth();
      tree_id++;
    }
    return result;
  }

  /**
   * \brief The fitting function for a forest.
   *
   * Fits this forest to the data given by the data provider.
   * Releases the GIL in Python!
   *
   * \param fdata_provider shared(IDataProvider)
   *   The data provider for the fitting process.
   * \param bootstrap bool
   *   Whether to resample the training set for each tree. Default: true.
   */
  Forest *fit_dprov(const std::shared_ptr<IDataProvider> &fdata_provider,
                    const bool &bootstrap = true);

  /**
   * Predicts new data points.
   *
   * \param data_v Variant of 2D array, col-major contiguous
   *   The data predict with one sample per row.
   *
   * \param num_threads int>=0
   *   The number of threads to use for prediction. The number of
   *   samples should be at least three times larger than the number
   *   of threads to observe very good parallelization behaviour. If 0, then all
   *   available hardware threads are used.
   *   Default: 1.
   *
   * \param use_fast_prediction_if_available bool
   *   Use or construct a fast prediction tree (a summarized version of the tree
   *   that is particularly fast to index) for making the predictions. Default:
   *   true.
   *
   * \param predict_proba bool
   *   Whether or not to provide the distribution of results instead of the
   *   mode. Default: false.
   */
  Data<Mat> predict(const Data<MatCRef> &data_v, const int &num_threads = 1,
                    const bool &use_fast_prediction_if_available = true,
                    const bool &predict_proba = false) {
    if (num_threads == 0)
      throw ForpyException("The number of threads must be >0!");
    if (num_threads != 1) throw ForpyException("Unimplemented!");
    std::vector<Data<Mat>> results(0);
    Vec<float> tree_weights(Vec<float>::Zero(trees.size()));
    results.reserve(trees.size());
    for (size_t i = 0; i < trees.size(); ++i) {
      results.push_back(trees[i]->predict(
          data_v, 1, use_fast_prediction_if_available, predict_proba, true));
      tree_weights(i) = trees[i]->get_weight();
    }
    return trees[0]->combine_leaf_results(results, tree_weights, predict_proba);
  };

  /** Predict the distribution of results. */
  Data<Mat> predict_proba(const Data<MatCRef> &data_v,
                          const int &num_threads = 1,
                          const bool &use_fast_prediction_if_available = true) {
    return predict(data_v, num_threads, use_fast_prediction_if_available, true);
  };

  /** Get the required input data dimension. */
  inline size_t get_input_data_dimensions() const {
    return trees[0]->get_input_data_dimensions();
  }

  /** Get the decider of the first tree. */
  inline std::shared_ptr<const IDecider> get_decider() const {
    return trees[0]->get_decider();
  };

  /** Get the tree vector. */
  inline std::vector<std::shared_ptr<Tree>> get_trees() const { return trees; }

  /** Enable fast prediction for all trees. */
  inline void enable_fast_prediction() {
    for (auto &tree : trees) tree->enable_fast_prediction();
  };

  /** Disable fast prediction for all trees. */
  inline void disable_fast_prediction() {
    for (auto &tree : trees) tree->disable_fast_prediction();
  };

  /** Gets the leaf manager of the first tree. */
  inline std::shared_ptr<const ILeaf> get_leaf_manager() const {
    return trees[0]->get_leaf_manager();
  }

  /** Get all tree weights. */
  inline std::vector<float> get_tree_weights() const {
    std::vector<float> result(trees.size());
    for (std::size_t tree_idx = 0; tree_idx < trees.size(); ++tree_idx) {
      result[tree_idx] = trees.at(tree_idx)->get_weight();
    }
    return result;
  }

  /** Set the tree weights. */
  inline void set_tree_weights(const std::vector<float> &weights) const {
    if (weights.size() != trees.size())
      throw ForpyException("Need " + std::to_string(trees.size()) +
                           " weights, received " +
                           std::to_string(weights.size()));
    for (std::size_t tree_idx = 0; tree_idx < trees.size(); ++tree_idx)
      trees[tree_idx]->set_weight(weights[tree_idx]);
  }

  /**
   * Saves the forest to a file with the specified name.
   *
   * \param filename string
   *   The filename to use.
   */
  void save(const std::string &filename) const;

  inline bool operator==(const Forest &rhs) const {
    if (trees.size() != rhs.trees.size()) return false;
    for (size_t i = 0; i < trees.size(); ++i) {
      if (!(*(trees.at(i)) == *(rhs.trees.at(i)))) return false;
    }
    if (random_seed != rhs.random_seed) return false;
    return true;
  };

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const Forest &self) {
    stream << "forpy::Forest[" << self.trees.size() << " trees]";
    return stream;
  };

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    // To make this work, I had to take out a static assertion at cereal.hpp,
    // at least for clang. Everything works as expected without the assertion
    // and I wouldn't know why not.
    ar(CEREAL_NVP(trees), CEREAL_NVP(random_seed));
  };

  std::vector<std::shared_ptr<Tree>> trees;
  uint random_seed;
  DISALLOW_COPY_AND_ASSIGN(Forest);
};  // class Forest

class ClassificationForest : public Forest {
 public:
  inline ClassificationForest(const std::string &filename) : Forest(filename){};
  ClassificationForest(const size_t &n_trees = 10,
                       const uint &max_depth = std::numeric_limits<uint>::max(),
                       const uint &min_samples_at_leaf = 1,
                       const uint &min_samples_at_node = 2,
                       const uint &n_valid_features_to_use = 0,
                       const bool &autoscale_valid_features = true,
                       const uint &random_seed = 1,
                       const size_t &n_thresholds = 0,
                       const float &gain_threshold = 1E-7f);

  inline std::unordered_map<std::string, mu::variant<uint, size_t, float, bool>>
  get_params(const bool & /*deep*/ = false) const {
    return params;
  }

  inline std::shared_ptr<ClassificationForest> set_params(
      const std::unordered_map<
          std::string, mu::variant<uint, size_t, float, bool>> &params) {
    return std::make_shared<ClassificationForest>(
        GetWithDefVar<size_t>(params, "n_trees", 10),
        GetWithDefVar<uint>(params, "max_depth",
                            std::numeric_limits<uint>::max()),
        GetWithDefVar<uint>(params, "min_samples_at_leaf", 1),
        GetWithDefVar<uint>(params, "min_samples_at_node", 2),
        GetWithDefVar<uint>(params, "n_valid_features_to_use", 0),
        GetWithDefVar<bool>(params, "autoscale_valid_features", true),
        GetWithDefVar<uint>(params, "random_seed", 1),
        GetWithDefVar<size_t>(params, "n_thresholds", 0),
        GetWithDefVar<float>(params, "gain_threshold", 1E-7f));
  }

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const ClassificationForest &self) {
    stream << "forpy::ClassificationForest["
           << mu::static_variant_cast<size_t>(self.params.at("n_trees"))
           << " trees]";
    return stream;
  };

 private:
  std::unordered_map<std::string, mu::variant<uint, size_t, float, bool>>
      params;
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(cereal::make_nvp("base", cereal::base_class<Forest>(this)),
       CEREAL_NVP(params));
  }
  DISALLOW_COPY_AND_ASSIGN(ClassificationForest);
};

class RegressionForest : public Forest {
 public:
  inline RegressionForest(const std::string &filename) : Forest(filename){};
  RegressionForest(const size_t &n_trees = 10,
                   const uint &max_depth = std::numeric_limits<uint>::max(),
                   const uint &min_samples_at_leaf = 1,
                   const uint &min_samples_at_node = 2,
                   const uint &n_valid_features_to_use = 0,
                   const bool &autoscale_valid_features = false,
                   const uint &random_seed = 1, const size_t &n_thresholds = 0,
                   const float &gain_threshold = 1E-7f,
                   const bool &store_variance = false,
                   const bool &summarize = false);

  inline std::unordered_map<std::string, mu::variant<uint, size_t, float, bool>>
  get_params(const bool & /*deep*/ = false) const {
    return params;
  }

  inline std::shared_ptr<RegressionForest> set_params(
      const std::unordered_map<
          std::string, mu::variant<uint, size_t, float, bool>> &params) {
    return std::make_shared<RegressionForest>(
        GetWithDefVar<size_t>(params, "n_trees", 10),
        GetWithDefVar<uint>(params, "max_depth",
                            std::numeric_limits<uint>::max()),
        GetWithDefVar<uint>(params, "min_samples_at_leaf", 1),
        GetWithDefVar<uint>(params, "min_samples_at_node", 2),
        GetWithDefVar<uint>(params, "n_valid_features_to_use", 0),
        GetWithDefVar<bool>(params, "autoscale_valid_features", false),
        GetWithDefVar<uint>(params, "random_seed", 1),
        GetWithDefVar<size_t>(params, "n_thresholds", 0),
        GetWithDefVar<float>(params, "gain_threshold", 1E-7f),
        GetWithDefVar<bool>(params, "store_variance", false),
        GetWithDefVar<bool>(params, "summarize", false));
  }

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const RegressionForest &self) {
    stream << "forpy::RegressionForest["
           << mu::static_variant_cast<size_t>(self.params.at("n_trees"))
           << " trees]";
    return stream;
  };

 private:
  std::unordered_map<std::string, mu::variant<uint, size_t, float, bool>>
      params;
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(cereal::make_nvp("base", cereal::base_class<Forest>(this)),
       CEREAL_NVP(params));
  }
  DISALLOW_COPY_AND_ASSIGN(RegressionForest);
};

};      // namespace forpy
#endif  // FORPY_FOREST_H_
