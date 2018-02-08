/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_TREE_H_
#define FORPY_TREE_H_

#include "./global.h"

#include "./util/serialization/basics.h"

#include <atomic>
#include <fstream>
#include <future>
#include <mapbox/variant_cast.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "./data_providers/fastdprov.h"
#include "./data_providers/idataprovider.h"
#include "./deciders/idecider.h"
#include "./leafs/ileaf.h"
#include "./types.h"
#include "./util/desk.h"

namespace forpy {
class Forest;

/**
 * \brief The main tree class for the forpy framework.
 *
 * This class is the core element of the framework. It can be used as a
 * standalone tree or to form a forest.
 */
class Tree {
 public:
  /**
   * \brief The standard constructor for the forpy trees.
   *
   * \param max_depth uint > 0
   *     The maximum tree depth, including leafs (up to including).
   * \param min_samples_at_leaf uint > 0
   *     The minimum number of samples at a leaf (from including).
   * \param min_samples_at_node uint>=2*min_samples_at_leaf
   *     The minimum number of samples at a node (from including).
   * \param decider \ref IDecider
   *     The decider that stores, optimizes and applies the decision rules
   *     for each inner tree node.
   * \param leaf_manager The leaf manager generates, stores and handles
   *     the return values of the leaf nodes.
   * \param random_seed uint>0
   *     Seed for the random engine.
   */
  Tree(const uint &max_depth = std::numeric_limits<uint>::max(),
       const uint &min_samples_at_leaf = 1, const uint &min_samples_at_node = 2,
       const std::shared_ptr<IDecider> &decider = nullptr,
       const std::shared_ptr<ILeaf> &leaf_manager = nullptr,
       const uint &random_seed = 1);

  /**
   * \brief Deserialization constructor for the forpy trees.
   *
   * \param filename string
   *   The filename to deserialize the tree from.
   */
  Tree(std::string filename);

  /**
   * \brief Handle the creation of one tree node.
   *
   * Takes the next one of the list of marked nodes and fits it to the
   * data. If necessary, creates two child nodes and a split criterion,
   * otherwise makes it a leaf.
   *
   * The function is to be used within a thread (see forpy::Tree::parallel_DFS).
   * It is marked `const` so as to avoid concurrent writes to member elements.
   * Everything that is written to must be available in a forpy::Desk.
   *
   * \param data_provider shared(IDataProvider)
   *   The data provider to use.
   * \param d Desk
   *   Desk to use thread local memory from.
   */
  void make_node(const IDataProvider *data_provider, Desk *d);

  /**
   * \brief Do one DFS step with given completion level.
   *
   * For \ref CompletionLevel::Level, the branch of the tree below the
   * currently marked node is completed.
   *
   * The function is to be used within a thread (see forpy::Tree::parallel_DFS).
   *
   * \param data_provider forpy::IDataProvider*
   *   The data provider to use to get the samples with
   *   the relevant ids.
   * \param completion CompletionLevel
   *   The \ref ECompletionLevel to reach before returning
   *   from the function.
   * \param d Desk
   *   Desk to use thread local memory from.
   */
  void DFS(const IDataProvider *data_provider,
           const ECompletionLevel &completion, Desk *d);
  void parallel_DFS(Desk *d, TodoMark &mark, IDataProvider *data_provider,
                    const bool &finalize = true);
  void DFS_and_store(Desk *d, TodoMark &mark, const IDataProvider *dprov,
                     const ECompletionLevel &comp);

  /**
   * Get the tree depth.
   *
   * The depth is defined to be 0 for an "empty" tree (only a leaf/root node)
   * and as the amount of edges on the longest path in the tree otherwise.
   *
   */
  size_t get_depth() const;

  /**
   * \brief Standard fitting function.
   *
   * Fits this tree to the data given by the data provider. If
   * complete_dfs is true, the tree is completely fitted to the data
   * Otherwise, just a node todo for the root node is added and the tree
   * may be performed step-by-step by calling the \ref BFS or \ref DFS
   * functions.
   *
   * Releases the GIL in Python!
   *
   * \param data_v Variant of 2D array, col-major contiguous
   *   Col-wise data points.
   * \param annotation_v Variant of 2D array, row-major contiguous
   *   Row-wise annotations.
   * \param n_threads size_t
   *   The number of threads to use. If set to 0, use all hardware threads.
   * \param complete_dfs bool
   *   If set to true, finishes training the tree. Otherwise, the training
   *   is just set up, and \ref make_node must be called. Default: true.
   * \param weights vector<float>
   *   A vector with positive weights for each sample or an empty vector.
   */
  Tree *fit(const Data<MatCRef> &data_v, const Data<MatCRef> &annotation_v,
            const size_t &n_threads, const bool &complete_dfs = true,
            const std::vector<float> &weights = std::vector<float>());

  /**
   * \brief The fitting function for a single tree.
   *
   * Fits this tree to the data given by the data provider. If
   * complete_dfs is true, the tree is completely fitted to the data
   * Otherwise, just a node todo for the root node is added and the tree
   * may be performed step-by-step by calling the \ref BFS or \ref DFS
   * functions.
   *
   * \param data_provider shared(IDataProvider)
   *   The data provider for the fitting process.
   * \param complete_dfs bool
   *   If true, complete the fitting process.
   */
  Tree *fit_dprov(std::shared_ptr<IDataProvider> data_provider,
                  const bool &complete_dfs = true);

  /**
   * \brief Get the leaf id of the leaf where the given data will arrive.
   *
   * \param data The data to propagate through the tree.
   * \param start_node The node to start from, doesn't have to be the root.
   * \param dptf Feature mapping function; disabled at the moment.
   * \return The node id of the leaf.
   */
  id_t predict_leaf(const Data<MatCRef> &data, const id_t &start_node = 0,
                    const std::function<void(void *)> &dptf = nullptr) const;

  /**
   * Predicts new data points.
   *
   * Releases the GIL in Python!
   *
   * \param data_v Variant of 2D data, row-major contiguous
   *   The data predict with one sample per row.
   *
   * \param num_threads int>0
   *   The number of threads to use for prediction. The number of
   *   samples should be at least three times larger than the number
   *   of threads to observe good parallelization behavior. Currently disabled.
   *
   * \param use_fast_prediction_if_available bool If set to true (default), this
   *   will create a compressed version of the tree that has particularly
   *   favorable properties for fast access and use it for predictions. You can
   *   trigger the creation manually by calling Tree::enable_fast_prediction.
   *
   * \param predict_proba bool
   *   If enabled, will ask the leaf manager to provide probability information
   *   additionally to the prediction output.
   *
   * \param for_forest bool
   *   If set to true, will create an intermediate result that can be fused to
   *   a whole forest result. Not relevant for end-users.
   */
  Data<Mat> predict(const Data<MatCRef> &data_v, const int &num_threads = 1,
                    const bool &use_fast_prediction_if_available = true,
                    const bool &predict_proba = false,
                    const bool &for_forest = false);

  /**
   * \brief Overload for consistency with the sklearn interface.
   *
   * @ref Tree::predict.
   */
  Data<Mat> predict_proba(const Data<MatCRef> &data_v,
                          const int &num_threads = 1,
                          const bool &use_fast_prediction_if_available = true);

  /**
   * \brief Get the data prediction result for the given data.
   */
  inline Data<Mat> predict_leaf_result(
      const Data<MatCRef> &data, const id_t &start_node = 0,
      const std::function<void(void *)> &dptf = nullptr) const {
    return leaf_manager->get_result(predict_leaf(data, start_node, dptf));
  };

  /**
   * Combine the leaf results of several trees to the forest result.
   */
  inline Data<Mat> combine_leaf_results(
      const std::vector<Data<Mat>> &leaf_results,
      const Vec<float> &weights = Vec<float>(),
      const bool &predict_proba = false) const {
    return leaf_manager->get_result(leaf_results, weights, predict_proba);
  };

  /**
   * \brief Whether the trees \ref fit method has been called and its DFS
   * and BFS methods can now be used.
   */
  inline bool is_initialized() const { return is_initialized_for_training; };

  /**
   * \brief The tree weight.
   */
  inline float get_weight() const { return weight; };

  /**
   * \brief The number of tree nodes.
   */
  inline size_t get_n_nodes() const { return tree.size(); };

  /**
   * \brief Sets the tree weight.
   */
  inline void set_weight(const float &new_weight) { weight = new_weight; };

  /**
   * \brief The data dimension that is required by this tree.
   */
  inline size_t get_input_data_dimensions() const {
    return decider->get_data_dim();
  };

  /**
   * \brief The classifier manager used by this tree.
   */
  inline std::shared_ptr<const IDecider> get_decider() const {
    return decider;
  };

  /**
   * \brief The leaf manager used by this tree.
   */
  inline std::shared_ptr<const ILeaf> get_leaf_manager() const {
    return leaf_manager;
  };

  /**
   * \brief The number of samples stored in leafs.
   */
  inline size_t get_samples_stored() const { return stored_in_leafs.load(); };

  inline const std::vector<std::pair<id_t, id_t>> get_tree() const {
    return tree;
  };

  /**
   * Unpack the hash maps for thresholds and feature IDs for fast predictions.
   *
   * This only works for trees with threshold deciders and
   * AlignedSurfaceCalcluators for the features. Requires more memory than the
   * default trees, but is significantly faster.
   */
  void enable_fast_prediction();

  /**
   * Frees the memory from the unpacked trees for fast predictions.
   */
  inline void disable_fast_prediction() {
    VLOG(9) << "Disabling fast prediction; freeing memory.";
    fast_tree.reset();
  };

  bool operator==(Tree const &rhs) const;

  /**
   * \brief Save the tree.
   *
   * \param filename string
   *   The filename of the file to store the tree in.
   */
  void save(const std::string &filename) const;

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const Tree &self) {
    stream << "forpy::Tree[depth " << self.get_depth() << "]";
    return stream;
  };

 private:
  friend class forpy::Forest;
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(CEREAL_NVP(max_depth), CEREAL_NVP(is_initialized_for_training),
       CEREAL_NVP(min_samples_at_node), CEREAL_NVP(min_samples_at_leaf),
       CEREAL_NVP(weight), CEREAL_NVP(decider), CEREAL_NVP(leaf_manager),
       CEREAL_NVP(tree), CEREAL_NVP(stored_in_leafs), CEREAL_NVP(next_id),
       CEREAL_NVP(random_seed));
  };

  /**
   * The maximum depth of the tree. Non-const for serialization purposes
   * only.
   */
  uint max_depth;
  /**
   * Whether the \ref fit method has been called and the DFS and BFS
   * methods can now be used for training.
   */
  bool is_initialized_for_training;
  /** The minimum number of samples that must arrive at an inner node. */
  unsigned int min_samples_at_node;
  /** The minimum number of samples that must arrive at a leaf. */
  unsigned int min_samples_at_leaf;
  /** A weight assigned to this tree. Can be used by learning algorithms. */
  float weight;
  /** The amount of samples stored in leafs so far. */
  std::atomic<size_t> stored_in_leafs;
  /** The associated classifier manager. */
  std::shared_ptr<IDecider> decider;
  /** The associated leaf manager. */
  std::shared_ptr<ILeaf> leaf_manager;
  /** Holds the entire tree structure. */
  std::vector<std::pair<id_t, id_t>> tree;
  /** Pointer to a structure that can be used for fast predictions.

      Vector ids are node ids. The first value in the tuple is the threshold
      value at that node. If the first and second tuple elements are the same,
      they contain a leaf ID.
   */
  std::unique_ptr<
      mu::variant<std::vector<std::tuple<size_t, float, size_t, size_t>>,
                  std::vector<std::tuple<size_t, double, size_t, size_t>>,
                  std::vector<std::tuple<size_t, uint32_t, size_t, size_t>>,
                  std::vector<std::tuple<size_t, uint8_t, size_t, size_t>>>>
      fast_tree;
  std::vector<std::future<void>> futures;
  std::mutex fut_mtx;
  std::atomic<id_t> next_id;
  uint random_seed;
  // If any, a deep copy must be made of a tree to guarantee consistency
  // between the tree layout and the saved features, classifiers and leafs.
  // This is disallowed for the first.
  DISALLOW_COPY_AND_ASSIGN(Tree);
};

class ClassificationTree : public Tree {
 public:
  inline ClassificationTree(const std::string &filename) : Tree(filename){};
  ClassificationTree(const uint &max_depth = std::numeric_limits<uint>::max(),
                     const uint &min_samples_at_leaf = 1,
                     const uint &min_samples_at_node = 2,
                     const uint &n_valid_features_to_use = 0,
                     const bool &autoscale_valid_features = false,
                     const uint &random_seed = 1,
                     const size_t &n_thresholds = 0,
                     const float &gain_threshold = 1E-7f);

  inline std::unordered_map<std::string, mu::variant<uint, size_t, float, bool>>
  get_params(const bool & /*deep*/ = false) const {
    return params;
  }

  inline std::shared_ptr<ClassificationTree> set_params(
      const std::unordered_map<
          std::string, mu::variant<uint, size_t, float, bool>> &params) {
    return std::make_shared<ClassificationTree>(
        GetWithDefVar<uint>(params, "max_depth",
                            std::numeric_limits<uint>::max()),
        GetWithDefVar<uint>(params, "min_samples_at_leaf", 1),
        GetWithDefVar<uint>(params, "min_samples_at_node", 2),
        GetWithDefVar<uint>(params, "n_valid_features_to_use", 0),
        GetWithDefVar<bool>(params, "autoscale_valid_features", false),
        GetWithDefVar<uint>(params, "random_seed", 1),
        GetWithDefVar<size_t>(params, "n_thresholds", 0),
        GetWithDefVar<float>(params, "gain_threshold", 1E-7f));
  }

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const ClassificationTree &self) {
    stream << "forpy::ClassificationTree[depth " << self.get_depth() << "]";
    return stream;
  };

 private:
  std::unordered_map<std::string, mu::variant<uint, size_t, float, bool>>
      params;
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(cereal::make_nvp("base", cereal::base_class<Tree>(this)),
       CEREAL_NVP(params));
  }
  DISALLOW_COPY_AND_ASSIGN(ClassificationTree);
};

class RegressionTree : public Tree {
 public:
  inline RegressionTree(const std::string &filename) : Tree(filename){};
  RegressionTree(const uint &max_depth = std::numeric_limits<uint>::max(),
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

  inline std::shared_ptr<RegressionTree> set_params(
      const std::unordered_map<
          std::string, mu::variant<uint, size_t, float, bool>> &params) {
    return std::make_shared<RegressionTree>(
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
                                         const RegressionTree &self) {
    stream << "forpy::RegressionTree[depth " << self.get_depth() << "]";
    return stream;
  };

 private:
  std::unordered_map<std::string, mu::variant<uint, size_t, float, bool>>
      params;
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(cereal::make_nvp("base", cereal::base_class<Tree>(this)),
       CEREAL_NVP(params));
  }
  DISALLOW_COPY_AND_ASSIGN(RegressionTree);
};

};      // namespace forpy
#endif  // FORPY_TREE_H_
