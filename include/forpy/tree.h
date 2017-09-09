/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_TREE_H_
#define FORPY_TREE_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/utility.hpp>

#include <tuple>
#include <vector>
#include <deque>
#include <memory>
#include <utility>
#include <string>
#include <fstream>

#include "./forpy.h"

namespace forpy {
  /**
   * \brief The main tree class for the forpy framework.
   *
   * This class is the core element of the framework. It can be used as a
   * standalone tree or to form a forest. It is highly customizable by
   * providing the IClassifierManager and ILeafManager.
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
     *                  the return values of the leaf nodes.
     */
    Tree(const uint &max_depth,
         const uint &min_samples_at_leaf,
         const uint &min_samples_at_node,
         const std::shared_ptr<IDecider> &decider,
         const std::shared_ptr<ILeaf> &leaf_manager);

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
     * The function is always using the frontmost mark in the todo-list.
     *
     * \param data_provider shared(IDataProvider)
     *   The data provider to use.
     * \param append_on_different bool
     *   Where to append the produced node-marks in
     *   the deque "todo-list". Appending the nodes on the
     *   same side they're read from leads to performing a
     *   DFS, while appending them on the different end
     *   leads to performing a BFS.
     */
    void make_node(IDataProvider *data_provider,
                   const bool &append_on_different);

    /**
     * \brief Do one DFS step with given completion level.
     *
     * For \ref CompletionLevel::Level, the branch of the tree below the
     * currently marked node is completed.
     *
     * \param data_provider shared(IDataProvider)
     *   The data provider to use to get the samples with
     *   the relevant ids.
     * \param completion CompletionLevel
     *   The \ref CompletionLevel to reach before returning
     *   from the function.
     */
    size_t DFS(IDataProvider *data_provider,
             const ECompletionLevel &completion);

    /**
     * \brief Do one BFS step with given completion level.
     *
     * For \ref CompletionLevel::Level, all nodes with the same depth than the
     * currently processed one are completed. It is assumed that the current
     * deque state originates from a BFS search, i.e. all node marks on the
     * same level are next to each other in the todo-deque.
     *
     * \param data_provider shared(IDataProvider)
     *   The data provider to use to get the samples with
     *   the relevant ids.
     * \param completion CompletionLevel
     *   The \ref CompletionLevel to reach before returning
     *   from the function.
     */
    size_t BFS(IDataProvider *data_provider,
               const ECompletionLevel &completion);

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
     * \param data Array<input_dtype>, 2D, row-major contiguous
     *   Row-wise data points.
     * \param annotations Array<annotation_dtype>, 2D, row-major contiguous
     *   Row-wise annotations.
     * \param complete_dfs bool
     *   If set to true, finishes training the tree. Otherwise, the training
     *   is just set up, and \ref make_node must be called. Default: true.
     */
    void fit(const Data<MatCRef> &data,
             const Data<MatCRef> &annotations,
             const bool &complete_dfs = true);

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
    void fit_dprov(IDataProvider *data_provider,
                   const bool complete_dfs = true);

    /**
     * \brief Get the leaf id of the leaf where the given data will arrive.
     *
     * \param data The data to propagate through the tree.
     * \param data_step A step size in memory from one data element to the
     *             next.
     * \return The node id of the leaf.
     */
    node_id_t predict_leaf(const Data<MatCRef> &data,
                           const node_id_t &start_node = 0,
                           const std::function<void(void*)> &dptf = nullptr)
     const;

    /**
     * Predicts new data points.
     *
     * Releases the GIL in Python!
     *
     * \param data Array<input_data>, 2D, row-major contiguous
     *   The data predict with one sample per row.
     *
     * \param num_threads int>0
     *   The number of threads to use for prediction. The number of
     *   samples should be at least three times larger than the number
     *   of threads to observe very good parallelization behavior.
     */
    Data<Mat> predict(const Data<MatCRef> &data_v,
                      const int &num_threads=1,
                      const bool &use_fast_prediction_if_available=true);

    /**
     * \brief Get the data prediction result for the given data.
     *
     * \param data The data to propagate through the tree.
     * \param data_step A step size in memory from one data element to the
     *             next.
     * \return The prediction result from the leaf.
     */
    Data<Mat> predict_leaf_result(const Data<MatCRef> &data,
                                  const node_id_t &start_node = 0,
                                  const std::function<void(void*)> &dptf = nullptr)
      const;

    /**
     * Combine the leaf results of several trees to the forest result.
     */
    Data<Mat> combine_leaf_results(
        const std::vector<Data<Mat>> &leaf_results,
        const Vec<float> &weights=Vec<float>()) const;

    /**
     * \brief Whether the trees \ref fit method has been called and its DFS
     * and BFS methods can now be used.
     */
    bool is_initialized() const;

    /**
     * \brief The tree weight.
     */
    float get_weight() const;

    /**
     * \brief The number of tree nodes.
     */
    size_t get_n_nodes() const;

    /**
     * \brief Sets the tree weight.
     */
    void set_weight(const float &new_weight);

    /**
     * \brief The data dimension that is required by this tree.
     */
    size_t get_input_data_dimensions() const;

    /**
     * \brief The classifier manager used by this tree.
     */
    std::shared_ptr<const IDecider> get_decider() const;

    /**
     * \brief The leaf manager used by this tree.
     */
    std::shared_ptr<const ILeaf> get_leaf_manager() const;

    /**
     * \brief The number of samples stored in leafs.
     */
    size_t get_samples_stored() const;

    /**
     * Get the vector of marked nodes.
     */
    std::deque<node_todo_tuple_t> get_marks() const;

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
    void disable_fast_prediction();

    bool operator==(Tree const &rhs) const;

    /**
     * \brief Saves the tree.
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
    // cppcheck-suppress uninitVar
    Tree();

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar, const uint &) {
      ar(CEREAL_NVP(max_depth),
         CEREAL_NVP(is_initialized_for_training),
         CEREAL_NVP(min_samples_at_node),
         CEREAL_NVP(min_samples_at_leaf),
         CEREAL_NVP(weight),
         CEREAL_NVP(decider),
         CEREAL_NVP(leaf_manager),
         CEREAL_NVP(tree),
         CEREAL_NVP(marks),
         CEREAL_NVP(stored_in_leafs));
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
    size_t stored_in_leafs;
    /** The associated classifier manager. */
    std::shared_ptr<IDecider> decider;
    /** The associated leaf manager. */
    std::shared_ptr<ILeaf> leaf_manager;
    /** Holds the entire tree structure. */
    std::vector<node_id_pair_t> tree;
    /** Pointer to a structure that can be used for fast predictions.

        Vector ids are node ids. The first value in the tuple is the threshold
        value at that node. If the first and second tuple elements are the same,
        they contain a leaf ID.
     */
    std::unique_ptr<mu::variant<std::vector<std::tuple<size_t, float, size_t, size_t>>,
                                std::vector<std::tuple<size_t, double, size_t, size_t>>,
                                std::vector<std::tuple<size_t, uint32_t, size_t, size_t>>,
                                std::vector<std::tuple<size_t, uint8_t, size_t, size_t>>>>
    fast_tree;
    /**
     * A storage for marked nodes that still must to be processed during the
     * tree building process.
     */
    std::deque<node_todo_tuple_t> marks;
    // If any, a deep copy must be made of a tree to guarantee consistency
    // between the tree layout and the saved features, classifiers and leafs.
    // This is disallowed for the first.
    DISALLOW_COPY_AND_ASSIGN(Tree);
  };
};  // namespace forpy
#endif  // FORPY_TREE_H_
