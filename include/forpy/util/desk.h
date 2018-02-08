/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_UTIL_DESK_H_
#define FORPY_UTIL_DESK_H_

#include "../global.h"
#include "../types.h"

namespace forpy {

/**
 * \brief Desk for tree training.
 *
 * Storage container for threaded tree training (see also @ref
 * forpydeskGroup).
 *
 * @ingroup forpydeskGroup
 */
struct TreeDesk {
  /** Storing marked nodes for DFS. (`std::stack` has no `clear()` method,
   *  that's why I gave preference to an `std::vector`.) */
  std::vector<TodoMark> marks;
  /** Tracking how many samples have been stored (mainly as sanity check).
   *  Usually points to forpy::Tree::stored_in_leafs. */
  std::atomic<size_t> *stored_in_leafs;
  /** To determine the ID of the next tree node to use. Usually points to
   *  forpy::Tree::next_id (see the doc there for the storage concept). */
  std::atomic<id_t> *next_id_p;
  /** A vector representation of the tree. Usually points to forpy::Tree::tree.
   */
  std::vector<std::pair<id_t, id_t>> *tree_p;

  /**
   * \brief Set up all the internal pointers.
   */
  inline void setup(std::atomic<size_t> *silp, std::atomic<id_t> *np,
                    std::vector<std::pair<id_t, id_t>> *tsp) {
    stored_in_leafs = silp;
    next_id_p = np;
    tree_p = tsp;
  }
  /**
   * \brief Clear the marks and reset all pointers.
   */
  inline void reset() {
    marks.clear();
    stored_in_leafs = nullptr;
    next_id_p = nullptr;
    tree_p = nullptr;
  }
};

/**
 * \brief Desk for decider training.
 *
 * Storage container for threaded decider training (see also @ref
 * forpydeskGroup).
 *
 * @ingroup forpydeskGroup
 */
struct DeciderDesk {
  //@{
  /// These variables are populated in IDecider::make_node.
  size_t n_samples, input_dim, annot_dim;
  uint min_samples_at_leaf;
  /// Points only to the relevant section of the full id list! E.g., the full
  /// id list is [0, 1, 2, 3, 4] and this node is optimized for IDs 1:3, the
  /// pointer points to element 1.
  id_t *elem_id_p;
  id_t start_id, end_id, node_id;
  //@}

  //@{
  /// Variables used during the threshold optimization.
  OptSplitV best_res_v, opt_res_v;
  id_t best_feat_idx;
  bool presorted, need_sort;
  std::vector<id_t> feature_indices;
  //@}

  //@{
  /// Variables initialized in IThreshOpt::full_entropy.
  std::vector<float> full_sum;
  float *full_sum_p;
  float fullentropy, maxproxy;
  const float *annot_p;
  const uint *class_annot_p;
  size_t annot_os;
  const float *weights_p;
  float full_w;
  std::vector<id_t> sort_perm;
  id_t *sort_perm_p;
  std::vector<id_t> elem_ids_sorted;
  id_t *elem_ids_sorted_p;
  std::vector<float> feat_values;
  float *feat_p;
  DataV class_feat_values;
  std::vector<float> left_sum_vec;
  float *left_sum_p;
  //@}

  /// Must be initialized before calling IThreshOpt::optimize. Points to the
  /// a selected feature vector (full samples, so even if for this node only
  /// samples 1:3 are needed, this points to the feature for sample 0) with
  /// stride 1. This can be done using IDataProvider::get_feature .
  mu::variant<const float *, const double *, const uint *, const uint8_t *>
      full_feat_p_v;

  //@{
  /// Return values from IThreshOpt::optimize.
  bool make_to_leaf;
  // These only contain valid values if `make_to_leaf` is false.
  interv_t left_int, right_int;
  id_t left_id, right_id;
  //@}

  /// Stores the number of elements in a vector of feature IDs that have been
  /// determined as invalid (e.g., because they are constant). For these values
  /// to work correctly, only DFS may be used (otherwise, the vector of features
  /// may be re-sorted and counts may become invalid before the node
  /// optimization reaches relevant nodes).
  std::vector<size_t> invalid_counts;

  /// Pointer to a shared vector of a mapping node_id->feature. Since multiple
  /// threads never write to the same node and the vector is guaranteed to be
  /// large enough, concurrent writes can be performed.
  std::vector<size_t> *node_to_featsel_p;
  /// Pointer to a shared vector of a mapping node_id->threshold. Since multiple
  /// threads never write to the same node and the vector is guaranteed to be
  /// large enough, concurrent writes can be performed.
  mu::variant<std::vector<float>, std::vector<double>, std::vector<uint32_t>,
              std::vector<uint8_t>> *node_to_thresh_v_p;

  inline void setup(
      std::vector<size_t> *ntfp,
      mu::variant<std::vector<float>, std::vector<double>,
                  std::vector<uint32_t>, std::vector<uint8_t>> *nttp) {
    node_to_featsel_p = ntfp;
    node_to_thresh_v_p = nttp;
    if (ntfp != nullptr) invalid_counts.resize(ntfp->size());
  }
  inline void reset() {
    n_samples = input_dim = annot_dim = 0;
    min_samples_at_leaf = 0;
    elem_id_p = nullptr;
    weights_p = nullptr;
    full_w = 0.f;
    start_id = end_id = node_id = 0;
    node_to_featsel_p = nullptr;
    node_to_thresh_v_p = nullptr;
    invalid_counts.clear();
  }
};

/**
 * \brief Desk for leaf manager training.
 *
 * Storage container for threaded decider training.
 *
 * @ingroup forpydeskGroup
 */
struct LeafDesk {
  /// Pointer to a shared vector of a mapping node_id->regression value. Since
  /// multiple threads never write to the same node and the vector is guaranteed
  /// to be large enough, concurrent writes can be performed.
  std::vector<Mat<float>> *leaf_regression_map_p;
  inline void setup(std::vector<Mat<float>> *lrmp) {
    leaf_regression_map_p = lrmp;
  };
  inline void reset() { leaf_regression_map_p = nullptr; }
};

/**
 * \brief Desk for coordinating the random engines.
 *
 * Storage container for threaded decider training (see also \ref
 * forpydeskGroup).
 *
 * @ingroup forpydeskGroup
 */
struct RandomDesk {
  std::mt19937 random_engine;
  uint seed;
  inline void setup(const uint &seed) {
    this->seed = seed;
    random_engine.seed(seed);
  };
  inline void reset() {
    this->seed = 0;
    random_engine.seed(1);
  };
};

/**
 * \brief Main thread desk object.
 *
 * Contains all thread-local variables for one thread.
 *
 * @ingroup forpydeskGroup
 */
struct Desk {
  inline Desk(int i) : thread_id(i){};
  TreeDesk t;
  DeciderDesk d;
  LeafDesk l;
  RandomDesk r;

  const int thread_id;
  inline void setup(
      std::atomic<size_t> *stored_in_leaf_p, std::atomic<id_t> *next_id_p,
      std::vector<std::pair<id_t, id_t>> *tree_p,
      std::vector<size_t> *ntfp = nullptr,
      mu::variant<std::vector<float>, std::vector<double>,
                  std::vector<uint32_t>, std::vector<uint8_t>> *nttp = nullptr,
      std::vector<Mat<float>> *lrmp = nullptr, const uint &random_seed = 0) {
    /// Setup for thread-local processing.
    t.setup(stored_in_leaf_p, next_id_p, tree_p);
    d.setup(ntfp, nttp);
    l.setup(lrmp);
    r.setup(random_seed);
  };
  inline void reset() {
    t.reset();
    d.reset();
    l.reset();
    r.reset();
  }
};

};      // namespace forpy
#endif  // FORPY_UTIL_DESK_H_
