/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_TYPES_H_
#define FORPY_TYPES_H_

#include <unordered_set>
#include <utility>
#include <vector>
#include <map>
#include <memory>
#include <numeric>
#include <Eigen/Dense>

#include <cereal/types/unordered_set.hpp>

#include "./util/hash.h"
#include "./util/storage.h"

namespace forpy {
  template<typename T>
  struct Name {
    static std::string value() { return "unknown"; };
  };
  template<>
  struct Name<double> {
    static std::string value() { return "d"; }
  };
  template<>
  struct Name<float> {
    static std::string value() { return "f"; }
  };
  template<>
  struct Name<uint> {
    static std::string value() { return "ui"; }
  };
  template<>
  struct Name<uint8_t> {
    static std::string value() { return "ui8"; }
  };
  template<>
  struct Name<int16_t> {
    static std::string value() { return "i16"; }
  };
  template<>
  struct Name<int> {
    static std::string value() { return "i"; }
  };

  /** Parameterized Matrix type. */
  template <typename DT>
  using Mat = Eigen::Matrix<DT,
                            Eigen::Dynamic,
                            Eigen::Dynamic,
                            Eigen::RowMajor>;
  template <typename DT>
  using MatCM = Eigen::Matrix<DT,
                              Eigen::Dynamic,
                              Eigen::Dynamic,
                              Eigen::ColMajor>;

  /** Parameterized standard const Matrix Ref type. */
  template <typename DT>
  using MatCRef = Eigen::Ref<const Mat<DT>>;

  template <typename DT>
  using MatCMCRef = Eigen::Ref<const MatCM<DT>>;

  /** Parameterized standard non-const Matrix Ref type. */
  template <typename DT>
  using MatRef = Eigen::Ref<Mat<DT>>;

  /** Parameterized Vector type. */
  template <typename DT>
  using Vec = Eigen::Matrix<DT,
                            Eigen::Dynamic,
                            1,
                            Eigen::ColMajor>;

  template <typename DT>
  using VecRM = Eigen::Matrix<DT,
                              1,
                              Eigen::Dynamic,
                              Eigen::RowMajor>;

  template <typename DT>
  using VecRef = Eigen::Ref<Vec<DT>>;

  template <typename DT>
  using VecRMRef = Eigen::Ref<VecRM<DT>>;

  template <typename DT>
  using VecCRef = Eigen::Ref<const Vec<DT>>;

  template <typename DT>
  using VecCMap = Eigen::Map<const Eigen::Matrix<DT,
                                                 1,
                                                 Eigen::Dynamic,
                                                 Eigen::RowMajor>,
                             0,
                             Eigen::InnerStride<>>;
  // Eigen::Stride<-1, 1>
  typedef std::pair<ptrdiff_t, ptrdiff_t> regint_t;

  /**
   * \brief Specifies the completion level for one training step.
   */
  enum class ECompletionLevel {
    /** Train one node only. */
    Node,
    /** Train one level of the tree (i.e. one depth level for BFS, one
      * branch for DFS). */
    Level,
    /** Complete the training for the entire tree. */
    Complete
  };

  /** \brief Node id type. Is size_t because it is used for array acces. */
  typedef size_t node_id_t;

  /** \brief Sample id type. Is size_t for fast array access. */
  typedef size_t element_id_t;

  /** \brief Feature id type. Is size_t for fast array access. */
  typedef size_t feature_id_t;

  /** \brief Feature id type. Is size_t for fast array access. */
  typedef size_t tree_id_t;

  typedef std::function<node_id_t(const Data<MatCRef> &,
                                  const node_id_t &,
                                  const std::function<void(void*)>&)> node_predf;

  /** \brief Convenience typedef for unsigned int. */
  typedef unsigned int uint;

  /** \brief Specifies which thresholds should be used for a decision. */
  enum class EThresholdSelection { LessOnly, GreaterOnly, Both };

  /**
   * \brief Tuple containing the optimization results.
   *
   * The order of elements is the following:
   *  -# Pair of thresholds for 'less_than' and 'greater_than' criterion.
   *  -# Types of thresholds used.
   *  -# Number of elements going to 'left'.
   *  -# Number of elements going to 'right'.
   *  -# The calculated gain value.
   *  -# Whether a 'valid' split has been found.
   */
  template <typename feature_dtype>
  using optimized_split_tuple_t = std::tuple<
    std::pair<feature_dtype, feature_dtype>,
    EThresholdSelection,
    unsigned int, unsigned int, float, bool>;

  /** \brief Denotes an element list container. */
  typedef std::vector<element_id_t> elem_id_vec_t;

  /**
   * \brief Stores the parameters for one marked tree node.
   *
   * Contains the following elements:
   *  -# A pointer to an \ref elem_id_vec_t.
   *  -# A node id (\ref node_id_t).
   *  -# The node's depth (uint).
   */
  typedef std::tuple<elem_id_vec_t,
                     node_id_t,
                     unsigned int> node_todo_tuple_t;

  /** \brief A pair containing the two child ids for a decision node. */
  typedef std::pair<node_id_t, node_id_t> node_id_pair_t;

  /** Specifies the action to do for one sample for a sample manager. */
  enum class ESampleAction {
    AddToTraining,
    RemoveFromTraining,
    AddToValidation,
    RemoveFromValidation };

  /** \brief Describes how each sample is used for each tree. */
  typedef std::vector<std::vector<size_t>> usage_map_t;

  typedef std::map<tree_id_t, std::map<ESampleAction, std::vector<element_id_t>>>
    sample_action_map_t;

  /**
   * \brief A pair containing information about newly included samples.
   *
   * The content is the following:
   *   -# A pointer to a vector of assigned sample ids.
   *   -# The threshold of already existing sample ids to new ones
   *      (old are up to excluding this value).
   */
  typedef std::pair<std::shared_ptr<std::vector<element_id_t>>,
                    element_id_t> include_pair_t;

  /** Specifies the type of tree search. */
  enum class ESearchType { DFS, BFS };

  /** \brief The type of a set of dimension selections. */
  typedef std::unordered_set<std::vector<size_t>, vector_hasher>
      proposal_set_t;

};  // namespace forp
#endif  // FORPY_TYPES_H_
