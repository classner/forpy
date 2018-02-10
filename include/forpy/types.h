/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_TYPES_H_
#define FORPY_TYPES_H_

#include <Eigen/Dense>
#include <map>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "./util/hash.h"
#include "./util/storage.h"

namespace forpy {

/** \brief Convenience typedef for unsigned int. */
typedef unsigned int uint;

/**
 * \brief Struct for translating primitive types to a short name.
 */
template <typename T>
struct Name {
  static std::string value() { return "unknown"; };
};
template <>
struct Name<double> {
  static std::string value() { return "d"; }
};
template <>
struct Name<float> {
  static std::string value() { return "f"; }
};
template <>
struct Name<uint> {
  static std::string value() { return "ui"; }
};
template <>
struct Name<uint8_t> {
  static std::string value() { return "ui8"; }
};
template <>
struct Name<int16_t> {
  static std::string value() { return "i16"; }
};
template <>
struct Name<int> {
  static std::string value() { return "i"; }
};

/** \brief Parameterized Matrix type (row major). */
template <typename DT>
using Mat = Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/** \brief Parameterized column major matrix type. */
template <typename DT>
using MatCM =
    Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

/** \brief Parameterized const matrix ref type. */
template <typename DT>
using MatCRef = Eigen::Ref<const Mat<DT>>;

/** \brief Parameterized const matrix column major matrix ref type. */
template <typename DT>
using MatCMCRef = Eigen::Ref<const MatCM<DT>>;

/** \brief Parameterized standard non-const matrix ref type. */
template <typename DT>
using MatRef = Eigen::Ref<Mat<DT>>;

/** Parameterized vector type. */
template <typename DT>
using Vec = Eigen::Matrix<DT, Eigen::Dynamic, 1, Eigen::ColMajor>;

template <typename DT>
using VecRM = Eigen::Matrix<DT, 1, Eigen::Dynamic, Eigen::RowMajor>;

template <typename DT>
using VecRef = Eigen::Ref<Vec<DT>>;

template <typename DT>
using VecRMRef = Eigen::Ref<VecRM<DT>>;

template <typename DT>
using VecCRef = Eigen::Ref<const Vec<DT>>;

template <typename DT>
using VecCMap =
    Eigen::Map<const Eigen::Matrix<DT, 1, Eigen::Dynamic, Eigen::RowMajor>,
               Eigen::Unaligned, Eigen::InnerStride<>>;

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

/** \brief Element id type. */
typedef size_t id_t;

typedef std::function<id_t(const Data<MatCRef> &, const id_t &,
                           const std::function<void(void *)> &)>
    node_predf;

/** \brief Specifies which thresholds should be used for a decision. */
enum class EThresholdSelection { LessEqOnly, GreaterOnly, Both };

template <typename FT>
struct SplitOptRes {
  id_t split_idx;
  FT thresh;
  float gain;
  bool valid;

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const SplitOptRes<FT> &self) {
    stream << "forpy::SplitOptRes_X[<=" << self.thresh
           << "; gain: " << self.gain << ", valid: " << self.valid << "]";
    return stream;
  };

  bool operator==(SplitOptRes<FT> const &rhs) const {
    return split_idx == rhs.split_idx && thresh == rhs.thresh &&
           gain == rhs.gain && valid == rhs.valid;
  }
};
typedef mu::variant<SplitOptRes<float>, SplitOptRes<double>, SplitOptRes<uint>,
                    SplitOptRes<uint8_t>>
    OptSplitV;

typedef std::pair<id_t, id_t> interv_t;

/**
 * \brief Stores the parameters for one marked tree node.
 *
 * Contains the following elements:
 *  -# shared_ptr to the element id list,
 *  -# interval of the list to work with,
 *  -# A node id (\ref id_t).
 *  -# The node's depth (uint).
 */
struct TodoMark {
  inline TodoMark() : node_id(0), depth(0){};
  inline TodoMark(std::shared_ptr<std::vector<id_t>> sample_ids,
                  const interv_t &interv, const id_t &node_id,
                  const uint &depth)
      : sample_ids(sample_ids),
        interv(interv),
        node_id(node_id),
        depth(depth){};
  std::shared_ptr<std::vector<id_t>> sample_ids;
  interv_t interv;
  id_t node_id;
  uint depth;
  inline bool operator==(TodoMark const &rhs) const {
    return node_id == rhs.node_id && depth == rhs.depth &&
           interv == rhs.interv && *sample_ids == *(rhs.sample_ids);
  }
  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const TodoMark &self) {
    stream << "forpy::TodoMark[node_id:  " << self.node_id << ", depth "
           << self.depth << "]";
    return stream;
  };
  MOVE_ASSIGN(TodoMark);

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(CEREAL_NVP(sample_ids), CEREAL_NVP(interv), CEREAL_NVP(node_id),
       CEREAL_NVP(depth));
  };
  DISALLOW_COPY_AND_ASSIGN(TodoMark);
};

typedef std::pair<ptrdiff_t, ptrdiff_t> regint_t;

/** \brief Describes how each sample is used for each tree. */
typedef std::vector<std::pair<std::shared_ptr<std::vector<size_t>>,
                              std::shared_ptr<std::vector<float> const>>>
    usage_map_t;

/**
 * \brief A pair containing information about newly included samples.
 *
 * The content is the following:
 *   -# A pointer to a vector of assigned sample ids.
 *   -# The threshold of already existing sample ids to new ones
 *      (old are up to excluding this value).
 */
typedef std::pair<std::shared_ptr<std::vector<id_t>>, id_t> include_pair_t;

/** Specifies the type of tree search. */
enum class ESearchType { DFS, BFS };

/** \brief The type of a set of dimension selections. */
typedef std::unordered_set<std::vector<size_t>, vector_hasher> proposal_set_t;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
const double GAIN_EPS = 1E-7;
#pragma clang diagnostic pop
const MatCRef<float> FORPY_ZERO_MATR(Mat<float>::Zero(0, 1));
};      // namespace forpy
#endif  // FORPY_TYPES_H_
