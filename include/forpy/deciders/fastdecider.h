/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_DECIDERS_FASTDECIDER_H_
#define FORPY_DECIDERS_FASTDECIDER_H_

#include "../global.h"

#include "../util/serialization/basics.h"

#include <limits>
#include <typeinfo>
#include <utility>
#include <vector>

#include "../types.h"
#include "../util/desk.h"
#include "../util/storage.h"
#include "./idecider.h"

namespace forpy {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
const int DLOG_FD_V = 100;
const size_t LOG_FD_NID = 12043;
const bool LOG_FD_ALLN = true;
#pragma clang diagnostic pop

/**
 * \brief A classifier manager for weak classifiers with a filter function,
 * a feature calculation function and a thresholding.
 *
 * The classifier design is heavily inspired by "Decision Forests for
 * Classification, Regression, Density Estimation, Manifold Learning and
 * Semi-Supervised Learning" (Criminisi, Shotton and Konukoglu, 2011).
 * With their definition, node classifier parameters \f$\theta\f$ can
 * be split into three parts:
 *  - \f$\tau\f$: thresholding parameters for the calculated scalar.
 *
 * With this model, a decision can be made at each node based on whether the
 * calculated scalar lies withing the thresholding bounds.
 *
 * \ingroup forpydecidersGroup
 */
class FastDecider : public IDecider {
 public:
  /**
   * \param threshold_optimizer shared(IThreshOpt)
   *   Optimizes \f$\tau\f$.
   * \param n_valid_features_to_use size_t
   *   The threshold optimizer may hint that
   *   a selected feature may be completely inappropriate for the
   *   currently searched split. If the feature selection provider
   *   does provide sufficiently many features, the classifier may
   *   use the next one and "not count" the inappropriate one.
   *   This is the maximum number of "valid" features that are
   *   used per split. If 0, ignore the flag returned by the
   *   optimizer and always use all suggested feature combinations
   *   provided by the feature selection provider. Default: 0.
   * \param autoscale_valid_features bool
   *   If set to true, automatically scale to sqrt(number of features) of the
   *   input data.
   */
  FastDecider(const std::shared_ptr<IThreshOpt> &threshold_optimizer = nullptr,
              const size_t &n_valid_features_to_use = 0,
              const bool &autoscale_valid_features = false);

  virtual std::shared_ptr<IDecider> create_duplicate(
      const uint &random_seed) const {
    return std::make_shared<FastDecider>(
        threshold_optimizer->create_duplicate(random_seed),
        n_valids_to_use != data_dim && !autoscale_valid_features
            ? n_valids_to_use
            : 0,
        autoscale_valid_features);
  }

  inline bool is_compatible_with(const IDataProvider &dprov) {
    if (n_valids_to_use > dprov.get_feat_vec_dim()) {
      LOG(WARNING) << "`n_valid_features_to_use` is greater than the number of "
                   << "features (" << n_valids_to_use << ">"
                   << dprov.get_feat_vec_dim() << ")! I'm reducing "
                   << "the number accordingly.";
      n_valids_to_use = dprov.get_feat_vec_dim();
    }
    if (n_valids_to_use == 0) {
      if (autoscale_valid_features) {
        n_valids_to_use = std::round(std::sqrt(dprov.get_feat_vec_dim()));
      } else {
        n_valids_to_use = dprov.get_feat_vec_dim();
      }
    }

    if (dprov.get_feat_vec_dim() != data_dim)
      throw ForpyException("Incompatible data provider detected!");
    return true;
  }

  inline void transfer_or_run_check(const std::shared_ptr<IDecider> &other,
                                    IDataProvider *dprov) {
    threshold_optimizer->transfer_or_run_check(other->get_threshopt().get(),
                                               dprov);
    other->set_data_dim(data_dim);
    auto *ot_fd = dynamic_cast<FastDecider *>(other.get());
    if (ot_fd == nullptr)
      other->is_compatible_with(*dprov);
    else
      ot_fd->n_valids_to_use = n_valids_to_use;
  }

  inline void set_data_dim(const size_t &val) { data_dim = val; };

  inline void ensure_capacity(const size_t &n_samples) {
    node_to_featsel.resize(n_samples);
    node_to_thresh_v.match([&n_samples](auto &vec) { vec.resize(n_samples); });
  };

  inline void finalize_capacity(const size_t &size) { ensure_capacity(size); };

  void make_node(const TodoMark &todo_info, const uint &min_samples_at_leaf,
                 const IDataProvider &data_provider, Desk *d) const;

  bool decide(
      const id_t &node_id, const Data<MatCRef> &data_v,
      const std::function<void(void *)> &decision_param_transf = nullptr) const;

  bool supports_weights() const;

  size_t get_data_dim() const;

  std::shared_ptr<IThreshOpt> get_threshopt() const;

  bool operator==(const IDecider &rhs) const;

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const FastDecider &self) {
    stream << "forpy::FastDecider[" << self.node_to_featsel.size()
           << " stored]";
    return stream;
  };
  std::pair<const std::vector<size_t> *,
            const mu::variant<std::vector<float>, std::vector<double>,
                              std::vector<uint32_t>, std::vector<uint8_t>> *>
  get_maps() const;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(cereal::make_nvp("base", cereal::base_class<IDecider>(this)),
       CEREAL_NVP(threshold_optimizer), CEREAL_NVP(n_valids_to_use),
       CEREAL_NVP(autoscale_valid_features), CEREAL_NVP(node_to_featsel),
       CEREAL_NVP(node_to_thresh_v), CEREAL_NVP(data_dim));
  }

  ///////// Utility functions.
  void _make_node__checks(const TodoMark &todo_info,
                          const IDataProvider &data_provider,
                          const uint &min_samples_at_leaf, Desk *d) const;

  void _make_node__opt(const IDataProvider &dprov, Desk *d) const;

  void _make_node__postprocess(const IDataProvider &dprov, Desk *d) const;

  // Fields.
  std::shared_ptr<IThreshOpt> threshold_optimizer;
  size_t n_valids_to_use;
  bool autoscale_valid_features;
  std::vector<size_t> node_to_featsel;
  mu::variant<std::vector<float>, std::vector<double>, std::vector<uint32_t>,
              std::vector<uint8_t>>
      node_to_thresh_v;
  size_t data_dim;
};
};  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::FastDecider);
#endif  // FORPY_DECIDERS_FASTDECIDER_H_
