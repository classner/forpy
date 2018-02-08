#include <forpy/threshold_optimizers/regression_opt.h>
#include <forpy/types.h>

#include <skasort.hpp>

namespace forpy {

RegressionOpt::RegressionOpt(const size_t &n_thresholds,
                             const float &gain_threshold)
    : n_thresholds(n_thresholds), gain_threshold(gain_threshold) {
  if (gain_threshold < REGOPT_EPS)
    throw ForpyException("The minimum gain threshold must be >= " +
                         std::to_string(REGOPT_EPS));
};

void RegressionOpt::check_annotations(IDataProvider *dprov) {
  const auto &annotation_v = dprov->get_annotations();
  const auto &weights_ptr = dprov->get_weights();
  const float *weights_p =
      weights_ptr != nullptr ? &(weights_ptr->at(0)) : nullptr;
  if (weights_ptr != nullptr) {
    for (size_t i = 0; i < dprov->get_n_samples(); ++i) {
      if (weights_p[i] < 0.f)
        throw ForpyException("Invalid weight detected: " +
                             std::to_string(weights_p[i]));
    }
  }
  annotation_v.match(
      [](const Empty &) { throw EmptyException(); },
      [](const auto &annotations) {
        typedef typename get_core<decltype(annotations.data())>::type AT;
        if (typeid(AT) != typeid(float)) {
          throw ForpyException(
              "Regression is only possible with float32 data "
              "and annotations.");
        }
        if (annotations.innerStride() != 1) {
          throw ForpyException(
              "The annotation data must have inner stride 1 (has " +
              std::to_string(annotations.innerStride()) + ")!");
        }
      });
  const auto &data_v = dprov->get_feature(0);
  data_v.match(
    [](const Empty &) { throw EmptyException(); },
    [](const auto &data) {
    typedef typename get_core<decltype(data.data())>::type IT;
    if (typeid(IT) != typeid(float)) {
      throw ForpyException(
          "Regression is only possible with float32 data "
          "and annotations.");
    }
  });
};

void RegressionOpt::full_entropy(const IDataProvider &dprov, Desk *desk) const {
  DeciderDesk *d = &desk->d;
  const auto &annot_mat =
      dprov.get_annotations().get_unchecked<MatCRef<float>>();
  d->annot_p = annot_mat.data();
  d->annot_os = annot_mat.outerStride();
  const auto &weights_ptr = dprov.get_weights();
  const float *weights_p =
      weights_ptr != nullptr ? &(weights_ptr->at(0)) : nullptr;
  DLOG(INFO) << "weights_p: " << weights_p;
  DLOG_IF(INFO, weights_p != nullptr) << "weights_p[0]: " << weights_p[0];
  DLOG_IF(INFO, weights_p != nullptr) << "weights_p[1]: " << weights_p[1];
  float trace = 0.f;  // Important to use a local cache variable here instead of
  // a reference (speed!).
  float full_w = 0.f;
  d->full_sum.resize(d->annot_dim);
  std::fill(d->full_sum.begin(), d->full_sum.end(), 0.f);
  d->full_sum_p = &(d->full_sum[0]);
  const size_t ad = d->annot_dim;
  float *fsp = d->full_sum_p;
  const size_t *eip = d->elem_id_p;
  const size_t annot_os = d->annot_os;
  const float *annot_p = d->annot_p;
  const size_t n_samples = d->n_samples;
  if (weights_p == nullptr) {
    full_w = static_cast<float>(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
      const float *Cp = annot_p + eip[i] * annot_os;
      for (size_t j = 0; j < ad;) {
        trace += *Cp * *Cp;
        fsp[j++] += *(Cp++);
      }
    }
  } else {
    for (size_t i = 0; i < n_samples; ++i) {
      const float *Cp = annot_p + eip[i] * annot_os;
      float current_weight = weights_p[eip[i]];
      full_w += current_weight;
      for (size_t j = 0; j < ad;) {
        float w_y = current_weight * *Cp;
        fsp[j++] += w_y;
        trace += w_y * *(Cp++);
      }
    }
  }
  d->weights_p = weights_p;
  d->full_w = full_w;
  float maxproxy = 0.f;  // Important for efficiency (see above).
  for (size_t idx = 0; idx < ad; ++idx) maxproxy += fsp[idx] * fsp[idx];
  maxproxy /= full_w;
  trace -= maxproxy;
  trace /= full_w;
  d->maxproxy = maxproxy;
  d->fullentropy = trace;
  // Use the trace of the diagonal of the covariance matrix.
  VLOG(57) << "Full variance proxy calculation done. Sum[0]: "
           << d->full_sum_p[0] << ", full proxy: " << maxproxy;
  if (d->sort_perm.size() != d->n_samples) {
    d->sort_perm.resize(d->n_samples);
    d->sort_perm_p = &(d->sort_perm[0]);
    d->elem_ids_sorted.resize(d->n_samples);
    d->elem_ids_sorted_p = &(d->elem_ids_sorted[0]);
    std::iota(d->sort_perm.begin(), d->sort_perm.end(), 0);
  }
  d->feat_values.resize(d->n_samples);
  d->feat_p = &(d->feat_values[0]);
  d->left_sum_vec.resize(d->annot_dim);
  d->left_sum_p = &(d->left_sum_vec[0]);
};

inline void RegressionOpt::optimize__sort(DeciderDesk &d) const {
  float *feat_p = d.feat_p;  // The analyzed feature, only samples at this node.
  size_t *sort_perm_p = d.sort_perm_p;
  const float *full_feat_p =
      d.full_feat_p_v
          .get_unchecked<const float *>();  // See above, but for all samples.
  size_t *elem_id_p = d.elem_id_p;
  const size_t n_samples = d.n_samples;
  for (size_t i = 0; i < d.n_samples; ++i)
    feat_p[i] = full_feat_p[elem_id_p[i]];
  if (!d.presorted) {
    DLOG_IF(INFO,
            DLOG_ROPT_V >= 4 && (d.node_id == LOG_ROPT_NID || LOG_ROPT_ALLN))
        << "Sorting...";
    id_t *elem_ids_sorted_p = d.elem_ids_sorted_p;
    ska_sort(sort_perm_p, sort_perm_p + n_samples,
             [&](const size_t &i1) { return feat_p[i1]; });
    for (size_t w_idx = 0; w_idx < n_samples; ++w_idx) {
      size_t elem_id = sort_perm_p[w_idx];
      size_t sample_id = elem_id_p[elem_id];
      elem_ids_sorted_p[w_idx] = sample_id;
      feat_p[w_idx] = full_feat_p[sample_id];
    }
    std::copy(elem_ids_sorted_p, elem_ids_sorted_p + n_samples, elem_id_p);
  }
  DLOG_IF(INFO,
          DLOG_ROPT_V >= 1 && (d.node_id == LOG_ROPT_NID || LOG_ROPT_ALLN))
      << "First three sorted feature values: " << d.feat_p[0] << ", "
      << d.feat_p[1] << ", " << feat_p[2];
};

inline SplitOptRes<float> &RegressionOpt::optimize__setup(
    DeciderDesk &d) const {
  if (!d.opt_res_v.is<SplitOptRes<float>>())
    d.opt_res_v =
        SplitOptRes<float>{.split_idx = 0,
                           .thresh = std::numeric_limits<float>::lowest(),
                           .gain = 0.f,
                           .valid = false};
  SplitOptRes<float> &ret_res = d.opt_res_v.get_unchecked<SplitOptRes<float>>();
  ret_res.valid = false;
  return ret_res;
};

inline std::unique_ptr<std::vector<float>> RegressionOpt::optimize__thresholds(
    Desk *desk) const {
  if (n_thresholds == 0) {
    return nullptr;
  } else {
    float maxval = desk->d.feat_p[desk->d.n_samples - 1];
    float minval = desk->d.feat_p[0];
    id_t n_thresholds_capped = std::min<size_t>(
        n_thresholds, std::ceil((maxval - minval) / REGOPT_EPS));
    n_thresholds_capped =
        std::min<size_t>(n_thresholds_capped, desk->d.n_samples);
    FASSERT(n_thresholds_capped >= 1);
    auto retvec_up = std::make_unique<std::vector<float>>(n_thresholds_capped);
    std::uniform_real_distribution<float> udist(minval, maxval);
    for (size_t i = 0; i < n_thresholds_capped; ++i)
      retvec_up->at(i) = udist(desk->r.random_engine);
    ska_sort(retvec_up->begin(), retvec_up->end());
    return retvec_up;
  }
};

void RegressionOpt::optimize(Desk *desk) const {
  DeciderDesk &d = desk->d;  // Solely for convenience.
  SplitOptRes<float> &ret_res = this->optimize__setup(d);
  this->optimize__sort(d);
  float *feat_p = d.feat_p;  // The analyzed feature, only samples at this node.
  size_t n_samples = d.n_samples;
  if (feat_p[n_samples - 1] - feat_p[0] <= REGOPT_EPS) {
    DLOG_IF(INFO,
            DLOG_ROPT_V >= 1 && (d.node_id == LOG_ROPT_NID || LOG_ROPT_ALLN))
        << "Not optimizing because min and max features are too close!";
    return;
  }
  std::unique_ptr<std::vector<float>> thresholds = this->optimize__thresholds(desk);
  std::vector<float>::const_iterator feat_val_it;
  if (n_thresholds > 0) feat_val_it = thresholds->begin();
  id_t *elem_id_p = d.elem_id_p;  // The element IDs are global. The pointer
  // points to the first one relevant for this node.
  std::fill(d.left_sum_p, d.left_sum_p + d.annot_dim, 0.f);
  float left_w = 0.f;
  float const *weights_p = d.weights_p;
  const float full_w = d.full_w;
  float current_gain;
  float last_val = std::numeric_limits<float>::lowest(), current_val = 0.f;
  const float *last_ant, *current_ant;
  float maxval = d.feat_p[d.n_samples - 1];
  const size_t ad = d.annot_dim;
  float *lsp = d.left_sum_p;
  const float *fsp = d.full_sum_p;
  const size_t msal = d.min_samples_at_leaf;
  const float *anp = d.annot_p;
  const size_t annot_os = d.annot_os;
  const float maxproxy = d.maxproxy;
  float current_weight, last_weight;
  FASSERT(d.min_samples_at_leaf > 0);
  for (size_t index = 0;
       index < n_samples - msal + 1 &&
       (n_thresholds == 0 || feat_val_it != thresholds->end());
       ++index, left_w += current_weight, last_weight = current_weight,
              last_val = current_val, last_ant = current_ant) {
    if (full_w - left_w <= 0.f) break;
    current_val = feat_p[index];
    current_ant = anp + elem_id_p[index] * annot_os;
    current_weight = weights_p == nullptr ? 1.f : weights_p[elem_id_p[index]];
    DLOG_IF(INFO,
            DLOG_ROPT_V >= 3 && (d.node_id == LOG_ROPT_NID || LOG_ROPT_ALLN))
        << "Current val: " << std::setprecision(17) << current_val
        << ", current ant [0]: " << current_ant[0]
        << ", current weight: " << current_weight;
    if (index > 0) {
      for (size_t idx = 0; idx < ad; ++idx) {
        DLOG_IF(INFO, DLOG_ROPT_V >= 4 &&
                          (d.node_id == LOG_ROPT_NID || LOG_ROPT_ALLN))
            << "Left sum [" << idx << "] += " << last_ant[idx];
        lsp[idx] += last_weight * last_ant[idx];
      }
      DLOG_IF(INFO,
              DLOG_ROPT_V >= 4 && (d.node_id == LOG_ROPT_NID || LOG_ROPT_ALLN))
          << "Feature delta: " << current_val - last_val;
      if (current_val <= last_val + REGOPT_EPS) continue;
      // Check if gain calculation is necessary.
      if ((n_thresholds == 0 ||
           (current_val >= *feat_val_it && last_val < *feat_val_it)) &&
          index >= msal) {
        current_gain = 0.f;
        float proxy_impurity_left = 0., proxy_impurity_right = 0.;
        for (size_t idx = 0; idx < ad; ++idx) {
          proxy_impurity_left += lsp[idx] * lsp[idx];
          float rval = fsp[idx] - lsp[idx];
          proxy_impurity_right += rval * rval;
        }
        DLOG_IF(INFO, DLOG_ROPT_V >= 4 &&
                          (d.node_id == LOG_ROPT_NID || LOG_ROPT_ALLN))
            << "Left sum[0]: " << lsp[0]
            << ", right sum[0]: " << fsp[0] - lsp[0];
        DLOG_IF(INFO, DLOG_ROPT_V >= 4 &&
                          (d.node_id == LOG_ROPT_NID || LOG_ROPT_ALLN))
            << "proxy left: " << proxy_impurity_left
            << ", proxy right: " << proxy_impurity_right;
        float current_proxy = proxy_impurity_left / left_w +
                              proxy_impurity_right / (full_w - left_w);
        current_gain = current_proxy - maxproxy;
        DLOG_IF(INFO, DLOG_ROPT_V >= 3 &&
                          (d.node_id == LOG_ROPT_NID || LOG_ROPT_ALLN))
            << "Current proxy: " << current_proxy
            << ", current gain: " << current_gain
            << ", current best: " << ret_res.gain;
        ret_res.valid = true;
        if (current_gain > ret_res.gain
#ifndef FORPY_SKLEARN_COMPAT
                               + GAIN_EPS
#endif
        ) {
          DLOG_IF(INFO, DLOG_ROPT_V >= 2 &&
                            (d.node_id == LOG_ROPT_NID || LOG_ROPT_ALLN))
              << "New best gain: " << current_gain
              << ", (best former: " << ret_res.gain << ").";
          ret_res.gain = current_gain;
          ret_res.split_idx = index;
        }
      }
    }
    if (maxval <= current_val + REGOPT_EPS) {
      DLOG_IF(INFO,
              DLOG_ROPT_V >= 1 && (d.node_id == LOG_ROPT_NID || LOG_ROPT_ALLN))
          << "Stopping optimization at index " << index
          << " because difference (" << maxval - current_val << ") to max ("
          << maxval << ") is less than " << REGOPT_EPS << ".";
      break;
    }
    if (n_thresholds > 0)
      while (current_val > *feat_val_it) ++feat_val_it;
  }
  if (ret_res.valid) {
    ret_res.thresh =
        (feat_p[ret_res.split_idx] + feat_p[ret_res.split_idx - 1]) / 2.f;
    if (ret_res.thresh ==
        feat_p[ret_res.split_idx])  // Deal with numerical instabilities.
      ret_res.thresh = feat_p[ret_res.split_idx - 1];
    DLOG_IF(INFO,
            DLOG_ROPT_V >= 1 && (d.node_id == LOG_ROPT_NID || LOG_ROPT_ALLN))
        << "Threshold optimized. Best split index: " << ret_res.split_idx
        << ", threshold: " << std::setprecision(17) << ret_res.thresh
        << ", samples left: " << ret_res.split_idx
        << ", samples right: " << n_samples - ret_res.split_idx << ".";
  }
};

bool RegressionOpt::operator==(const IThreshOpt &rhs) const {
  const auto *rhs_c = dynamic_cast<RegressionOpt const *>(&rhs);
  if (rhs_c == nullptr) {
    return false;
  } else {
    bool eq_thresh = n_thresholds == rhs_c->n_thresholds;
    bool eq_gaint = gain_threshold == rhs_c->gain_threshold;
    return eq_thresh && eq_gaint;
  }
};
}  // namespace forpy
