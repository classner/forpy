#include <forpy/threshold_optimizers/fastclassopt.h>
#include <forpy/types.h>

#include <skasort.hpp>

namespace forpy {

FastClassOpt::FastClassOpt(const size_t &n_thresholds,
                           const float &gain_threshold)
    : ClassificationOpt(n_thresholds, gain_threshold){};

void FastClassOpt::full_entropy(const IDataProvider &dprov, Desk *desk) const {
  DeciderDesk *d = &desk->d;
  const auto &annot_v = dprov.get_annotations();
  const auto &weights_ptr = dprov.get_weights();
  const float *weights_p =
      weights_ptr != nullptr ? &(weights_ptr->at(0)) : nullptr;
  DLOG(INFO) << "weights_p: " << weights_p;
  DLOG_IF(INFO, weights_p != nullptr) << "weights_p[0]: " << weights_p[0];
  DLOG_IF(INFO, weights_p != nullptr) << "weights_p[1]: " << weights_p[1];
  annot_v.match(
      [&](const MatCRef<uint> &annot_mat) {
        d->annot_os = annot_mat.outerStride();
        d->class_annot_p = annot_mat.data();
        d->full_sum.resize(n_classes);
        std::fill(d->full_sum.begin(), d->full_sum.end(), 0.f);
        d->full_sum_p = &(d->full_sum[0]);
        const uint *cap = d->class_annot_p;
        const size_t *eip = d->elem_id_p;
        float *fsp = d->full_sum_p;
        if (weights_p != nullptr)
          for (size_t i = 0; i < d->n_samples; ++i)
            fsp[cap[eip[i]]] += weights_p[eip[i]];
        else
          for (size_t i = 0; i < d->n_samples; ++i) fsp[cap[eip[i]]]++;
        float full_w = 0.f;
        for (size_t i = 0; i < n_classes; ++i) full_w += fsp[i];
        d->full_w = full_w;
      },
      [&](const auto &) { throw EmptyException(); });
  d->weights_p = weights_p;
  float sqsum = 0.f;
  float *fsp = d->full_sum_p;
  for (size_t i = 0; i < n_classes; ++i) {
    DLOG(INFO) << "Class " << i << " weight: " << fsp[i];
    sqsum += fsp[i] * fsp[i];
  }
  d->fullentropy = 1.f - sqsum / (d->full_w * d->full_w);
  VLOG(57) << "Full entropy calculation for " << n_classes
           << " classes done: " << d->fullentropy;
  VLOG(57) << "Full weight: " << d->full_w;
  if (d->sort_perm.size() != d->n_samples) {
    d->sort_perm.resize(d->n_samples);
    d->sort_perm_p = &(d->sort_perm[0]);
    d->elem_ids_sorted.resize(d->n_samples);
    d->elem_ids_sorted_p = &(d->elem_ids_sorted[0]);
    std::iota(d->sort_perm.begin(), d->sort_perm.end(), 0);
  }
  const auto &feat_v = dprov.get_feature(0);
  feat_v.match([&](const auto &feat) {
    typedef typename get_core<decltype(feat.data())>::type IT;
    if (!d->class_feat_values.is<std::vector<IT>>())
      d->class_feat_values.set<std::vector<IT>>(d->n_samples);
    auto &feat_vec = d->class_feat_values.get_unchecked<std::vector<IT>>();
    feat_vec.resize(d->n_samples);
  });
  d->left_sum_vec.resize(n_classes);
  d->left_sum_p = &(d->left_sum_vec[0]);
};

template <typename IT>
inline void FastClassOpt::optimize__sort(DeciderDesk &d) const {
  IT *feat_p = &d.class_feat_values.get_unchecked<std::vector<IT>>()[0];
  size_t *sort_perm_p = d.sort_perm_p;
  const IT *full_feat_p =
      d.full_feat_p_v
          .get_unchecked<const IT *>();  // See above, but for all samples.
  size_t *elem_id_p = d.elem_id_p;
  const size_t n_samples = d.n_samples;
  for (size_t i = 0; i < d.n_samples; ++i)
    feat_p[i] = full_feat_p[elem_id_p[i]];
  if (!d.presorted) {
    DLOG_IF(INFO,
            DLOG_FCOPT_V >= 4 && (d.node_id == LOG_FCOPT_NID || LOG_FCOPT_ALLN))
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
          DLOG_FCOPT_V >= 1 && (d.node_id == LOG_FCOPT_NID || LOG_FCOPT_ALLN))
      << "First three sorted feature values: " << feat_p[0] << ", " << feat_p[1]
      << ", " << feat_p[2];
};

template <typename IT>
inline SplitOptRes<IT> &FastClassOpt::optimize__setup(DeciderDesk &d) const {
  if (!d.opt_res_v.is<SplitOptRes<IT>>())
    d.opt_res_v = SplitOptRes<IT>{.split_idx = 0,
                                  .thresh = std::numeric_limits<IT>::lowest(),
                                  .gain = 0.f,
                                  .valid = false};
  SplitOptRes<IT> &ret_res = d.opt_res_v.get_unchecked<SplitOptRes<IT>>();
  ret_res.valid = false;
  return ret_res;
};

template <typename IT>
inline std::unique_ptr<std::vector<IT>> FastClassOpt::optimize__thresholds(
    Desk *desk) const {
  if (n_thresholds == 0) {
    return nullptr;
  } else {
    IT maxval = desk->d.feat_p[desk->d.n_samples - 1];  // TODO fix.
    IT minval = desk->d.feat_p[0];
    id_t n_thresholds_capped = std::min<size_t>(
        n_thresholds, std::ceil((maxval - minval) / CLASSOPT_EPS));
    n_thresholds_capped =
        std::min<size_t>(n_thresholds_capped, desk->d.n_samples);
    FASSERT(n_thresholds_capped >= 1);
    auto retvec_up = std::make_unique<std::vector<IT>>(n_thresholds_capped);
    std::uniform_real_distribution<float> udist(minval, maxval);
    for (size_t i = 0; i < n_thresholds_capped; ++i)
      retvec_up->at(i) = static_cast<IT>(udist(desk->r.random_engine));
    ska_sort(retvec_up->begin(), retvec_up->end());
    return retvec_up;
  }
};

void FastClassOpt::optimize(Desk *desk) const {
  DeciderDesk &d = desk->d;  // Solely for convenience.
  d.class_feat_values.match([&](auto &class_feats) {
    typedef typename get_core<decltype(class_feats.data())>::type IT;
    IT *feat_p = &class_feats[0];
    SplitOptRes<IT> &ret_res = this->optimize__setup<IT>(d);
    this->optimize__sort<IT>(d);
    size_t n_samples = d.n_samples;
    if (feat_p[n_samples - 1] - feat_p[0] <= CLASSOPT_EPS) {
      DLOG_IF(INFO, DLOG_FCOPT_V >= 1 &&
                        (d.node_id == LOG_FCOPT_NID || LOG_FCOPT_ALLN))
          << "Not optimizing because min and max features are too close!";
      return;
    }
    std::unique_ptr<std::vector<IT>> thresholds =
        this->optimize__thresholds<IT>(desk);
    typename std::vector<IT>::const_iterator feat_val_it;
    if (n_thresholds > 0) feat_val_it = thresholds->begin();
    id_t *elem_id_p = d.elem_id_p;  // The element IDs are global. The pointer
    // points to the first one relevant for this node.
    std::fill(d.left_sum_p, d.left_sum_p + n_classes, 0.f);
    float left_w = 0.f;
    float const *weights_p = d.weights_p;
    const float full_w = d.full_w;
    float current_gain;
    IT last_val = std::numeric_limits<IT>::lowest(), current_val = 0;
    const uint *last_ant, *current_ant;
    IT maxval = feat_p[d.n_samples - 1];
    float *lsp = d.left_sum_p;
    float *fsp = &d.full_sum[0];
    DLOG_IF(INFO,
            DLOG_FCOPT_V >= 1 && (d.node_id == LOG_FCOPT_NID || LOG_FCOPT_ALLN))
        << "fsp[0]: " << fsp[0] << ", fsp[1]: " << fsp[1];
    float lssq = 0.f, rssq = 0.f;
    double dlssq = 0., drssq = 0.;
    float ent_left, ent_right;
    // As long as we have <5000 samples, tracking the sum of squares works
    // reliably wrt. numerics. If there are more, always do the live
    // calculation.
    if (n_samples < 5000)
      for (size_t i = 0; i < n_classes; ++i) rssq += (fsp[i] * fsp[i]);
    DLOG_IF(INFO,
            DLOG_FCOPT_V >= 1 && (d.node_id == LOG_FCOPT_NID || LOG_FCOPT_ALLN))
        << "rssq: " << std::setprecision(17) << rssq;
    const size_t msal = d.min_samples_at_leaf;
    const uint *anp = d.class_annot_p;
    float new_val;
    float current_weight, last_weight;
    FASSERT(d.min_samples_at_leaf > 0);
    for (size_t index = 0;
         index < n_samples - msal + 1 &&
         (n_thresholds == 0 || feat_val_it != thresholds->end());
         ++index, left_w += current_weight, last_weight = current_weight,
                last_val = current_val, last_ant = current_ant) {
      if (full_w - left_w <= 0.f) break;
      current_val = feat_p[index];
      current_ant = anp + elem_id_p[index];
      current_weight = weights_p == nullptr ? 1.f : weights_p[elem_id_p[index]];
      DLOG_IF(INFO, DLOG_FCOPT_V >= 3 &&
                        (d.node_id == LOG_FCOPT_NID || LOG_FCOPT_ALLN))
          << "Current val: " << std::setprecision(17) << current_val
          << ", current ant: " << current_ant[0]
          << ", current weight: " << current_weight;
      if (index > 0) {
        float old_val = lsp[last_ant[0]];
        lsp[last_ant[0]] += last_weight;
        DLOG_IF(INFO, DLOG_FCOPT_V >= 7 &&
                          (d.node_id == LOG_FCOPT_NID || LOG_FCOPT_ALLN))
            << "Updating weight for class " << last_ant[0] << " with weight "
            << last_weight << " to " << lsp[last_ant[0]];
        if (n_samples < 5000) {
          new_val = old_val + last_weight;
          lssq = lssq - old_val * old_val + new_val * new_val;
          old_val = fsp[last_ant[0]] - old_val;
          new_val = old_val - last_weight;
          rssq = rssq - old_val * old_val + new_val * new_val;
        }
        DLOG_IF(INFO, DLOG_FCOPT_V >= 4 &&
                          (d.node_id == LOG_FCOPT_NID || LOG_FCOPT_ALLN))
            << "Feature delta: " << current_val - last_val;
        if (current_val <= last_val + CLASSOPT_EPS) continue;
        // Check if gain calculation is necessary.
        if ((n_thresholds == 0 ||
             (current_val >= *feat_val_it && last_val < *feat_val_it)) &&
            index >= msal) {
          if (n_samples >= 5000) {
            dlssq = 0.;
            for (size_t i = 0; i < n_classes; ++i) {
              // DLOG(INFO) << "lsp[i]: " << lsp[i];
              double tmp = static_cast<double>(lsp[i]);
              dlssq += tmp * tmp;
            }
            ent_left = 1.f - dlssq / (left_w * left_w);
            drssq = 0.;
            for (size_t i = 0; i < n_classes; ++i) {
              double tmp = static_cast<double>(fsp[i] - lsp[i]);
              drssq += tmp * tmp;
            }
            ent_right = 1.f - drssq / ((full_w - left_w) * (full_w - left_w));
          } else {
            ent_left = 1.f - lssq / (left_w * left_w);
            ent_right = 1.f - rssq / ((full_w - left_w) * (full_w - left_w));
          }
          FASSERT(ent_right >= -0.01 && ent_right <= 1.01);
          DLOG_IF(INFO, DLOG_FCOPT_V >= 5 &&
                            (d.node_id == LOG_FCOPT_NID || LOG_FCOPT_ALLN))
              << "Left entropy: " << ent_left
              << ", right entropy: " << ent_right << ", lssq: " << lssq
              << ", rssq: " << rssq << ", dlssq: " << dlssq
              << ", drssq: " << drssq << ", full_w: " << full_w
              << ", left_w: " << left_w;
          current_gain = d.fullentropy - left_w / full_w * ent_left -
                         (full_w - left_w) / full_w * ent_right;
          DLOG_IF(INFO, DLOG_FCOPT_V >= 5 &&
                            (d.node_id == LOG_FCOPT_NID || LOG_FCOPT_ALLN))
              << "gain: " << current_gain;
          DLOG_IF(INFO, DLOG_FCOPT_V >= 4 &&
                            (d.node_id == LOG_FCOPT_NID || LOG_FCOPT_ALLN))
              << "Left sum[0]: " << lsp[0] << ", left sum[1]: " << lsp[1];
          ret_res.valid = true;
          if (current_gain > ret_res.gain
#ifndef FORPY_SKLEARN_COMPAT
                                 + GAIN_EPS
#endif
          ) {
            DLOG_IF(INFO, DLOG_FCOPT_V >= 2 &&
                              (d.node_id == LOG_FCOPT_NID || LOG_FCOPT_ALLN))
                << "New best gain: " << current_gain
                << ", (best former: " << ret_res.gain << ").";
            ret_res.gain = current_gain;
            ret_res.split_idx = index;
          }
        }
      }
      if (maxval <= current_val + CLASSOPT_EPS) {
        DLOG_IF(INFO, DLOG_FCOPT_V >= 1 &&
                          (d.node_id == LOG_FCOPT_NID || LOG_FCOPT_ALLN))
            << "Stopping optimization at index " << index
            << " because difference (" << maxval - current_val << ") to max ("
            << maxval << ") is less than " << CLASSOPT_EPS << ".";
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
      DLOG_IF(INFO, DLOG_FCOPT_V >= 1 &&
                        (d.node_id == LOG_FCOPT_NID || LOG_FCOPT_ALLN))
          << "Threshold optimized. Best split index: " << ret_res.split_idx
          << ", threshold: " << std::setprecision(17) << ret_res.thresh
          << ", samples left: " << ret_res.split_idx
          << ", samples right: " << n_samples - ret_res.split_idx << ".";
    }
  });
};

bool FastClassOpt::operator==(const IThreshOpt &rhs) const {
  const auto *rhs_c = dynamic_cast<FastClassOpt const *>(&rhs);
  if (rhs_c == nullptr) {
    return false;
  } else {
    bool eq_thresh = n_thresholds == rhs_c->n_thresholds;
    bool eq_gaint = gain_threshold == rhs_c->gain_threshold;
    return eq_thresh && eq_gaint;
  }
};
}  // namespace forpy
