#include <forpy/deciders/fastdecider.h>
#include <forpy/threshold_optimizers/fastclassopt.h>
#include <forpy/threshold_optimizers/ithreshopt.h>

namespace forpy {

FastDecider::FastDecider(const std::shared_ptr<IThreshOpt> &threshold_optimizer,
                         const size_t &n_valid_features_to_use,
                         const bool &autoscale_valid_features)
    : threshold_optimizer(threshold_optimizer),
      n_valids_to_use(n_valid_features_to_use),
      autoscale_valid_features(autoscale_valid_features),
      node_to_featsel(),
      node_to_thresh_v(),
      data_dim(0) {
  if (threshold_optimizer == nullptr)
    this->threshold_optimizer = std::make_shared<FastClassOpt>();
  if (autoscale_valid_features && n_valid_features_to_use != 0)
    throw ForpyException(
        "If autoscaling of valid features is used, "
        "n_valid_features must be set to 0!");
};

void FastDecider::_make_node__checks(const TodoMark &todo_info,
                                     const IDataProvider &data_provider,
                                     const uint &min_samples_at_leaf,
                                     Desk *desk) const {
  FASSERT(todo_info.interv.second > todo_info.interv.first)
  auto &d = desk->d;
  d.n_samples = todo_info.interv.second - todo_info.interv.first;
  d.input_dim = data_provider.get_feat_vec_dim();
  d.annot_dim = data_provider.get_annot_vec_dim();
  d.min_samples_at_leaf = min_samples_at_leaf;
  DLOG_IF(INFO, (todo_info.node_id == LOG_FD_NID || LOG_FD_ALLN))
      << "Checks for node optimization: " << d.n_samples << " samples "
      << "with " << d.input_dim << " features and " << d.annot_dim
      << " annotations.";
  if (data_dim == 0)
    throw ForpyException(
        "This decider hasn't been initialized properly. "
        "Call `set_data_dim` before usage!");
  d.elem_id_p = &(todo_info.sample_ids->at(todo_info.interv.first));
  d.node_id = todo_info.node_id;
  d.start_id = todo_info.interv.first;
  d.end_id = todo_info.interv.second;
}

/**
 * It would be possible to always start the optimization with the feature that
 * the samples are currently sorted by. However, this would bring in a bias
 * towards features and due to the guidelines I favor correctness before speed.
 */
void FastDecider::_make_node__opt(const IDataProvider &dprov,
                                  Desk *desk) const {
  auto &d = desk->d;
  d.best_res_v =
      SplitOptRes<float>{0, std::numeric_limits<float>::lowest(), 0.f, false};
  d.opt_res_v.match([](auto &opt_res) {
    opt_res.gain = 0.f;
    opt_res.valid = false;
  });
  d.need_sort = false;
  uint valids_tried = 0;
  size_t feat_idx;
  float best_gain = 0.f;
  d.presorted = (d.input_dim == 1 && d.node_id > 0);
  id_t draw_idx = d.invalid_counts[d.node_id];
  id_t invalid_count = draw_idx;
  if (d.feature_indices.size() != d.input_dim) {
    VLOG(23) << "Initializing feature index list.";
    d.feature_indices.resize(d.input_dim);
    std::iota(d.feature_indices.begin(), d.feature_indices.end(), 0);
  }
  threshold_optimizer->full_entropy(dprov, desk);
  if (d.fullentropy <= 1E-7) return;
  for (; valids_tried < n_valids_to_use && draw_idx < d.input_dim; ++draw_idx) {
    id_t offset = std::uniform_int_distribution<>(
        0, d.input_dim - draw_idx - 1)(desk->r.random_engine);
    std::swap(d.feature_indices[draw_idx],
              d.feature_indices[draw_idx + offset]);
    feat_idx = d.feature_indices[draw_idx];
    DLOG_IF(INFO, DLOG_FD_V >= 1 && (d.node_id == LOG_FD_NID || LOG_FD_ALLN))
        << "Evaluating feature " << feat_idx << ". This is feature "
        << valids_tried << " of " << n_valids_to_use << " to use and "
        << d.input_dim << " dimensions";
    dprov.get_feature(feat_idx).match(
        [&](const auto &feat_dta) { d.full_feat_p_v = feat_dta.data(); });
    threshold_optimizer->optimize(desk);
    d.opt_res_v.match([&](const auto &opt_res) {
      if (!opt_res.valid) {
        DLOG_IF(INFO,
                DLOG_FD_V >= 1 && (d.node_id == LOG_FD_NID || LOG_FD_ALLN))
            << "Received invalid flag.";
        std::swap(d.feature_indices[draw_idx],
                  d.feature_indices[invalid_count++]);
      } else {
        valids_tried += 1;
        if (opt_res.gain >= best_gain + GAIN_EPS) {
          DLOG_IF(INFO,
                  DLOG_FD_V >= 1 && (d.node_id == LOG_FD_NID || LOG_FD_ALLN))
              << "New best split identified. Old gain: " << best_gain
              << ". New: " << opt_res.gain << ".";
          best_gain = opt_res.gain;
          d.best_res_v = opt_res;
          d.best_feat_idx = feat_idx;
        }
      }
    });
  }
  d.invalid_counts[d.node_id] = invalid_count;
  d.need_sort = feat_idx != d.best_feat_idx;
};

void FastDecider::_make_node__postprocess(const IDataProvider &dprov,
                                          Desk *desk) const {
  auto &d = desk->d;
  d.best_res_v.match([&](auto &best_res) {
    typedef typename get_core<decltype(best_res.thresh)>::type IT;
    const id_t &pivot_id = best_res.split_idx;
    VLOG_IF(20, (d.node_id == LOG_FD_NID || LOG_FD_ALLN))
        << "Best gain found: " << best_res.gain << ", threshold: "
        << threshold_optimizer->get_gain_threshold_for(d.node_id);
    FASSERT(threshold_optimizer->get_gain_threshold_for(d.node_id) >= 0.f);
    if (pivot_id > 0) {
      FASSERT(d.n_samples > pivot_id);
      VLOG_IF(20, (d.node_id == LOG_FD_NID || LOG_FD_ALLN))
          << "For that gain, samples that need to go left: " << pivot_id;
      VLOG_IF(20, (d.node_id == LOG_FD_NID || LOG_FD_ALLN))
          << "For that gain, samples that need to go right: "
          << d.n_samples - pivot_id;
    }
    if (best_res.gain <
            threshold_optimizer->get_gain_threshold_for(d.node_id) ||
        pivot_id < d.min_samples_at_leaf ||
        d.n_samples - pivot_id < d.min_samples_at_leaf) {
      VLOG_IF(20, (d.node_id == LOG_FD_NID || LOG_FD_ALLN))
          << "Suggesting to create a leaf.";
      d.make_to_leaf = true;
    } else {
      VLOG_IF(20, (d.node_id == LOG_FD_NID || LOG_FD_ALLN))
          << "Suggesting to create a split for feature " << d.best_feat_idx;
      d.make_to_leaf = false;
      if (d.node_to_featsel_p->size() == 0) {
        VLOG_IF(20, (d.node_id == LOG_FD_NID || LOG_FD_ALLN))
            << "Initializing threshold map.";
        d.node_to_thresh_v_p->set<std::vector<IT>>();
      }
      FASSERT(d.node_to_featsel_p->size() > d.node_id);
      d.node_to_featsel_p->at(d.node_id) = d.best_feat_idx;
      auto &node_to_thresh = d.node_to_thresh_v_p->get<std::vector<IT>>();
      FASSERT(node_to_thresh.size() > d.node_id);
      node_to_thresh[d.node_id] = best_res.thresh;
      id_t rw_idx = pivot_id;
      if (d.need_sort) {
        VLOG(27) << "Sorting samples.";
        const IT *fullfeat = dprov.get_feature(d.best_feat_idx)
                                 .get_unchecked<VecCMap<IT>>()
                                 .data();
        for (size_t i = 0; i < pivot_id && rw_idx < d.n_samples;) {
          if (fullfeat[d.elem_id_p[i]] > best_res.thresh) {
            std::swap(d.elem_id_p[i], d.elem_id_p[rw_idx++]);
          } else
            i++;
        }
      }
      d.left_int.first = d.start_id;
      d.left_int.second = d.start_id + pivot_id;
      d.right_int.first = d.start_id + pivot_id;
      d.right_int.second = d.end_id;
      FASSERT(d.invalid_counts.size() > d.node_id);
      auto known_invalid = d.invalid_counts[d.node_id];
      d.left_id = desk->t.next_id_p->fetch_add(1);
      d.right_id = desk->t.next_id_p->fetch_add(1);
      FASSERT(d.invalid_counts.size() > d.left_id);
      d.invalid_counts[d.left_id] = known_invalid;
      FASSERT(d.invalid_counts.size() > d.right_id);
      d.invalid_counts[d.right_id] = known_invalid;
    }
  });
}

void FastDecider::make_node(const TodoMark &todo_info,
                            const uint &min_samples_at_leaf,
                            const IDataProvider &data_provider, Desk *d) const {
  _make_node__checks(todo_info, data_provider, min_samples_at_leaf, d);
  _make_node__opt(data_provider, d);
  _make_node__postprocess(data_provider, d);
};

bool FastDecider::decide(
    const id_t &node_id, const Data<MatCRef> &data_v,
    const std::function<void(void *)> & /*decision_param_transf*/) const {
  // Get the decision parameters.
  const auto &featsel = node_to_featsel[node_id];
  bool retval;
  data_v.match(
      [&](const auto &data) {
        typedef typename get_core<decltype(data.data())>::type IT;
        const auto &node_to_thresh = node_to_thresh_v.get<std::vector<IT>>();
        const auto &thresh = node_to_thresh.at(node_id);
        retval = data(0, featsel) <= thresh;
      },
      [](const Empty &) { throw ForpyException("Empty data!"); });
  return retval;
};

bool FastDecider::supports_weights() const {
  return threshold_optimizer->supports_weights();
};

size_t FastDecider::get_data_dim() const {
  if (data_dim == 0) {
    throw ForpyException("This decider hasn't been used yet!");
  }
  return data_dim;
};

std::shared_ptr<IThreshOpt> FastDecider::get_threshopt() const {
  return threshold_optimizer;
};

bool FastDecider::operator==(const IDecider &rhs) const {
  const auto *rhs_c = dynamic_cast<FastDecider const *>(&rhs);
  if (rhs_c == nullptr) {
    return false;
  } else {
    bool eq_valid = n_valids_to_use == rhs_c->n_valids_to_use;
    bool eq_scale = autoscale_valid_features == rhs_c->autoscale_valid_features;
    bool eq_opt = *threshold_optimizer == *(rhs_c->threshold_optimizer);
    bool eq_sfeatsel = node_to_featsel == rhs_c->node_to_featsel;
    bool eq_snts = node_to_thresh_v == rhs_c->node_to_thresh_v;
    bool eq_ddim = data_dim == rhs_c->data_dim;
    return eq_valid && eq_scale && eq_opt && eq_sfeatsel && eq_snts && eq_ddim;
  }
};

std::pair<const std::vector<size_t> *,
          const mu::variant<std::vector<float>, std::vector<double>,
                            std::vector<uint32_t>, std::vector<uint8_t>> *>
FastDecider::get_maps() const {
  return std::make_pair(&node_to_featsel, &node_to_thresh_v);
};

}  // namespace forpy
