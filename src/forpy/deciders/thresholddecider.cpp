
#include <forpy/deciders/thresholddecider.h>
#include <iomanip>

namespace forpy {

  ThresholdDecider::ThresholdDecider() {};

  ThresholdDecider::ThresholdDecider(
    const std::shared_ptr<IThresholdOptimizer> &threshold_optimizer,
    const size_t &n_valid_features_to_use,
    const std::shared_ptr<ISurfaceCalculator> &feature_calculator,
    const int &num_threads,
    const uint &random_seed)
  : feature_calculator(feature_calculator),
    threshold_optimizer(threshold_optimizer),
    n_valids_to_use(n_valid_features_to_use),
    node_to_featsel(),
    node_to_thresh_v(),
    num_threads(num_threads),
    compat_SurfCalc_DProv_checked(false),
    random_seed(random_seed),
    data_dim(0) {
    if (num_threads <= 0) {
      throw Forpy_Exception("The number of threads must be >0!");
    }
#ifndef _OPENMP
    if (num_threads > 1) {
      throw Forpy_Exception("This executable has been built without "
                            "OpenMP support. The number of threads must =1!");
    }
#endif
    if (random_seed == 0) {
      throw Forpy_Exception("Random seed must be > 0!");
    }
    if (n_valids_to_use < 1) {
      throw Forpy_Exception("n_valid_features_to_use must be > 0!");
    }
  };

  void ThresholdDecider::set_data_dim(const size_t &val) {
    data_dim = val;
  }

  std::tuple<bool, elem_id_vec_t, elem_id_vec_t> ThresholdDecider::make_node(
    const node_id_t &node_id,
    const uint &/*node_depth*/,
    const uint &min_samples_at_leaf,
    const elem_id_vec_t &element_id_list,
    const IDataProvider &data_provider) {
    //////////////////////////////////////////
    // Checks.
    if (element_id_list.size() == 0) {
      throw Forpy_Exception("Received an empty element list at a decision node!");
    }
    size_t n_samples = element_id_list.size();
    size_t input_dim = data_provider.get_feat_vec_dim();
    size_t annot_dim = data_provider.get_annot_vec_dim();
    if (feature_calculator->required_num_features() > input_dim) {
      throw Forpy_Exception
        ("The feature calc. input dim (" +
         std::to_string(feature_calculator->required_num_features()) +
         ") is configured to be higher than the data input "
         "dimension (" + std::to_string(input_dim) + "!");
    }
    if (! compat_SurfCalc_DProv_checked) {
      if (! feature_calculator->is_compatible_to(data_provider)) {
        throw Forpy_Exception("Incompatible feature calculator and data provider!");
      }
      threshold_optimizer->check_annotations(data_provider);
      compat_SurfCalc_DProv_checked = true;
    }
    if (data_dim == 0) {
      throw Forpy_Exception("This decider hasn't been initialized properly. "
                            "Call `set_data_dim` before usage!");
    }
    if (input_dim != data_dim) {
      throw Forpy_Exception("Incompatible data provider detected!");
    }
    ////////////////////////////////////////////////////////////////////////////
    // Get the samples and start processing.
    const auto &sample_list_v = data_provider.get_samples();
    auto ret_tpl = std::make_tuple<bool, elem_id_vec_t, elem_id_vec_t>(
        false, elem_id_vec_t(), elem_id_vec_t());
    sample_list_v.match(
        [&](const auto &sample_vec) {
          typedef typename get_core<decltype(sample_vec[0].data[0])>::type IT;
          typedef typename get_core<decltype(sample_vec[0].annotation[0])>::type AT;
          if (node_to_featsel.size() == 0) {
            node_to_thresh_v.set<std::unordered_map<node_id_t, IT>>();
          }
          //////////////////////////////////////////
          // Setup.
          // Create the annotation matrix.
          Mat<AT> annotations(n_samples, annot_dim);
          for (size_t s_idx = 0; s_idx < n_samples; ++s_idx) {
            annotations.block(s_idx, 0, 1, annot_dim) = \
              sample_vec[element_id_list[s_idx]].annotation;
          }
          // Feature selector.
          if (feature_calculator->required_num_features() != 1) {
            throw Forpy_Exception("Implementation must be improved here "
                                  "to support multiple inputs for features.");
          }
          // Prepare shuffled indices to test. It's required to prepare all of
          // them to be potentially tested, since it's a priori unknown how many
          // will be used (`invalid` results can lead to higher numbers).
          std::mt19937 node_engine(random_seed + node_id);
          std::vector<size_t> shuffled_indices(input_dim);
          std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
          std::shuffle(shuffled_indices.begin(),
                       shuffled_indices.end(),
                       node_engine);
          // The data matrix that will be used.
          auto data_p = std::make_shared<MatCM<IT>>(
              n_samples,
              feature_calculator->required_num_features());
          uint valids_tried = 0;
          size_t current_feat_id = 0;
          float best_gain = 0.f;
          optimized_split_tuple_t<IT> best_tpl;
          std::vector<size_t> best_feats;
          std::vector<size_t> feat_ids(1);
          while (valids_tried < n_valids_to_use) {
            feat_ids[0] = shuffled_indices[current_feat_id];
            size_t feat_id = feat_ids[0];
            IT* dp = data_p->data();
            for (size_t s_idx = 0; s_idx < n_samples; ++s_idx) {
              dp[s_idx] = sample_vec[element_id_list[s_idx]].data.data()[feat_id];
            }
            std::shared_ptr<const MatCM<IT>> feature_p;
            std::shared_ptr<const MatCM<IT>> __data_in = std::const_pointer_cast<const MatCM<IT>>(data_p);
            feature_calculator->calculate(__data_in,
                                          feature_p,
                                          feat_ids,
                                          sample_list_v,
                                          element_id_list);
            auto opt_res = threshold_optimizer->optimize(*__data_in,
                                                         feature_p->col(0),
                                                         annotations,
                                                         node_id,
                                                         min_samples_at_leaf);
            current_feat_id++;
            if (! std::get<5>(opt_res)) {
              // Not valid.
              VLOG(20) << "Received invalid flag.";
              continue;
            } else {
              valids_tried += 1;
              VLOG(25) << "Received valid result.";
            }
            float gain = std::get<4>(opt_res);
            VLOG(25) << "Gain: " << std::setprecision(17) << gain;
            if (gain >= best_gain + GAIN_EPS) {
              VLOG(24) << "New best gain split found. Old gain: " << best_gain
                       << ". New: " << gain;
              best_gain = gain;
              best_tpl = opt_res;
              best_feats = feat_ids;
            }
          }
          if (best_gain < threshold_optimizer->get_gain_threshold_for(node_id) ||
              std::get<2>(best_tpl) < min_samples_at_leaf ||
              std::get<3>(best_tpl) < min_samples_at_leaf) {
            VLOG(20) << "Suggesting to create a leaf. Best gain found: "
                     << best_gain;
            VLOG(20) << "For that gain, samples that needed to go left: "
                     << std::get<2>(best_tpl);
            VLOG(20) << "For that gain, samples that needed to go right: "
                     << std::get<3>(best_tpl);
            std::get<0>(ret_tpl) = true;
          } else {
            VLOG(20) << "Suggesting to create a split.";
            std::get<0>(ret_tpl) = false;
            auto ret = node_to_featsel.emplace(node_id,
                                               best_feats);
            if (! ret.second) {
              throw Forpy_Exception("Tried to recreate a node with existing "
                                    "parameters: id " + std::to_string(node_id));
            }
            auto &node_to_thresh = node_to_thresh_v.get<std::unordered_map<node_id_t, IT>>();
            auto ret_2 = node_to_thresh.emplace(node_id,
                                                std::get<0>(best_tpl).first);
            if (! ret_2.second) {
              throw Forpy_Exception("Internal error.");
            }
            // Create the element lists.
            auto element_list_left = &std::get<1>(ret_tpl);
            auto element_list_right = &std::get<2>(ret_tpl);
            const auto *fc = dynamic_cast<AlignedSurfaceCalculator const *>
              (feature_calculator.get());
            if (fc == nullptr) {
              for (const auto &element_id : element_id_list) {
                if (this->decide(node_id,
                                 sample_vec[element_id].data,
                                 data_provider.get_decision_transf_func(element_id))) {
                  element_list_left -> push_back(element_id);
                } else {
                  element_list_right -> push_back(element_id);
                }
              }
            } else {
              size_t feat_id = best_feats[0];
              for (const auto &element_id : element_id_list) {
                if (sample_vec[element_id].data.data()[feat_id] <
                    std::get<0>(best_tpl).first) {
                  element_list_left -> push_back(element_id);
                } else {
                  element_list_right -> push_back(element_id);
                }
              }
            }
            // Check.
            FASSERT(std::get<2>(best_tpl) == element_list_left->size() &&
                    std::get<3>(best_tpl) == element_list_right->size());
            FASSERT(element_list_left->size() +
                    element_list_right->size() == element_id_list.size());
          }
        });
    return ret_tpl;
  };

  bool ThresholdDecider::decide
    (const node_id_t &node_id,
     const Data<MatCRef> &data_v,
     const std::function<void(void*)> &/*decision_param_transf*/)
   const {
    // Get the decision parameters.
    auto ntf_pos = node_to_featsel.find(node_id);
    if (ntf_pos == node_to_featsel.end()) {
      throw Forpy_Exception("No decision stored for node ID " +
                            std::to_string(node_id));
    }
    const auto &featsel = ntf_pos->second;
    bool retval;
    data_v.match([&](const auto &data){
        typedef typename get_core<decltype(data.data())>::type IT;
        const auto &node_to_thresh = node_to_thresh_v.get<std::unordered_map<node_id_t, IT>>();
        const auto &thresh = node_to_thresh.at(node_id);
        // Calc feature.
        IT ftval;
        feature_calculator->calculate_pred(data,
                                           &ftval,
                                           featsel);
        retval = ftval < thresh;
      },
      [](const Empty &) { throw Forpy_Exception("Empty data!"); });
    return retval;
  };

  bool ThresholdDecider::supports_weights() const {
   return threshold_optimizer -> supports_weights();
  };

  size_t ThresholdDecider::get_data_dim() const {
    if (data_dim == 0) {
      throw Forpy_Exception("This decider hasn't been used yet!");
    }
    return data_dim;
  };

  std::shared_ptr<IThresholdOptimizer> ThresholdDecider::get_threshopt() const {
    return threshold_optimizer;
  };

  std::shared_ptr<ISurfaceCalculator> ThresholdDecider::get_featcalc() const {
    return feature_calculator;
  };

  bool ThresholdDecider::operator==(const IDecider &rhs) const {
    const auto *rhs_c = dynamic_cast<ThresholdDecider const *>(&rhs);
    if (rhs_c == nullptr) {
      return false;
    } else {
      bool eq_surf = *feature_calculator == *(rhs_c -> feature_calculator);
      bool eq_valid = n_valids_to_use == rhs_c -> n_valids_to_use;
      bool eq_opt = *threshold_optimizer == *(rhs_c -> threshold_optimizer);
      bool eq_sfeatsel = node_to_featsel == rhs_c->node_to_featsel;
      bool eq_snts = node_to_thresh_v == rhs_c->node_to_thresh_v;
      bool eq_seed = random_seed == rhs_c->random_seed;
      bool eq_ddim = data_dim == rhs_c->data_dim;
      return eq_surf && eq_valid && eq_opt && eq_sfeatsel && eq_snts &&
        eq_seed && eq_ddim;
    }
  }; 

  std::pair<const std::unordered_map<node_id_t, std::vector<size_t>> *,
            const mu::variant<std::unordered_map<node_id_t, float>,
                              std::unordered_map<node_id_t, double>,
                              std::unordered_map<node_id_t, uint32_t>,
                              std::unordered_map<node_id_t, uint8_t>> *>
    ThresholdDecider::get_maps() const {
    return std::make_pair(&node_to_featsel, &node_to_thresh_v);
  };

} // namespace forpy
