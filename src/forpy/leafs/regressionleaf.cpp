#include <forpy/leafs/regressionleaf.h>

namespace forpy {

  RegressionLeaf::RegressionLeaf(
      const std::shared_ptr<IRegressor> &regressor_template,
      const uint &summary_mode,
      const size_t &regression_input_dim,
      const size_t &selections_to_try,
      const int &num_threads,
      const uint &random_seed)
    : reg_calc_template(regressor_template),
      summary_mode(summary_mode),
      regression_input_dim(regression_input_dim),
      selections_to_try(selections_to_try),
      num_threads(num_threads),
      random_seed(random_seed),
      leaf_regression_map(std::unordered_map<node_id_t,
                                             std::pair<std::unique_ptr<IRegressor>,
                                                       std::vector<size_t>>>()),
      max_input_dim(0),
      annot_dim(0) {
    if (num_threads <= 0) {
      throw Forpy_Exception("The number of threads must be >0!");
    }
    if (regression_input_dim > 0) {
      if (selections_to_try == 0) {
        throw Forpy_Exception("The number of selections to try must be >0!");
      }
#ifndef _OPENMP
      if (num_threads > 1) {
        throw Forpy_Exception("This executable has been built without "
                              "OpenMP support. The number of threads must =1!");
      }
#endif
      if (! regressor_template->needs_input_data()) {
        throw Forpy_Exception("Number of input features set, but selected "
                              "regressor doesn't use input data!");
      }
    }
    if (summary_mode > 2) {
      throw Forpy_Exception("Unknown summary mode (supported: 0,1,2)!");
    }
    if (random_seed == 0) {
      throw Forpy_Exception("Random seed must be >0!");
    }
    regressor_template->freeze();
  };

  bool RegressionLeaf::is_compatible_with(const IDataProvider &data_provider) {
    max_input_dim = data_provider.get_feat_vec_dim();
    if (regression_input_dim == 0) {
      regression_input_dim = max_input_dim;
    }
    annot_dim = data_provider.get_annot_vec_dim();
    const auto &sample_list_v = data_provider.get_samples();
    sample_list_v.match(
                        [&](const auto &sample_vec) {
                          typedef typename get_core<decltype(sample_vec[0].data[0])>::type IT;
                          typedef typename get_core<decltype(sample_vec[0].annotation[0])>::type AT;
                          if (typeid(IT) == typeid(double) ||
                              typeid(AT) == typeid(double)) {
                            double_mode = true;
                          } else {
                            double_mode = false;
                          }
                        });
    if (max_input_dim == 0)
      return false;
    else
      return true;
  };

  bool RegressionLeaf::is_compatible_with(const IThresholdOptimizer &threshopt) { return true; };

  bool RegressionLeaf::needs_data() const { return true; };

  void RegressionLeaf::make_leaf(
      const node_id_t &node_id,
      const elem_id_vec_t &element_list,
      const IDataProvider &data_provider) {
    //////////////////////////////////////////
    // Checks.
    if (element_list.size() == 0) {
      throw Forpy_Exception("Received an empty element list at a leaf!");
    }
    if (max_input_dim == 0) {
      throw Forpy_Exception("This regression leaf has not been initialized "
                            "yet by calling `is_compatible_with` with the "
                            "data provider!");
    }
    if (data_provider.get_feat_vec_dim() != max_input_dim) {
      throw Forpy_Exception("The data provider data dimension does not "
                            "agree with the one obtained from the compat. "
                            "check!");
    }
    size_t n_samples = element_list.size();
    size_t input_dim = data_provider.get_feat_vec_dim();
    size_t annot_dim = data_provider.get_annot_vec_dim();
    if (regression_input_dim > input_dim) {
      throw Forpy_Exception("The regression input dim (" +
                            std::to_string(regression_input_dim) + ") is "
                            "configured to be higher than the data input "
                            "dimension (" + std::to_string(input_dim) +
                            "!");
    }
    //////////////////////////////////////////
    // Setup.
    // Create the annotation matrix.
    auto sample_list_v = data_provider.get_samples();
    RegData<Mat> annotation_v;
    //std::cout << "making annotmat" << std::endl;
    sample_list_v.match(
        [&](const auto &sample_vec) {
          typedef typename get_core<decltype(sample_vec[0].data[0])>::type IT;
          typedef typename get_core<decltype(sample_vec[0].annotation[0])>::type AT;
          if (typeid(IT) == typeid(double) ||
              typeid(AT) == typeid(double)) {
            if (! double_mode) {
              throw Forpy_Exception("Invalid data mode!");
            }
            annotation_v.set<Mat<double>>(n_samples, annot_dim);
            auto &annotation = annotation_v.get_unchecked<Mat<double>>();
            for (size_t s_idx = 0; s_idx < n_samples; ++s_idx) {
              annotation.block(s_idx, 0, 1, annot_dim) = \
                sample_vec[element_list[s_idx]].annotation.template cast<double>();
            }
          } else {
            // float.
            if (double_mode) {
              throw Forpy_Exception("Invalid data mode!");
            } 
            annotation_v.set<Mat<float>>(Mat<float>::Zero(n_samples,
                                                          annot_dim));
            auto &annotation = annotation_v.get_unchecked<Mat<float>>();
            for (size_t s_idx = 0; s_idx < n_samples; ++s_idx) {
              annotation.block(s_idx, 0, 1, annot_dim) = \
                sample_vec[element_list[s_idx]].annotation.template cast<float>();
            }
          }
        });
    //std::cout << "annotmat done" << std::endl;
    // Setup the feature sets to try.
    proposal_set_t to_try;
    if (regression_input_dim == max_input_dim) {
      std::vector<size_t> fullfeat(max_input_dim);
      std::iota(fullfeat.begin(), fullfeat.end(), static_cast<size_t>(0));
      to_try.emplace(fullfeat);
    } else {
      if (selections_to_try == max_input_dim &&
          regression_input_dim == 1) {
        // try all.
        for (size_t i = 0; i < max_input_dim; ++i) {
          to_try.emplace(std::vector<size_t>{i});
        }
      } else {
        FeatureSelector featsel(selections_to_try,
                                regression_input_dim,
                                max_input_dim, 0,
                                random_seed + node_id);
        to_try = featsel.get_proposals();
      }
    }
    float best_residual = std::numeric_limits<float>::infinity();
    std::unique_ptr<IRegressor> best_regr;
    std::vector<size_t> best_features;
    /////////////////////////////////////////
    // Work.
    auto featsel_it = to_try.begin();
    for (size_t featsel_idx = 0; featsel_idx < to_try.size();
         ++featsel_idx, ++featsel_it) {
      // Create the regressor.
      auto regr = this->reg_calc_template->empty_duplicate();
      // Prepare the input data.
      RegData<Mat> input_v;
      if (regr->needs_input_data()) {
        sample_list_v.match([&](const auto &sample_vec) {
            if (double_mode) {
              input_v.set<Mat<double>>(Mat<double>::Ones(n_samples,
                                                         regression_input_dim + 1));
              auto &input = input_v.get_unchecked<Mat<double>>();
              for (size_t s_idx = 0; s_idx < n_samples; ++s_idx) {
                for (size_t feat_idx_idx = 0;
                     feat_idx_idx < regression_input_dim;
                     ++feat_idx_idx) {
                  input(s_idx, 1 + feat_idx_idx) = \
                    static_cast<double>(\
             sample_vec[element_list[s_idx]].data[(*featsel_it)[feat_idx_idx]]);
                }
              }
            } else {
              input_v.set<Mat<float>>(Mat<float>::Ones(n_samples,
                                                         regression_input_dim + 1));
              auto &input = input_v.get_unchecked<Mat<float>>();
              for (size_t s_idx = 0; s_idx < n_samples; ++s_idx) {
                for (size_t feat_idx_idx = 0;
                     feat_idx_idx < regression_input_dim;
                     ++feat_idx_idx) {
                  input(s_idx, 1 + feat_idx_idx) = \
                    static_cast<float>(\
             sample_vec[element_list[s_idx]].data[(*featsel_it)[feat_idx_idx]]);
                }
              }
            }
          });
      } else {
        // Fake it.
        if (double_mode) {
          input_v.set<Mat<double>>(Mat<double>::Zero(0, 0));
        } else {
          input_v.set<Mat<float>>(Mat<float>::Zero(0, 0));
        }
      }
      // Let the regressor work.
      if (double_mode) {
        const auto &input = input_v.get_unchecked<Mat<double>>();
        const auto &annot = annotation_v.get_unchecked<Mat<double>>();
        regr->initialize_nocopy(input, annot);
      } else {
        const auto &input = input_v.get_unchecked<Mat<float>>();
        const auto &annot = annotation_v.get_unchecked<Mat<float>>();
        regr->initialize_nocopy(input, annot);          
      }
      regr->freeze();
      if (! regr->has_solution()) {
        continue;
      }
      float regr_err = regr->get_residual_error();
      if (regr_err <= best_residual - GAIN_EPS) {
        best_regr.swap(regr);
        best_features = *featsel_it;
        best_residual = regr_err;
      }
    }
    if (best_regr == nullptr) {
      throw Forpy_Exception("Could not find a regression solution for this "
                            "leaf! (ID: " + std::to_string(node_id) + ")");
    }
    if (! best_regr->get_frozen()) {
      throw Forpy_Exception("internal error: regressor not frozen before "
                            "stored.");
    }
    auto retval = leaf_regression_map.emplace(node_id,
                                              std::make_pair(std::move(best_regr),
                                                             std::move(best_features)));
    if (! retval.second) {
      throw Forpy_Exception("Tried to create the leaf value for a node with "
                            "an existing one! Node ID: " +
                            std::to_string(node_id));
    }
  };

  /** Gets the number of summary dimensions per sample. */
  size_t RegressionLeaf::get_result_columns(const size_t &n_trees) const {
    if (annot_dim == 0) {
      throw Forpy_Exception("This leaf has not been initialized yet!");
    }
    if (summary_mode == 2) {
      return static_cast<int>(2 * annot_dim * n_trees);
    } else {
      return static_cast<int>(2 * annot_dim);
    }
  };

  Data<Mat> RegressionLeaf::get_result_type() const {
    Data<Mat> ret_mat;
    if (double_mode) {
      ret_mat.set<Mat<double>>();
    } else {
      ret_mat.set<Mat<float>>();
    }
    return ret_mat;
  };

  void RegressionLeaf::get_result(
      const node_id_t &node_id,
      Data<MatRef> &target,
      const Data<MatCRef> &data,
      const std::function<void(void*)> &/*dptf*/) const {
    if (annot_dim == 0) {
      throw Forpy_Exception("This leaf has not been initialized yet!");
    }
    const auto resit = leaf_regression_map.find(node_id);
    if (resit == leaf_regression_map.end()) {
      throw Forpy_Exception("No result stored for this leaf! (Node ID: " +
                            std::to_string(node_id) + ")!");
    }
    const auto &regr_ftid_pair = resit->second;
    const auto &regr = regr_ftid_pair.first;
    const auto &ftid_vec = regr_ftid_pair.second;
    if (double_mode) {
      Mat<double> input(regression_input_dim, 1);
      if (regr->needs_input_data())  {
        data.match([&](const auto &data) {
            if (data.rows() != 1) {
              throw Forpy_Exception("Can only predict 1 sample here!");
            }
            if (static_cast<size_t>(data.cols()) != max_input_dim) {
              throw Forpy_Exception("Input data must have " +
                                    std::to_string(max_input_dim) +
                                    " columns!");
            }
            for (size_t ftid = 0; ftid < ftid_vec.size(); ++ftid) {
              input(ftid, 0) = static_cast<double>(data(0, ftid_vec[ftid]));
            }
          },
          [](const Empty &) {
            throw Forpy_Exception("Received empty input data, but it is "
                                  "required!");
          });
      }
      auto &tmat = target.get<MatRef<double>>();
      Mat<double> covar(annot_dim, annot_dim);
      regr->predict_covar(input,
                          tmat.block<1, Eigen::Dynamic>(0, 0, 1, annot_dim),
                          covar);
      tmat.block(0, annot_dim, 1, annot_dim) = covar.diagonal().transpose();
    } else {
      Mat<float> input(regression_input_dim, 1);
      if (regr->needs_input_data())  {
        data.match([&](const auto &data) {
            if (data.rows() != 1) {
              throw Forpy_Exception("Can only predict 1 sample here!");
            }
            if (static_cast<size_t>(data.cols()) != max_input_dim) {
              throw Forpy_Exception("Input data must have " +
                                    std::to_string(max_input_dim) +
                                    " columns!");
            }
            for (size_t ftid = 0; ftid < ftid_vec.size(); ++ftid) {
              input(ftid, 0) = static_cast<float>(data(0, ftid_vec[ftid]));
            }
          },
          [](const Empty &) {
            throw Forpy_Exception("Received empty input data.");
          });
      }
      auto &tmat = target.get<MatRef<float>>();
      Mat<float> covar(annot_dim, annot_dim);
      regr->predict_covar(input,
                          tmat.block<1, Eigen::Dynamic>(0, 0, 1, annot_dim),
                          covar);
      tmat.block(0, annot_dim, 1, annot_dim) = covar.diagonal().transpose();
    }
  };

  Data<Mat> RegressionLeaf::get_result(const node_id_t &node_id,
                                       const Data<MatCRef> &data,
                                       const std::function<void(void*)> &dptf) const {
    Data<Mat> ret;
    if (double_mode) {
      ret.set<Mat<double>>(Mat<double>::Zero(1, get_result_columns(1)));
      Data<MatRef> dref = MatRef<double>(ret.get_unchecked<Mat<double>>());
      get_result(node_id,
                 dref,
                 data,
                 dptf);
    } else {
      ret.set<Mat<float>>(Mat<float>::Zero(1, get_result_columns(1)));
      Data<MatRef> dref = MatRef<float>(ret.get_unchecked<Mat<float>>());
      get_result(node_id,
                 dref,
                 data,
                 dptf);
    }
    return ret;
  };

  void RegressionLeaf::get_result(const std::vector<Data<Mat>> &leaf_results,
                                  Data<MatRef> &target_v,
                                  const Vec<float> &weights) const {
    if (annot_dim == 0) {
      throw Forpy_Exception("This leaf has not been initialized yet!");
    }
    float weight_sum = 0.f;
    if (weights.rows() != 0 &&
        leaf_results.size() != static_cast<size_t>(weights.rows())) {
      throw Forpy_Exception("Invalid number of results/weights received!");
    }
    std::unique_ptr<Vec<float>> dws;
    const Vec<float> *rweightp;
    if (weights.rows() == 0) {
      dws.reset(new Vec<float>(Vec<float>::Ones(leaf_results.size())));
      rweightp = dws.get();
    } else {
      rweightp = &weights;
    }
    for (size_t res_idx = 0; res_idx < leaf_results.size(); ++res_idx) {
      leaf_results[res_idx].match([&, this](const auto &lr) {
          typedef typename get_core<decltype(lr.data())>::type regt;
          if (double_mode && typeid(regt) != typeid(double)) {
            throw Forpy_Exception("This leaf result type doesn't match the "
                                  "mode of this leaf (double)!");
          }
          if (!double_mode && typeid(regt) != typeid(float)) {
            throw Forpy_Exception("This leaf result type doesn't match the "
                                  "mode of this leaf (float)!");
          }
          auto &target = target_v.get<MatRef<regt>>();
          if (res_idx == 0) {
            target *= static_cast<regt>(0);
          }
          if (static_cast<size_t>(lr.cols()) !=
              this->get_result_columns(leaf_results.size()) &&
              !(summary_mode == 2 &&
                static_cast<size_t>(lr.cols()) == this->get_result_columns(1))) {
            throw Forpy_Exception("Invalid leaf result matrix received!");
          }
          if (lr.rows() != 1) {
            throw Forpy_Exception("Invalid leaf result matrix received!");
          }
          const float &weight = (*rweightp)(res_idx);
          weight_sum += weight;
          switch (summary_mode) {
          case 0:
            // Here, a new Gaussian distribution will be determined that spans
            // all the different models returned by the trees. This does, in
            // general, not make much sense, since the distribution can be strongly
            // multimodal. However, for this convenience interface, the value CAN
            // be helpful to determine the degree of uncertainty at this very
            // position. For the maths, see, e.g.,
            // http://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians.
            for (size_t i = 0; i < annot_dim; ++i) {
              target(0, i) += lr(0, i) * weight;
              target(0, i + annot_dim) += weight * (lr(0, i) * lr(0, i) +
                                                    lr(0, i + annot_dim));
            }
            break;
          case 1:
            // Mean prediction, mean variances.
            for (size_t i = 0; i < annot_dim; ++i) {
              target(0, i) += lr(i) * weight;
              target(0, i + annot_dim) += lr(0, i+annot_dim) * weight;
            }
            break;
          case 2:
            // All values.
            for (size_t i = 0; i < annot_dim; ++i) {
              target(0, res_idx*annot_dim*2+i) = lr(0, i);
              target(0, res_idx*annot_dim*2+i+annot_dim) = lr(0, i+annot_dim);
            }
            break;
          }
        },
        [](const Empty &) {
          throw Forpy_Exception("Empty leaf result received!");
        });
    }
    if (summary_mode != 2)
      target_v.match([&](auto &target) {
          target /= weight_sum;
        }, [](Empty &){});
    if (summary_mode == 0) {
      // The first part of the vector now contains \mu, the second \mu^{(2)}
      // (the second moment). The variance (assuming a normal distribution)
      // is then given by \mu^{(2)}-\mu^2.
      target_v.match([&](auto &target) {
        for (size_t i = 0; i < annot_dim; ++i) {
          target(0, i + annot_dim) -= target(0, i) * target(0, i);
        }
        },
        [](Empty &){});
    }
  };

  Data<Mat> RegressionLeaf::get_result(const std::vector<Data<Mat>> &leaf_results,
                                       const Vec<float> &weights) const {
    Data<Mat> ret;
    if (double_mode) {
      ret.set<Mat<double>>(Mat<double>::Zero(1,
                                             get_result_columns(leaf_results.size())));
      Data<MatRef> dref = MatRef<double>(ret.get_unchecked<Mat<double>>());
      get_result(leaf_results, dref, weights);
    } else {
      ret.set<Mat<float>>(Mat<float>::Zero(1,
                                           get_result_columns(leaf_results.size())));
      Data<MatRef> dref = MatRef<float>(ret.get_unchecked<Mat<float>>());
      get_result(leaf_results, dref, weights);
    }
    return ret;
  };

  bool RegressionLeaf::operator==(const ILeaf &rhs) const {
    const auto *rhs_c = dynamic_cast<RegressionLeaf const*>(&rhs);
    if (rhs_c == nullptr)
      return false;
    else {
      for (const auto &vpair : leaf_regression_map) {
        const auto key = vpair.first;
        if (rhs_c->leaf_regression_map.find(key) == rhs_c->leaf_regression_map.end()) {
          return false;
        } else {
          if (! (*(vpair.second.first) == *(rhs_c->leaf_regression_map.at(key).first))) {
            return false;
          }
          if (! (vpair.second.second == rhs_c->leaf_regression_map.at(key).second)) {
            return false;
          }
        }
      }

      return
        *reg_calc_template == *(rhs_c -> reg_calc_template) &&
        summary_mode == rhs_c->summary_mode &&
        regression_input_dim == rhs_c->regression_input_dim &&
        selections_to_try == rhs_c->selections_to_try &&
        num_threads == rhs_c -> num_threads &&
        random_seed == rhs_c->random_seed &&
        max_input_dim == rhs_c->max_input_dim &&
        annot_dim == rhs_c->annot_dim &&
        double_mode == rhs_c->double_mode;
    }
  };
} // namespace forpy
