#include <forpy/threshold_optimizers/regressionthresholdoptimizer.h>
#include <forpy/types.h>

namespace forpy {

  RegressionThresholdOptimizer::RegressionThresholdOptimizer(
      const size_t & n_thresholds,
      const std::shared_ptr<IRegressor> &regressor_template,
      const std::shared_ptr<IEntropyFunction> &entropy_function,
      const float &gain_threshold,
      const unsigned int  &random_seed)
  : n_thresholds(n_thresholds),
    entropy_function(entropy_function),
    reg_calc_template(regressor_template),
    gain_threshold(gain_threshold),
    random_engine(std::make_shared<std::mt19937>(random_seed)),
    seed_dist(0U, std::numeric_limits<unsigned int>::max()){
    if (gain_threshold < 0.f) {
      throw Forpy_Exception("The minimum gain threshold must be >=0.f!");
    }
    if (random_seed == 0) {
      throw Forpy_Exception("Need a random seed >= 1!");
    }
    regressor_template->freeze();
  };

  bool RegressionThresholdOptimizer::supports_weights() const { return false; };

  void RegressionThresholdOptimizer::prepare_for_optimizing(const size_t &node_id,
                              const int &num_threads) {
    if (num_threads == 0) {
      throw Forpy_Exception("The number of threads must be >1!");
    }
#ifndef _OPENMP
    if (num_threads > 1) {
      throw Forpy_Exception("This executable has been built without "
                            "OpenMP support. The number of threads must =1!");
    }
#endif
    if (num_threads < 0) {
      throw Forpy_Exception("Invalid number of threads: " +
                            std::to_string(num_threads) + "! Must be > 0!");
    }
    while (thread_engines.size() < static_cast<size_t>(num_threads)) {
      thread_engines.emplace_back(new std::mt19937());
    }
    main_seed = seed_dist(*random_engine);
  }

  FORPY_IMPL_DIRECT(FORPY_ITHRESHOPT_EARLYSTOP, , AT, RegressionThresholdOptimizer, {return false;};);

  FORPY_IMPL(FORPY_ITHRESHOPT_OPT, ITFTATR, RegressionThresholdOptimizer) {
    // Verification and input parsing.
    const size_t n_samples = annotations_matr.rows();
    if (reg_calc_template->needs_input_data() &&
        static_cast<size_t>(selected_data_matr.rows()) != n_samples) {
      throw Forpy_Exception("Invalid number of annotations/data: " +
                            std::to_string(selected_data_matr.rows()) +
                            " vs. " + std::to_string(n_samples) + "!");
    }
    if (annotations_matr.cols() == 0) {
      throw Forpy_Exception("Invalid number of annotation labels: " +
                            std::to_string(annotations_matr.cols()) +
                            " (must be >0)!");
    }
    if (static_cast<size_t>(feature_values_matr.rows()) != n_samples) {
      throw Forpy_Exception("Invalid number of feature values: " +
                            std::to_string(feature_values_matr.rows()) +
                            " vs. " + std::to_string(n_samples));
    }
    if (static_cast<size_t>(feature_values_matr.rows()) != n_samples) {
      throw Forpy_Exception("The number of feature values must be n_samples (is " +
                            std::to_string(feature_values_matr.rows()) +
                            ")!");
    }
    if (weights_matr.rows() != 0) {
      if (static_cast<size_t>(weights_matr.rows()) != n_samples) {
        throw Forpy_Exception("The number of weights does not match the "
                              "number of samples: " +
                              std::to_string(weights_matr.rows()) + " vs. " +
                              std::to_string(n_samples));
      }
      if (weights_matr.cols() != 1) {
        throw Forpy_Exception("Only one weight per sample may be given (" +
                              std::to_string(weights_matr.cols()) + " are)!");
      }
    }
    // Safe, because inner stride is 1 and we have col major format and the
    // number of cols is 1.
    const FT *feature_values = feature_values_matr.data();
    const size_t data_dimension = selected_data_matr.cols();
    const size_t annotation_dimension = annotations_matr.cols();

    if (n_samples == 0) {
      throw Forpy_Exception("Can't optimize for 0 samples.");
    }
    // Initialize search.
    bool valid = true;
    FT threshold = std::numeric_limits<FT>::lowest();
    auto best_result = optimized_split_tuple_t<FT>(
        std::make_pair(threshold, static_cast<FT>(0)),
        EThresholdSelection::LessOnly,
        0, static_cast<unsigned int>(n_samples), 0.f, valid);

    // Handle this case quickly.
    // Also care for sufficiently large data set to perform line fitting
    size_t min_samples = std::max(data_dimension + 2,
                                  2 * min_samples_at_leaf);
    if (min_samples > n_samples || n_samples <= 1) {
      // In this case, no valid threshold can be found.
      // Return the leftmost threshold directly.
      //std::cout << "too few samples" << std::endl;
      return best_result;
    }
    // Get a sorting permutation.
    std::vector<size_t> sort_perm = argsort(feature_values, n_samples);
    // No threshold fits "in between".
    if (feature_values[sort_perm[0]] == feature_values[*(sort_perm.end()-1)]) {
      //std::cout << "features constant!" << std::endl;
      return best_result;
    }
    std::set<FT> selected_feat_vals;
    if (n_thresholds >= n_samples || n_thresholds == 0) {
      // Test between all positions.
      for (size_t idx = 0; idx < sort_perm.size() - 1; ++idx) {
        if (feature_values[idx] == feature_values[idx+1])
          continue;
        selected_feat_vals.insert((feature_values[idx] +
                                   feature_values[idx + 1]) /
                                  static_cast<IT>(2));
      }
    } else {
      // Find the feature values to test.
      int thread_id = 0;
#if defined(_OPENMP)
      thread_id = omp_get_thread_num();
#endif
      FASSERT(thread_engines.size() > static_cast<size_t>(thread_id));
      unsigned int seed = main_seed + static_cast<unsigned int>(suggestion_index);
      if (seed == 0U) {
        seed += std::numeric_limits<unsigned int>::max() / 2;
      }
      thread_engines[thread_id] -> seed(seed);
      draw_feat_vals(feature_values[sort_perm[0]],
                     feature_values[*(sort_perm.end()-1)],
                     n_thresholds,
                     thread_engines[thread_id].get(),
                     &selected_feat_vals,
                     std::is_integral<FT>());
    }
    // Create the sample matrix. The sample-rows are sorted according to the
    // ascending feature order. It has +1 cols, due to "homogenous samples".
    // Only do this, if it is needed by the IRegressor. If not, avoid the
    // copying.
    size_t homogeneous_data_dimension = data_dimension + 1;
    Mat<IT> sample_mat;
    if (reg_calc_template->needs_input_data()) {
      sample_mat = Mat<IT>::Ones(n_samples, homogeneous_data_dimension);
      for (size_t col = 0; col < data_dimension; col++) {
        for (size_t row = 0; row < n_samples; row++) {
          sample_mat(row,col+1) = selected_data_matr(sort_perm[row], col);
        }
      }
    }
    // Create the annoation matrix
    const Mat<AT> &annot_mat = annotations_matr; //(n_samples, annotation_dimension);
    auto left_reg_calc = reg_calc_template->empty_duplicate();
    auto right_reg_calc = reg_calc_template->empty_duplicate();
    left_reg_calc->initialize_nocopy(sample_mat,
                                     annot_mat,
                                     std::make_pair(0,0));
    right_reg_calc->initialize_nocopy(sample_mat,
                                      annot_mat,
                                      std::make_pair(0,
                                                     static_cast<int>(n_samples)));
    // Initialize.
    // Calculate the entropy of the total sample set
    float total_entropy;
    bool numerically_instable = false;
    bool unique_solution_found = right_reg_calc->has_solution();
    Vec<FT> prediction = Vec<FT>::Zero(annotation_dimension);
    Mat<FT> pred_covar_mat = Mat<FT>::Zero(annotation_dimension,
                                           annotation_dimension);

    if (unique_solution_found) {
      // Now calculate the combined entropy
      total_entropy = 0.f;
      // faster entropy calculation, if the prediction (co)-variance does not
      // depend on the actual input
      if (right_reg_calc->has_constant_prediction_covariance()) {
        // Calculate the determinant of the prediction covariance matrix (is
        // diagonal!).
        right_reg_calc->get_constant_prediction_covariance(pred_covar_mat);
        IT determinant = pred_covar_mat.diagonal().unaryExpr([](IT elem) {return std::max<IT>(elem, ENTROPY_EPS);}).array().log().sum();
          total_entropy = entropy_function->differential_normal(
              static_cast<float>(determinant),
              static_cast<const uint>(annotation_dimension)) *
            static_cast<float>(n_samples);
      } else {
        throw Forpy_Exception("update impl");
        for (size_t i = 0; i < n_samples; i++) {
          // Calculate the determinant of the prediction covariance matrix (is
          // diagonal!).
          right_reg_calc->predict_covar_nocopy(sample_mat.row(i).transpose(),
                                               prediction,
                                               pred_covar_mat);
          IT determinant = pred_covar_mat.diagonal().prod();
          if (determinant < 0.f) {
            numerically_instable = true;
            break;
          }
          total_entropy += entropy_function->differential_normal(
              static_cast<float>(determinant),
              static_cast<const uint>(annotation_dimension));
        }
      }
    }
    if ((!unique_solution_found) || numerically_instable) {
      // If no line fitting can be performed on the total sample set, it won't
      // work on the subsets either.
      //std::cout << "regthropt: no unique sol found or numerically unst." << std::endl;
      valid = false;
      return best_result;
    }
    // Storage for the left regression prediction and covariance
    auto left_prediction = Vec<FT>(annotation_dimension);
    auto left_pred_covar_mat = Mat<FT>(annotation_dimension,
                                                  annotation_dimension);
    // Storage for the right regression prediction and covariance
    auto right_prediction = Vec<FT>(annotation_dimension);
    auto right_pred_covar_mat = Mat<FT>(annotation_dimension,
                                        annotation_dimension);
    // Gain trackers.
    float current_gain;
    float current_left_entropy;
    float current_right_entropy;
    size_t best_split_index = 0;
    float best_gain = std::numeric_limits<float>::lowest();
    // Feature value trackers.
    FT last_val = std::numeric_limits<FT>::lowest();
    FT current_val;
    auto feat_val_it = selected_feat_vals.begin();
    bool usable_split;
    valid = false;
    // Iterate over the possible thresholds.
    // During the iteration it is assumed that the split is "left" of the
    // current index.
    for (size_t index = 0;
         (index < n_samples + 1) &&
           (feat_val_it != selected_feat_vals.end()); ++index) {
      usable_split = true;
      if (index < n_samples) {
        current_val = feature_values[ sort_perm[index] ];
      } else {
        current_val = std::numeric_limits<FT>::max();
      }
      // Check if gain calculation is necessary
      if ((current_val >= *feat_val_it) &&
          (last_val < *feat_val_it) &&
          (index >= min_samples_at_leaf) &&
          ((n_samples - index) >= min_samples_at_leaf)) {
        // Update the regression calculators
        left_reg_calc->set_index_interval(
           std::make_pair(0, static_cast<int>(std::min(index, n_samples))));
        right_reg_calc->set_index_interval(
           std::make_pair(static_cast<int>(std::min(index, n_samples)),
                          static_cast<int>(n_samples)));
        // Get the left entropy
        if (! left_reg_calc->has_solution()) {
          usable_split = false;
          current_left_entropy = std::numeric_limits<float>::infinity();
        } else {
          // Now calculate the left entropy
          current_left_entropy = 0.f;
          // faster entropy calculation, if the prediction (co)-variance does
          // not depend on the actual input
          if (left_reg_calc->has_constant_prediction_covariance()) {
            // Calculate the determinant of the prediction covariance matrix
            // (is diagonal!).
            left_reg_calc->get_constant_prediction_covariance(
                left_pred_covar_mat);
            //std::cout << "left pred covar mat: " << left_pred_covar_mat << std::endl;
            IT determinant = left_pred_covar_mat.diagonal().unaryExpr([](IT elem) {return std::max<IT>(elem, ENTROPY_EPS);}).array().log().sum();
            //std::cout << "left determinant: " << determinant << std::endl;
            //std::cout << "index: " << index << std::endl;
            /*if (determinant < 0.f) {
              usable_split = false;
              current_left_entropy = std::numeric_limits<float>::infinity();
              numerically_instable = true;
              } else {*/
              current_left_entropy = entropy_function->differential_normal(
                  static_cast<float>(determinant),
                  static_cast<const uint>(annotation_dimension));
              //std::cout << "left entropy: " << current_left_entropy << std::endl;
              current_left_entropy *= static_cast<float>(index);
              //std::cout << "left entropy: " << current_left_entropy << std::endl;
              //}
          } else {
            throw Forpy_Exception("update impl");
            for (size_t i = 0; i < index; i++) {
              // Calculate the determinant of the prediction covariance
              // matrix (is diagonal!).
              left_reg_calc->predict_covar_nocopy(sample_mat.row(i),
                                                  left_prediction,
                                                  left_pred_covar_mat);
              IT determinant = left_pred_covar_mat.diagonal().prod();
              if (determinant < 0.f) {
                usable_split = false;
                current_left_entropy = std::numeric_limits<float>::infinity();
                break;
              }
              current_left_entropy += entropy_function->differential_normal(
                 static_cast<float>(determinant),
                 static_cast<const uint>(annotation_dimension)) *
                static_cast<float>(index);
            }
          }
        }
        // Get the right entropy
        if ((! usable_split) || (! right_reg_calc->has_solution())) {
          current_right_entropy = std::numeric_limits<float>::infinity();
          usable_split = false;
        } else {
          // Now calculate the right entropy
          current_right_entropy = 0.f;
          // faster entropy calculation, if the prediction (co)-variance does
          // not depend on the actual input
          if (right_reg_calc->has_constant_prediction_covariance()) {
            // Calculate the determinant of the prediction covariance matrix
            // (is diagonal!).
            right_reg_calc->get_constant_prediction_covariance(
                right_pred_covar_mat);
            //IT determinant = right_pred_covar_mat.diagonal().prod();
            IT determinant = right_pred_covar_mat.diagonal().unaryExpr([](IT elem) {return std::max<IT>(elem, ENTROPY_EPS);}).array().log().sum();
            /*
            if (determinant < 0.f) {
              usable_split = false;
              current_right_entropy = std::numeric_limits<float>::infinity();
              numerically_instable = true;
              } else {*/
              current_right_entropy = entropy_function->differential_normal(
                  static_cast<float>(determinant),
                  static_cast<const uint>(annotation_dimension))
                * static_cast<float>(n_samples - index);
              //}
          } else {
            throw Forpy_Exception("update impl");
            for (size_t i = index; i < n_samples; i++) {
              // Calculate the determinant of the prediction covariance
              // matrix (is diagonal!).
              right_reg_calc->predict_covar_nocopy(sample_mat.row(i),
                                                   right_prediction,
                                                   right_pred_covar_mat);
              IT determinant = right_pred_covar_mat.diagonal().prod();
              if (determinant < 0.f) {
                usable_split = false;
                current_right_entropy = std::numeric_limits<float>::infinity();
                break;
              }
              current_right_entropy += entropy_function->differential_normal(
                  static_cast<float>(determinant),
                  static_cast<const uint>(annotation_dimension))
                * static_cast<float>(n_samples - index);
            }
          }
          //std::cout << "right entropy: " << current_right_entropy << std::endl;
        }
        if (usable_split) {
          current_gain = total_entropy -
            (current_left_entropy + current_right_entropy);
        } else {
          current_gain = std::numeric_limits<float>::lowest();
        }
        //std::cout << "current gain: " << current_gain << std::endl;
        if (current_gain >= best_gain + GAIN_EPS) {
          //std::cout << "new best!" << current_gain << ", "<<current_left_entropy << ", " <<current_right_entropy << std::endl;
          valid = true;
          best_gain = current_gain;
          best_split_index = index;
          threshold = *feat_val_it;
        }
      }
      // Update the feature value pointer.
      while (feat_val_it != selected_feat_vals.end() &&
             *feat_val_it <= current_val)
         feat_val_it++;
      // Update the trackers.
      if (last_val != current_val)
          last_val = current_val;
    } // end for
    if (valid) {
      //std::cout << "creating best retval" << std::endl;
      // Create the best-result return value
      best_result = optimized_split_tuple_t<FT>(
          std::make_pair(threshold, static_cast<FT>(0)),
          EThresholdSelection::LessOnly,
          static_cast<unsigned int>(best_split_index),
          static_cast<unsigned int>(n_samples - best_split_index),
          best_gain,
          valid);
      //std::cout << best_split_index << std::endl;
    }
    // Check that the sum of elements going in either direction is the total.
    FASSERT(static_cast<size_t>(std::get<2>(best_result) +
                                std::get<3>(best_result)) == n_samples);
    //std::cout << "returning regularly" << std::endl;
    return best_result;
  };

  float RegressionThresholdOptimizer::get_gain_threshold_for(
      const size_t &node_id) {
    return gain_threshold;
  };

  std::shared_ptr<IRegressor> RegressionThresholdOptimizer::getRegressorTemplate() const {
    return reg_calc_template;
  };

  void RegressionThresholdOptimizer::check_annotations(const IDataProvider &dprov) {};

  bool RegressionThresholdOptimizer::operator==(const IThresholdOptimizer &rhs) const {
    const auto *rhs_c = dynamic_cast<RegressionThresholdOptimizer const *>(&rhs);
    if (rhs_c == nullptr) {
      return false;
    } else {
      bool eq_thresh = n_thresholds == rhs_c -> n_thresholds;
      bool eq_ef = *entropy_function == *(rhs_c -> entropy_function);
      bool eq_rc = *reg_calc_template == *(rhs_c -> reg_calc_template);
      bool eq_gaint = gain_threshold == rhs_c -> gain_threshold;
      bool eq_re = *random_engine == *(rhs_c -> random_engine);
      bool eq_ms = main_seed == rhs_c -> main_seed;
      return eq_thresh && eq_ef && eq_rc && eq_gaint && eq_re && eq_ms;
    }
  };
} // namespace forpy
