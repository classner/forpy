#include <forpy/threshold_optimizers/classificationthresholdoptimizer.h>
#include <forpy/types.h>

namespace forpy {

  ClassificationThresholdOptimizer::ClassificationThresholdOptimizer(
      const size_t &n_classes,                               
      const std::shared_ptr<IGainCalculator> &gain_calculator,
      const float &gain_threshold,
      const bool &use_fast_search_approximation)
  : use_fast_search_approximation(use_fast_search_approximation),
    n_classes(n_classes),
    gain_threshold(gain_threshold),
    gain_calculator(gain_calculator) {
    if (gain_threshold < 0.f) {
      throw Forpy_Exception("The gain threshold must be >= 0f!");
    }
  };

  bool ClassificationThresholdOptimizer::supports_weights() const {
    return true;
  };

  void ClassificationThresholdOptimizer::check_annotations(const IDataProvider &dprov) {
    if (dprov.get_initial_sample_list().size() == 0 ||
        dprov.get_annot_vec_dim() == 0) {
      throw Forpy_Exception("Need at least one sample with at least one dim!");
    }
    if (dprov.get_annot_vec_dim() > 1) {
      std::cerr << "Warning: more than one annotation dimension for the " <<
        "ClassificationThresholdOptimizer found! Only the first is used!" <<
        std::endl;
    }
    const auto &svec_v = dprov.get_samples();
    size_t maxval = 0;
    svec_v.match([&](const auto &svec) {
        for (const auto &sample : svec) {
          const auto &val = sample.annotation[0];
          if (std::is_floating_point<decltype(sample.annotation[0])>::value) {
            // Check integral value.
            if (ceilf(val) != val && floorf(val) != val) {
              throw Forpy_Exception("Invalid class label found in annotations: "
                                    + std::to_string(val) + "!");
            }
          }
          if (val < 0) {
            throw Forpy_Exception("Invalid negative label found in annotations: "
                                  + std::to_string(val) + "!");
          }
          if (val > maxval) {
            maxval = val;
          }
          if (n_classes != 0) {
            if (static_cast<size_t>(val) > n_classes - 1) {
              throw Forpy_Exception("Invalid class label found in annotations: "
                                    + std::to_string(val) + " for "
                                    + std::to_string(n_classes) + " classes!");
            }
          }
        }
      },
      [](const Empty &) {
        throw Forpy_Exception("Empty data provider detected.");
      });
    if (n_classes == 0) {
      n_classes = static_cast<size_t>(maxval + 1);
    }
  }

  FORPY_IMPL(FORPY_ITHRESHOPT_EARLYSTOP, AT, ClassificationThresholdOptimizer) {
    AT first_class = annotations(0, 0);
    for (Eigen::Index i = 1; i < annotations.rows(); ++i) {
      if (annotations(i, 0) != first_class) {
        return false;
      }
    }
    return true;
  };
  
  FORPY_IMPL(FORPY_ITHRESHOPT_OPT, ITFTAT, ClassificationThresholdOptimizer) {
    // Input checks and conversion.
    const size_t n_samples = annotations_matr.rows();
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
    if (feature_values_matr.cols() != 1) {
      throw Forpy_Exception("The number of feature columns must be 1 (is " +
                            std::to_string(feature_values_matr.cols()) +
                            ")!");
    }
    if (weights_matr.rows() != 0) {
      if (static_cast<size_t>(weights_matr.rows()) != n_samples) {
        throw Forpy_Exception("The number of weights does not match the "
                              "number of samples: " +
                              std::to_string(weights_matr.rows()) + " vs. " +
                              std::to_string(n_samples));
      }
    }
    // This is safe because we have inner stride one, one column and column
    // major layout of the features.
    const FT *feature_values = feature_values_matr.data();
    const FT *tmp_nextf = feature_values_matr.block(1, 0, 1, 1).data();
    if (tmp_nextf != feature_values + 1) {
      throw Forpy_Exception("Unstrided memory required for the features! "
                            "Stride is: " +
                            std::to_string(tmp_nextf - feature_values));
    }
    const float *weights;
    Vec<float> weight_mat;
    if (weights_matr.rows() == 0) {
      // Use standard weight 1. for all samples.
      weight_mat = Mat<float>::Ones(n_samples, 1);
      weights = weight_mat.data();
    } else {
      // Again, safe because of inner stride one, one column and column major
      // layout.
      weights = weights_matr.data();
    }

    if (n_classes == 0) {
      throw Forpy_Exception("n_classes was not specified in the constructor "
                            "and `check_annotations` was not called!");
    }
    // Work.
    bool valid = true;
    FT threshold = std::numeric_limits<FT>::lowest();
    auto best_result = optimized_split_tuple_t<FT>(
      std::make_pair(threshold,
                     static_cast<FT>(0)),
      EThresholdSelection::LessOnly,
      0, static_cast<unsigned int>(n_samples), 0.f, valid);
    // Handle this case quickly.
    if (2 * min_samples_at_leaf > n_samples ||
        n_samples <= 1) {
      // In this case, no valid threshold can be found.
      // Return the leftmost threshold directly.
      return best_result;
    }
    // Get a sorting permutation.
    std::vector<size_t> sort_perm = argsort(feature_values, n_samples);
    // No threshold fits "in between".
    if (feature_values[sort_perm[0]] == feature_values[*(sort_perm.end()-1)])
      return best_result;

    // Create the weighted occurrence histograms.
    std::vector<float> right_histogram(this -> n_classes, 0.f);
    for (size_t i = 0; i < n_samples; ++i) {
      if (static_cast<size_t>(annotations_matr(i, 0)) >= n_classes) {
        throw Forpy_Exception("Invalid class label detected: " +
                              std::to_string(annotations_matr(i, 0)) + ", " +
                              "n_classes: " + std::to_string(n_classes));
      }
      right_histogram[static_cast<size_t>(annotations_matr(i, 0))] += weights[i];
    }
    std::vector<float> left_histogram(this -> n_classes, 0);
    // Initialize.
    AT last_element_type;
    AT current_element_type;
    float current_weight;
    // Gain trackers.
    bool test_gain = true;
    float current_gain;
    float best_gain = std::numeric_limits<float>::lowest();
    // Feature value trackers.
    FT last_val = std::numeric_limits<FT>::lowest();
    FT current_val;
    valid = false;
    // Iterate over the feature list.
    // During the iteration it is assumed that the split is "left" of the
    // current index.
    for (size_t index = 0; index < n_samples; ++index) {
      current_val = feature_values[ sort_perm[index] ];
      current_element_type = annotations_matr(sort_perm[index], 0);
      if (std::is_floating_point<AT>::value) {
        if (floorf(current_element_type) != current_element_type &&
            ceilf(current_element_type) != current_element_type) {
          throw Forpy_Exception("Invalid class label found: " +
                                std::to_string(current_element_type));
        }
      }
      if (current_element_type < static_cast<AT>(0) ||
          current_element_type >= static_cast<AT>(n_classes)) {
        throw Forpy_Exception("Invalid class label found: " +
                              std::to_string(current_element_type) + ", " +
                              "n_classes: " + std::to_string(n_classes));
      }
      current_weight = weights[sort_perm[index]];
      // Check if a relevant change took place.
      if (use_fast_search_approximation) {
        if (index != 0 &&
              (current_val == last_val ||
               current_element_type == last_element_type)) {
          test_gain = false;
        } else {
          test_gain = true;
        }
      } else {
        test_gain = (current_val != last_val);
      }
      // Calculate the gain if necessary.
      // (Remember that the index corresponds to the number of elements left
      //  of the split! :)
      if (test_gain && index >= min_samples_at_leaf
                    && n_samples - index >= min_samples_at_leaf) {
        current_gain = gain_calculator -> approx(left_histogram,
                                                 right_histogram);
        if (current_gain >= best_gain + GAIN_EPS) {
          valid = true;
          best_gain = current_gain;
          // Take into account rounding issues if necessary.
          if (std::is_floating_point<FT>::value ||
              current_val != last_val + 1) {
            threshold = (last_val + current_val) /
                            static_cast<FT>(2);
            // If both values are just eps apart, there can't be a value
            // in between. This would lead to erroneous sample counts at
            // the following nodes. Correct it, if it happens.
            if (threshold == last_val) {
              // This is completely ok, since a 'less-than' comparison
              // is used.
              threshold = current_val;
            }
          } else {
            threshold = current_val;
          }
          best_result = optimized_split_tuple_t<FT>(
            std::make_pair(threshold, static_cast<FT>(0)),
            EThresholdSelection::LessOnly, static_cast<unsigned int>(index),
            static_cast<unsigned int>(n_samples - index),
            (*gain_calculator)(left_histogram, right_histogram), valid);
        }
      }
      // Update the histograms.
      left_histogram[ static_cast<size_t>(current_element_type) ] += current_weight;
      FASSERT(right_histogram[ static_cast<size_t>(current_element_type) ] >= current_weight);
      right_histogram[ static_cast<size_t>(current_element_type) ] -= current_weight;

      // Update the trackers.
      last_val = current_val;
      last_element_type = current_element_type;
    }
    // Check that the sum of elements going in either direction is the total.
    FASSERT(
     static_cast<size_t>(std::get<2>(best_result) + std::get<3>(best_result))
     == n_samples);
    return best_result;
  };

  float ClassificationThresholdOptimizer::get_gain_threshold_for(
      const size_t &node_id) {
    return gain_threshold;
  };

  bool ClassificationThresholdOptimizer::operator==(
      const IThresholdOptimizer &rhs) const {
    const auto *rhs_c = dynamic_cast<ClassificationThresholdOptimizer
                                     const *>(&rhs);
    if (rhs_c == nullptr) {
      return false;
    } else {
      bool eq_approx = use_fast_search_approximation == rhs_c ->
        use_fast_search_approximation;
      bool eq_cls = n_classes == rhs_c -> n_classes;
      bool eq_gainc = *gain_calculator == *(rhs_c -> gain_calculator);
      bool eq_gaint = gain_threshold == rhs_c -> gain_threshold;
      return eq_approx && eq_cls && eq_gainc && eq_gaint;
    }
  };

  bool ClassificationThresholdOptimizer::getUse_fast_search_approximation()
      const {
    return use_fast_search_approximation;
  }

  size_t ClassificationThresholdOptimizer::getN_classes() const {
    return n_classes;
  }

  float ClassificationThresholdOptimizer::getGain_threshold() const {
    return gain_threshold;
  }

  std::shared_ptr<IGainCalculator> ClassificationThresholdOptimizer::
      getGain_calculator() const {
    return gain_calculator;
  }
}
