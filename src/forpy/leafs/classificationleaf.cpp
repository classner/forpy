#include <forpy/leafs/classificationleaf.h>

namespace forpy {

  ClassificationLeaf::ClassificationLeaf(const uint &n_classes)
    : n_classes(n_classes)
    , stored_distributions(std::unordered_map<node_id_t, Vec<float>>()) {};

  bool ClassificationLeaf::is_compatible_with(const IDataProvider &/*data_provider*/) {
    return true;
  }

  bool ClassificationLeaf::is_compatible_with(const IThresholdOptimizer &threshopt) {
    const auto *to_c = dynamic_cast<ClassificationThresholdOptimizer const *>(
        &threshopt);
    if (to_c == nullptr) {
      throw Forpy_Exception("The ClassificationLeaf is only compatible with "
                            "the ClassificationThresholdOptimizer!");
    }
    if (n_classes == 0) {
      // Get them from there.
      n_classes = to_c->getN_classes();
      if (n_classes == 0) {
        throw Forpy_Exception("Internal error! "
                              "ClassificationThresholdOptimizer wrongly "
                              "initialized with 0 classes.");
      }
    } else {
      // Make sure they match.
      size_t tocl = to_c->getN_classes();
      if (tocl > n_classes) {
        throw Forpy_Exception("Internal error! The "
                              "ClassificationThresholdOptimizer indicates "
                              "a higher number of classes than the leaf.");
      } else if (tocl == 0) {
        throw Forpy_Exception("Internal error! "
                              "ClassificationThresholdOptimizer wrongly "
                              "initialized with 0 classes.");
      }
    }
    return true;
  };

  bool ClassificationLeaf::needs_data() const { return false; };

  void ClassificationLeaf::make_leaf(
      const node_id_t &node_id,
      const elem_id_vec_t &element_list,
      const IDataProvider &data_provider) {
    if (element_list.size() == 0) {
      throw Forpy_Exception("Can't create a leaf of 0 examples!");
    }
    if (n_classes == 0) {
      throw Forpy_Exception("This ClassificationLeaf has not been "
                            "constructed with a number of classes and "
                            "is_compatible_with has not been called yet!");
    }
    // Create the probability distribution at this leaf.
    Vec<float> distribution(Vec<float>::Zero(n_classes));
    float total = 0.f;
    const auto &sample_vec = data_provider.get_samples();
    sample_vec.match([&](const auto &sample_vec) {
        for (const auto &element_id : element_list) {
          float weight = sample_vec[element_id].weight;
          if (weight < 0.f) {
            throw Forpy_Exception("Negative sample weight detected!");
          }
          const size_t class_ = static_cast<const size_t>(
              sample_vec[element_id].annotation[0]);
          if (class_ >= n_classes) {
            throw Forpy_Exception("Invalid class detected: " +
                                  std::to_string(class_));
          }
          total += weight;
          distribution[class_] += weight;
        }
      },
      [](const Empty &) { throw Forpy_Exception("No samples received!"); }
      );
    if (total == 0.f) {
      throw Forpy_Exception("Received only samples with weight 0!");
    }
    // Normalize with the total count.
    distribution /= total;
    auto ret_val = stored_distributions.emplace(node_id, std::move(distribution));
    if (!ret_val.second) {
      throw Forpy_Exception("Tried to create the leaf value for a node "
                            "that has already been assigned one.");
    }
  };

  size_t ClassificationLeaf::get_result_columns(const size_t &n_trees) const {
    if (n_trees == 0) {
      throw Forpy_Exception("n_trees must be > 0!");
    }
    if (n_classes == 0) {
      throw Forpy_Exception("This classificaton leaf has not been "
                            "constructed with the number of classes "
                            "and the `is_compatible_with` method has not "
                            "been called yet.");
    }
    return n_classes;
  };

  Data<Mat> ClassificationLeaf::get_result_type() const {
    Data<Mat> ret_mat;
    ret_mat.set<Mat<float>>();
    return ret_mat;
  };

  void ClassificationLeaf::get_result(const node_id_t &node_id,
                                      Data<MatRef> &target_v,
                                      const Data<MatCRef> &/*data*/,
                                      const std::function<void(void*)> &/*dptf*/) const {
    MatRef<float> &target = target_v.get<MatRef<float>>();
    const auto stored_dist_it = stored_distributions.find(node_id);
    if (stored_dist_it == stored_distributions.end()) {
      throw Forpy_Exception("No leaf stored for node id " +
                            std::to_string(node_id));
    }
    target = (stored_dist_it->second).transpose();
  };

  /** Gets the mean of results. */
  void ClassificationLeaf::get_result(const std::vector<Data<Mat>> &leaf_results,
                                      Data<MatRef> &target_v,
                                      const Vec<float> &weights) const {
    MatRef<float> &target = target_v.get<MatRef<float>>();
    if (weights.rows() != 0) {
      if (static_cast<size_t>(weights.rows()) != leaf_results.size()) {
        throw Forpy_Exception("Invalid number of weights provided!");
      }
    }
    for (size_t tree_idx = 0; tree_idx < leaf_results.size(); ++tree_idx) {
      const auto &lr = leaf_results[tree_idx].get<Mat<float>>();
      if (lr.size() != n_classes) {
        throw Forpy_Exception("Inconsistent result size received!");
      }
      if (weights.rows() == 0) {
        target += lr;
      } else {
        target += lr * weights[tree_idx];
      }
    }
    if (weights.rows() == 0) {
      target /= static_cast<float>(leaf_results.size());
    } else {
      target /= weights.sum();
    }
  };

  bool ClassificationLeaf::operator==(const ILeaf &rhs) const {
    const auto *rhs_c = dynamic_cast<ClassificationLeaf const*>(&rhs);
    if (rhs_c == nullptr)
      return false;
    else {
      bool eq_nc = n_classes == rhs_c->n_classes;
      bool eq_std = true;
      for (const auto &std : stored_distributions) {
        const auto &rhs_std = rhs_c->stored_distributions.at(std.first);
        if (! std.second.isApprox(rhs_std)) {
          eq_std = false;
          break;
        }
      }
      return eq_nc && eq_std;
    }
  };

  const std::unordered_map<node_id_t, Vec<float>> &ClassificationLeaf::get_stored_dists() const {
    return stored_distributions;
  }

} // namespace forpy
