#include <forpy/leafs/classificationleaf.h>
#include <forpy/threshold_optimizers/classification_opt.h>

namespace forpy {

bool ClassificationLeaf::is_compatible_with(const IThreshOpt &threshopt) {
  const auto *to_c = dynamic_cast<ClassificationOpt const *>(&threshopt);
  if (to_c == nullptr) {
    throw ForpyException(
        "The ClassificationLeaf is only compatible with "
        "the ClassificationOpt!");
  }
  if (n_classes == 0) {
    // Get them from there.
    n_classes = to_c->get_n_classes();
    class_transl_ptr = to_c->get_class_translation();
    true_max_class = to_c->get_true_max_class();
    if (n_classes == 0)
      throw ForpyException(
          "Internal error! "
          "ClassificationThresholdOptimizer wrongly "
          "initialized with 0 classes.");
  } else {
    // Make sure they match.
    size_t tocl = to_c->get_n_classes();
    if (to_c->get_true_max_class() > n_classes || tocl > n_classes) {
      throw ForpyException(
          "Internal error! The "
          "ClassificationThresholdOptimizer indicates "
          "a higher number of classes than the leaf.");
    } else if (tocl == 0) {
      throw ForpyException(
          "Internal error! "
          "ClassificationThresholdOptimizer wrongly "
          "initialized with 0 classes.");
    } else if ((class_transl_ptr == nullptr &&
                to_c->get_class_translation() != nullptr) ||
               (to_c->get_class_translation() == nullptr &&
                class_transl_ptr != nullptr)) {
      throw ForpyException(
          "Internal error! ClassificationOpt wrongly initialized.");
    } else {
      if (class_transl_ptr != nullptr) {
        for (size_t idx = 0; idx < class_transl_ptr->size(); ++idx) {
          if (class_transl_ptr->at(idx) !=
              to_c->get_class_translation()->at(idx))
            throw ForpyException(
                "Internal error! ClassificationOpt wrongly initialized.");
        }
      }
    }
  }
  return true;
};

void ClassificationLeaf::make_leaf(const TodoMark &todo_info,
                                   const IDataProvider & /*data_provider*/,
                                   Desk *desk) const {
  FASSERT(todo_info.interv.second > todo_info.interv.first);
  const id_t &node_id = todo_info.node_id;
  const id_t *elem_list_p = &todo_info.sample_ids->at(0);
  VLOG(31) << "Making leaf for node id " << node_id << " with samples "
           << todo_info.interv.first << " to " << todo_info.interv.second
           << ".";
  if (n_classes == 0)
    throw ForpyException(
        "This ClassificationLeaf has not been "
        "constructed with a number of classes and "
        "is_compatible_with has not been called yet!");
  float *dist;
  Vec<float> *distribution =
      const_cast<Vec<float> *>(&stored_distributions[node_id]);
  *distribution = Vec<float>::Zero(n_classes);
  dist = distribution->data();
  const uint *annot_p = desk->d.class_annot_p;
  const float *weights_p = desk->d.weights_p;
  float total = 0.f;
  if (weights_p == nullptr) {
    for (size_t i = todo_info.interv.first; i < todo_info.interv.second; ++i) {
      const size_t class_ = annot_p[elem_list_p[i]];
      total += 1.f;
      dist[class_] += 1.f;
    }
  } else {
    for (size_t i = todo_info.interv.first; i < todo_info.interv.second; ++i) {
      const size_t elem_id = elem_list_p[i];
      const size_t class_ = annot_p[elem_id];
      total += weights_p[elem_id];
      dist[class_] += weights_p[elem_id];
    }
  }
  if (total == 0.f)
    throw ForpyException("Received only samples with weight 0!");
  for (size_t i = 0; i < n_classes; ++i) dist[i] /= total;
};

size_t ClassificationLeaf::get_result_columns(const size_t &n_trees,
                                              const bool &predict_proba,
                                              const bool &for_forest) const {
  if (n_trees == 0) {
    throw ForpyException("n_trees must be > 0!");
  }
  if (n_classes == 0) {
    throw ForpyException(
        "This classificaton leaf has not been "
        "constructed with the number of classes "
        "and the `is_compatible_with` method has not "
        "been called yet.");
  }
  if (!predict_proba && !for_forest)
    return 1;
  else {
    if (class_transl_ptr == nullptr || for_forest)
      return n_classes;
    else
      return true_max_class + 1;
  }
};

Data<Mat> ClassificationLeaf::get_result_type(const bool &predict_proba,
                                              const bool &for_forest) const {
  Data<Mat> ret_mat;
  if (predict_proba || for_forest)
    ret_mat.set<Mat<float>>();
  else
    ret_mat.set<Mat<uint>>();
  return ret_mat;
};

void ClassificationLeaf::get_result(const id_t &node_id, Data<MatRef> &target_v,
                                    const bool &predict_proba,
                                    const bool &for_forest) const {
  if (node_id >= stored_distributions.size())
    throw ForpyException("No leaf stored for node id " +
                         std::to_string(node_id));
  if (predict_proba || for_forest) {
    MatRef<float> &target = target_v.get_unchecked<MatRef<float>>();
    if (class_transl_ptr == nullptr || for_forest)
      target = stored_distributions[node_id].transpose();
    else {
      target = Mat<float>::Zero(target.rows(), 1);
      const auto &res_comp = stored_distributions[node_id].transpose();
      for (size_t idx = 0; idx < static_cast<size_t>(res_comp.rows()); ++idx) {
        target(class_transl_ptr->at(idx), 0) = res_comp(idx);
      }
    }
  } else {
    MatRef<uint> &target = target_v.get_unchecked<MatRef<uint>>();
    stored_distributions[node_id].maxCoeff(&target(0, 0));
    if (class_transl_ptr != nullptr) {
      target(0, 0) = class_transl_ptr->at(target(0, 0));
    }
  }
};

void ClassificationLeaf::get_result(const std::vector<Data<Mat>> &leaf_results,
                                    Data<MatRef> &target_v,
                                    const Vec<float> &weights,
                                    const bool &predict_proba) const {
  target_v.match(
      [&](auto &target) {
        typedef typename get_core<decltype(target.data()[0])>::type RT;
        if (weights.rows() != 0) {
          if (static_cast<size_t>(weights.rows()) != leaf_results.size())
            throw ForpyException("Invalid number of weights provided!");
        }
        if (predict_proba && class_transl_ptr == nullptr) {
          for (size_t tree_idx = 0; tree_idx < leaf_results.size();
               ++tree_idx) {
            const auto &lr = leaf_results[tree_idx].get<Mat<RT>>();
            target += weights.rows() == 0
                          ? lr
                          : (lr.template cast<double>() *
                             static_cast<double>(weights[tree_idx]))
                                .template cast<RT>();
          }
          target /= static_cast<float>(leaf_results.size());
        } else {
          Mat<float> tmp(Mat<float>::Zero(
              leaf_results[0].get_unchecked<Mat<float>>().rows(),
              leaf_results[0].get_unchecked<Mat<float>>().cols()));
          for (size_t tree_idx = 0; tree_idx < leaf_results.size();
               ++tree_idx) {
            const auto &lr = leaf_results[tree_idx].get_unchecked<Mat<float>>();
            tmp += weights.rows() == 0 ? lr : (lr * weights[tree_idx]);
          }
          if (predict_proba) {
            for (size_t sidx = 0; sidx < static_cast<size_t>(tmp.rows());
                 ++sidx) {
              tmp.row(sidx) /= static_cast<float>(leaf_results.size());
              for (size_t didx = 0; didx < static_cast<size_t>(tmp.cols());
                   ++didx)
                target(sidx, class_transl_ptr->at(didx)) = tmp(sidx, didx);
            }
          } else {
            for (size_t sidx = 0; sidx < static_cast<size_t>(tmp.rows());
                 ++sidx) {
              int transl_cls;
              tmp.row(sidx).maxCoeff(&transl_cls);
              if (class_transl_ptr != nullptr)
                target(sidx, 0) = class_transl_ptr->at(transl_cls);
              else
                target(sidx, 0) = transl_cls;
            }
          }
        }
      },
      [](Empty &) { throw EmptyException(); });
};

bool ClassificationLeaf::operator==(const ILeaf &rhs) const {
  const auto *rhs_c = dynamic_cast<ClassificationLeaf const *>(&rhs);
  if (rhs_c == nullptr)
    return false;
  else {
    bool eq_nc = n_classes == rhs_c->n_classes;
    bool eq_std = true;
    for (size_t i = 0; i < stored_distributions.size(); ++i) {
      if (stored_distributions[i].data() != nullptr) {
        if (rhs_c->stored_distributions[i].data() != nullptr) {
          if (!stored_distributions[i].isApprox(
                  rhs_c->stored_distributions[i])) {
            eq_std = false;
            break;
          }
        } else {
          eq_std = false;
          break;
        }
      } else {
        if (rhs_c->stored_distributions[i].data() != nullptr) {
          eq_std = false;
          break;
        }
      }
    }
    return eq_nc && eq_std;
  }
};

const std::vector<Vec<float>> &ClassificationLeaf::get_stored_dists() const {
  return stored_distributions;
}

}  // namespace forpy
