#include <forpy/leafs/regressionleaf.h>

namespace forpy {

void RegressionLeaf::make_leaf(const TodoMark &todo_info,
                               const IDataProvider &data_provider,
                               Desk *desk) const {
  FASSERT(todo_info.interv.second > todo_info.interv.first);
  size_t n_samples = todo_info.interv.second - todo_info.interv.first;
  const id_t &node_id = todo_info.node_id;
  const auto &element_list = *todo_info.sample_ids;
  if (n_samples == 0)
    throw ForpyException("Received an empty element list at a leaf!");
  if (annot_dim == 0)
    throw ForpyException(
        "This regression leaf has not been initialized yet by calling "
        "`is_compatible_with` with the data provider!");
  if (data_provider.get_annot_vec_dim() != annot_dim)
    throw ForpyException(
        "The data provider data dimension does not agree with the one obtained "
        "from the compat. check!");
  VLOG(31) << "Making regression leaf with " << n_samples << " samples and "
           << data_provider.get_annot_vec_dim() << " target dimensions.";
  const auto &full_annotation_v = data_provider.get_annotations();
  if (store_variance)
    desk->l.leaf_regression_map_p->at(node_id) =
        Mat<float>::Zero(2 * data_provider.get_annot_vec_dim(), 1);
  else
    desk->l.leaf_regression_map_p->at(node_id) =
        Mat<float>::Zero(data_provider.get_annot_vec_dim(), 1);
  float *res_dta = desk->l.leaf_regression_map_p->at(node_id).data();
  full_annotation_v.match(
      [&](const auto &annotations) {
        typedef typename get_core<decltype(annotations.data()[0])>::type AT;
        const AT *ant_dta = annotations.data();
        size_t an_os = annotations.outerStride();
        FASSERT(annotations.innerStride() == 1);
        float n_added = 1.f;
        if (!store_variance) {
          for (size_t sidx = todo_info.interv.first;
               sidx < todo_info.interv.second; ++sidx, n_added += 1.f) {
            const float *sample_p = ant_dta + element_list[sidx] * an_os;
            for (size_t didx = 0; didx < annot_dim; ++didx)
              res_dta[didx] += (sample_p[didx] - res_dta[didx]) / n_added;
          }
        } else {
          for (size_t sidx = todo_info.interv.first;
               sidx < todo_info.interv.second; ++sidx, n_added += 1.f) {
            const float *sample_p = ant_dta + element_list[sidx] * an_os;
            for (size_t didx = 0; didx < annot_dim; ++didx) {
              float mean_new = res_dta[didx * 2] +
                               (sample_p[didx] - res_dta[didx * 2]) / n_added;
              if (n_added > 1.f) {
                res_dta[didx * 2 + 1] *= (n_added - 1.f) / n_added;
                res_dta[didx * 2 + 1] += (sample_p[didx] - res_dta[didx * 2]) *
                                         (sample_p[didx] - mean_new) / n_added;
              }
              res_dta[didx * 2] = mean_new;
            }
          }
        }
      },
      [](const MatCRef<double> &) {
        throw ForpyException(
            "Regression is only supported with `float` annotations.");
      },
      [](const MatCRef<uint> &) {
        throw ForpyException(
            "Regression is only supported with `float` annotations.");
      },
      [](const MatCRef<uint8_t> &) {
        throw ForpyException(
            "Regression is only supported with `float` annotations.");
      },
      [](const Empty &) { throw EmptyException(); });
};

void RegressionLeaf::get_result(const id_t &node_id, Data<MatRef> &target,
                                const bool &predict_proba,
                                const bool & /*for_forest*/) const {
  if (annot_dim == 0)
    throw ForpyException("This leaf has not been initialized yet!");
  const auto &res = leaf_regression_map[node_id];
  auto &tmat = target.get<MatRef<float>>();
  if (!predict_proba && store_variance) {
    for (size_t i = 0; i < annot_dim; ++i) {
      tmat(0, i) = res(2 * i);
    }
  } else
    tmat.row(0) = Eigen::Map<const Vec<float>>(res.data(), res.size());
};

void RegressionLeaf::get_result(const std::vector<Data<Mat>> &leaf_results,
                                Data<MatRef> &target_v,
                                const Vec<float> &weights,
                                const bool &predict_proba) const {
  target_v.match(
      [&, this](auto &target) {
        typedef typename get_core<decltype(target.data()[0])>::type RT;
        if (weights.rows() != 0) {
          if (static_cast<size_t>(weights.rows()) != leaf_results.size())
            throw ForpyException("Invalid number of weights provided!");
        }
        if (predict_proba && summarize) {
          for (size_t tree_idx = 0; tree_idx < leaf_results.size();
               ++tree_idx) {
            const auto &lr = leaf_results[tree_idx].get<Mat<RT>>();
            for (size_t dim_idx = 0; dim_idx < annot_dim; ++dim_idx) {
              for (size_t row_idx = 0;
                   row_idx < static_cast<size_t>(target.rows()); ++row_idx) {
                target(row_idx, dim_idx * 2) +=
                    (weights.rows() > 0 ? weights[tree_idx] : 1.f) *
                    lr(row_idx, dim_idx * 2);
                target(row_idx, dim_idx * 2 + 1) +=
                    (weights.rows() > 0 ? weights[tree_idx] : 1.f) *
                    (lr(row_idx, dim_idx * 2) * lr(row_idx, dim_idx * 2) +
                     lr(row_idx, dim_idx * 2 + 1));
              }
            }
          }
          target /= weights.rows() == 0 ? leaf_results.size() : weights.sum();
          for (size_t dim_idx = 0; dim_idx < annot_dim; ++dim_idx) {
            target.block(0, 2 * dim_idx + 1, target.rows(), 1) -=
                (target.block(0, 2 * dim_idx, target.rows(), 1).array() *
                 target.block(0, 2 * dim_idx, target.rows(), 1).array())
                    .matrix();
          }
        } else if (predict_proba) {
          for (size_t tree_idx = 0; tree_idx < leaf_results.size();
               ++tree_idx) {
            const auto &lr = leaf_results[tree_idx].get<Mat<RT>>();
            target.block(0, tree_idx * lr.cols(), lr.rows(), lr.cols()) = lr;
          }
        } else {
          for (size_t tree_idx = 0; tree_idx < leaf_results.size(); ++tree_idx)
            target += leaf_results[tree_idx].get<Mat<RT>>();
          target /= weights.rows() == 0 ? leaf_results.size() : weights.sum();
        }
      },
      [](Empty &) { throw EmptyException(); });
};  // namespace forpy

bool RegressionLeaf::operator==(const ILeaf &rhs) const {
  const auto *rhs_c = dynamic_cast<RegressionLeaf const *>(&rhs);
  if (rhs_c == nullptr)
    return false;
  else {
    if (rhs_c->leaf_regression_map.size() != leaf_regression_map.size())
      return false;
    for (size_t i = 0; i < leaf_regression_map.size(); ++i) {
      if (leaf_regression_map[i].data() != nullptr) {
        if (rhs_c->leaf_regression_map[i].data() != nullptr) {
          if (leaf_regression_map[i] != rhs_c->leaf_regression_map[i])
            return false;
        } else
          return false;
      } else if (rhs_c->leaf_regression_map[i].data() != nullptr)
        return false;
    }
  }
  return true;
};

}  // namespace forpy
