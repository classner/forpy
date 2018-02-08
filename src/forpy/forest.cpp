#include "../../include/forpy/forest.h"
#include <forpy/leafs/regressionleaf.h>
#include <forpy/threshold_optimizers/regression_opt.h>
#include "../../include/forpy/util/sampling.h"

namespace forpy {

Forest::Forest(const uint &n_trees, const uint &max_depth,
               const uint &min_samples_at_leaf, const uint &min_samples_at_node,
               const std::shared_ptr<IDecider> &decider_template,
               const std::shared_ptr<ILeaf> &leaf_manager_template,
               const uint &random_seed)
    : trees(n_trees), random_seed(random_seed) {
  if (n_trees < 2)
    throw ForpyException(
        "The number of trees to form a forest must be greater 1!");
  if (decider_template == nullptr && leaf_manager_template == nullptr) {
    for (size_t i = 0; i < n_trees; ++i) {
      trees[i] = std::make_shared<Tree>(
          max_depth, min_samples_at_leaf, min_samples_at_node,
          std::make_shared<FastDecider>(std::make_shared<FastClassOpt>(), 0,
                                        random_seed + i + 1),
          std::make_shared<ClassificationLeaf>(), random_seed + i + 1);
    }
  } else {
    if (decider_template == nullptr || leaf_manager_template == nullptr)
      throw ForpyException(
          "If decider or leaf manager are specified, both must be specified!");
    for (size_t i = 0; i < n_trees; ++i) {
      trees[i] = std::make_shared<Tree>(
          max_depth, min_samples_at_leaf, min_samples_at_node,
          decider_template->create_duplicate(random_seed + i + 1),
          leaf_manager_template->create_duplicate(), random_seed + i + 1);
    }
  }
};

Forest::Forest(std::vector<std::shared_ptr<Tree>> &trees) {
  if (trees.size() < 2) {
    throw ForpyException(
        "The number of trees to form a forest must "
        "be greater 1!");
  }
  size_t tree_counter = 0;
  for (const auto &tree_ptr : trees) {
    if (!tree_ptr->is_initialized()) {
      throw ForpyException(
          "The method forest.CombineTrees "
          "is only meant to combine TRAINED trees! Otherwise, use "
          "'ForestFromTrees'! Tree " +
          std::to_string(tree_counter) + " is not initialized!");
      tree_counter++;
    }
  }
  this->trees.swap(trees);
};

Forest::Forest(std::string filename) {
  std::ifstream fstream(filename);
  std::stringstream sstream;
  if (fstream) {
    std::string json_ending = ".json";
    bool json_mode = false;
    if (ends_with(filename, ".json")) json_mode = true;
    if (json_mode) {
      cereal::JSONInputArchive iar(fstream);
      uint serialized_forpy_version;
      iar(CEREAL_NVP(serialized_forpy_version));
      iar(*this);
    } else {
      if (!ends_with(filename, ".fpf"))
        throw ForpyException("Forpy forests must be stored in `.fpf` files.");
      cereal::PortableBinaryInputArchive iar(fstream);
      uint serialized_forpy_version;
      iar(CEREAL_NVP(serialized_forpy_version));
      iar(*this);
    }
    fstream.close();
  } else {
    throw ForpyException("Could not load tree from file: " + filename);
  }
};  // namespace forpy

Forest *Forest::fit(const Data<MatCRef> &data_v,
                    const Data<MatCRef> &annotations_v, const size_t &n_threads,
                    const bool &bootstrap, const std::vector<float> &weights) {
  // On thread only sets up the trees, then waits until completion.
  ThreadControl::getInstance().set_num(n_threads);
  data_v.match(
      [&](const auto &data) {
        annotations_v.match(
            [&](const auto &annotations) {
              if (data.rows() == annotations.rows() &&
                  data.cols() != annotations.rows()) {
                LOG(WARNING)
                    << "The data and annotation counts don't match. "
                    << "Probably you did not transpose the data matrix "
                    << "(data cols: " << data.cols()
                    << ", annotation rows: " << annotations.rows()
                    << ", should be matching). "
                    << "I'll copy the data to fix this.";
                typedef typename get_core<decltype(data.data())>::type IT;
                typedef
                    typename get_core<decltype(annotations.data())>::type AT;
                DataStore<Mat> data_fixed_v, annot_fixed_v;
                data_fixed_v = std::make_shared<Mat<IT>>(data.transpose());
                annot_fixed_v = std::make_shared<Mat<AT>>(annotations);
                auto data_provider = std::make_shared<FastDProv>(
                    data_fixed_v, annot_fixed_v,
                    weights.size() == 0
                        ? nullptr
                        : std::make_shared<std::vector<float>>(weights));
                fit_dprov(data_provider, bootstrap);
              } else {
                auto data_provider = std::make_shared<FastDProv>(
                    data_v, annotations_v,
                    weights.size() == 0
                        ? nullptr
                        : std::make_shared<std::vector<float>>(weights));
                fit_dprov(data_provider, bootstrap);
              }
            },
            [&](const Empty &) { throw EmptyException(); });
      },
      [&](const Empty &) { throw EmptyException(); });
  return this;
};

Forest *Forest::fit_dprov(const std::shared_ptr<IDataProvider> &fdata_provider,
                          const bool &bootstrap) {
  auto &tc = ThreadControl::getInstance();
  if (tc.get_num() == 0) tc.set_num(1);
  // Run an initial set of tests on the elements of the first tree.
  auto *decider = const_cast<IDecider *>(get_decider().get());
  auto *lm = const_cast<ILeaf *>(get_leaf_manager().get());
  decider->get_threshopt()->check_annotations(fdata_provider.get());
  decider->set_data_dim(fdata_provider->get_feat_vec_dim());
  decider->is_compatible_with(*fdata_provider);
  if (!lm->is_compatible_with(*fdata_provider))
    throw ForpyException("Leaf manager (" + std::string(typeid(*lm).name()) +
                         ") incompatible with the selected data provider (" +
                         std::string(typeid(fdata_provider).name()) + ")!");
  if (!lm->is_compatible_with(*(decider->get_threshopt())))
    throw ForpyException(
        "Leaf manager (" + std::string(typeid(*lm).name()) +
        ") is incompatible with the selected threshold optimizer (" +
        std::string(typeid(decider->get_threshopt()).name()) + ")!");
  // Create the remaining elements.
  usage_map_t sample_ids_per_tree;
  std::shared_ptr<std::vector<id_t>> full_s_vec =
      std::make_shared<std::vector<id_t>>(
          fdata_provider->get_initial_sample_list());
  std::mt19937 random_engine(random_seed);
  auto weights = fdata_provider->get_weights();
  const float *weights_p = weights != nullptr ? &(weights->at(0)) : nullptr;
  VLOG(7) << "Seeding bootrap engine with seed " << random_seed;
  for (size_t i = 0; i < trees.size(); ++i) {
    if (bootstrap) {
      DLOG(INFO) << "Bootstrapping...";
      size_t n_samples = fdata_provider->get_n_samples();
      // For bootstrapping, we draw for each tree n samples with replacement
      // from the training set. Instead of really performing the drawing
      // experiment and counting the result, we can use a binomial distribution
      // to directly model the result:
      std::binomial_distribution<> distribution(
          n_samples, 1. / static_cast<double>(n_samples));
      auto subsamples_vec_ptr = std::make_shared<std::vector<id_t>>();
      subsamples_vec_ptr->reserve(n_samples);
      auto tree_weights = std::make_shared<std::vector<float>>(n_samples, 0.f);
      float *tree_weight_p = &(tree_weights->at(0));
      for (size_t i = 0; i < n_samples; ++i) {
        tree_weight_p[i] = weights_p == nullptr ? 1.f : weights_p[i];
        tree_weight_p[i] *= distribution(random_engine);
        if (tree_weight_p[i] > 0.f) subsamples_vec_ptr->push_back(i);
      }
      DLOG(INFO) << "subsamples size: " << subsamples_vec_ptr->size();
      DLOG(INFO) << "subsamples[0]: " << subsamples_vec_ptr->at(0);
      DLOG(INFO) << "subsamples[1]: " << subsamples_vec_ptr->at(1);
      sample_ids_per_tree.push_back({subsamples_vec_ptr, tree_weights});
    } else
      sample_ids_per_tree.push_back({full_s_vec, weights});
  }
  auto tree_provs = fdata_provider->create_tree_providers(sample_ids_per_tree);
  for (size_t i = 0; i < trees.size(); ++i) {
    if (i != 0) {
      decider->transfer_or_run_check(
          std::const_pointer_cast<IDecider>(trees[i]->get_decider()),
          tree_provs[i].get());
      lm->transfer_or_run_check(
          std::const_pointer_cast<ILeaf>(trees[i]->get_leaf_manager()).get(),
          trees[i]->get_decider()->get_threshopt().get(), tree_provs[i].get());
    }
    if (trees[i]->is_initialized_for_training)
      throw ForpyException("At least one of the trees has been fitted before!");
    trees[i]->is_initialized_for_training = true;
    auto sample_ids = std::make_shared<std::vector<id_t>>(
        tree_provs[i]->get_initial_sample_list());
    TodoMark mark(sample_ids, interv_t(0, sample_ids->size()),
                  trees[i]->next_id++, 0);
    tc.push_move(&Tree::parallel_DFS, trees[i].get(), std::move(mark),
                 tree_provs[i].get(), false);
  }
  tc.stop(true);
  for (size_t i = 0; i < trees.size(); ++i) {
    trees[i]->tree.resize(trees[i]->next_id);
    trees[i]->decider->finalize_capacity(trees[i]->next_id);
    trees[i]->leaf_manager->finalize_capacity(trees[i]->next_id);
  }
  return this;
};  // namespace forpy

void Forest::save(const std::string &filename) const {
  std::ofstream fstream(filename);
  {
    std::string json_ending = ".json";
    bool json_mode = false;
    if (ends_with(filename, ".json")) json_mode = true;
    if (json_mode) {
      cereal::JSONOutputArchive oa(fstream);
      oa(cereal::make_nvp("serialized_forpy_version", FORPY_LIB_VERSION()));
      oa(*this);
    } else {
      if (!ends_with(filename, ".fpf"))
        throw ForpyException("Forpy forests must be stored in `.fpf` files.");
      cereal::PortableBinaryOutputArchive oa(fstream);
      oa(cereal::make_nvp("serialized_forpy_version", FORPY_LIB_VERSION()));
      oa(*this);
    }
  }
  fstream.close();
}

ClassificationForest::ClassificationForest(
    const size_t &n_trees, const uint &max_depth,
    const uint &min_samples_at_leaf, const uint &min_samples_at_node,
    const uint &n_valid_features_to_use, const bool &autoscale_valid_features,
    const uint &random_seed, const size_t &n_thresholds,
    const float &gain_threshold)
    : Forest(n_trees, max_depth, min_samples_at_leaf, min_samples_at_node,
             std::make_shared<FastDecider>(
                 std::make_shared<FastClassOpt>(n_thresholds, gain_threshold),
                 n_valid_features_to_use, autoscale_valid_features),
             std::make_shared<ClassificationLeaf>(), random_seed),
      params{{"n_trees", n_trees},
             {"max_depth", max_depth},
             {"min_samples_at_leaf", min_samples_at_leaf},
             {"min_samples_at_node", min_samples_at_node},
             {"n_valid_features_to_use", n_valid_features_to_use},
             {"autoscale_valid_features", autoscale_valid_features},
             {"random_seed", random_seed},
             {"n_thresholds", n_thresholds},
             {"gain_threshold", gain_threshold}} {};

RegressionForest::RegressionForest(
    const size_t &n_trees, const uint &max_depth,
    const uint &min_samples_at_leaf, const uint &min_samples_at_node,
    const uint &n_valid_features_to_use, const bool &autoscale_valid_features,
    const uint &random_seed, const size_t &n_thresholds,
    const float &gain_threshold, const bool &store_variance,
    const bool &summarize)
    : Forest(n_trees, max_depth, min_samples_at_leaf, min_samples_at_node,
             std::make_shared<FastDecider>(
                 std::make_shared<RegressionOpt>(n_thresholds, gain_threshold),
                 n_valid_features_to_use, autoscale_valid_features),
             std::make_shared<RegressionLeaf>(store_variance, summarize),
             random_seed),
      params{{"n_trees", n_trees},
             {"max_depth", max_depth},
             {"min_samples_at_leaf", min_samples_at_leaf},
             {"min_samples_at_node", min_samples_at_node},
             {"n_valid_features_to_use", n_valid_features_to_use},
             {"autoscale_valid_features", autoscale_valid_features},
             {"random_seed", random_seed},
             {"n_thresholds", n_thresholds},
             {"gain_threshold", gain_threshold},
             {"store_variance", false},
             {"summarize", false}} {};

}  // namespace forpy
