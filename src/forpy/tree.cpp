#include <forpy/deciders/fastdecider.h>
#include <forpy/leafs/classificationleaf.h>
#include <forpy/leafs/regressionleaf.h>
#include <forpy/threshold_optimizers/fastclassopt.h>
#include <forpy/threshold_optimizers/regression_opt.h>
#include <forpy/tree.h>
#include <forpy/util/exponentials.h>
#include <forpy/util/serialization/stl/atomic.h>
#include <forpy/util/serialization/stl/random.h>
#include <forpy/util/serialization/variant.h>
#include <forpy/util/threading/ctpl.h>
#include <cereal/types/utility.hpp>

namespace forpy {

Tree::Tree(const uint &max_depth, const uint &min_samples_at_leaf,
           const uint &min_samples_at_node,
           const std::shared_ptr<IDecider> &decider,
           const std::shared_ptr<ILeaf> &leaf_manager, const uint &random_seed)
    : max_depth(max_depth),
      is_initialized_for_training(false),
      min_samples_at_node(min_samples_at_node),
      min_samples_at_leaf(min_samples_at_leaf),
      weight(1.0f),
      stored_in_leafs(0),
      decider(decider),
      leaf_manager(leaf_manager),
      tree(0),
      next_id(0),
      random_seed(random_seed) {
  if (max_depth == 0) throw ForpyException("The max depth must be >0!");
  if (min_samples_at_leaf == 0)
    throw ForpyException("The minimum number of samples at leafs must be >0!");
  if (min_samples_at_node < 2 * min_samples_at_leaf)
    throw ForpyException(
        "The minimum number of samples at a node must be >= "
        "2*min_samples_at_leaf!");
  // Default values.
  if (decider == nullptr) this->decider = std::make_shared<FastDecider>();
  if (leaf_manager == nullptr)
    this->leaf_manager = std::make_shared<ClassificationLeaf>();
  // No need to initialize the marks, they're empty.
  // Initialize the root node.
  tree.push_back(std::pair<id_t, id_t>(0, 0));
  if (min_samples_at_leaf < 1)
    throw ForpyException("min_samples_at_leaf must be greater 0!");
  if (min_samples_at_node < 2 * min_samples_at_leaf)
    throw ForpyException(
        "min_samples_at_node must be greater or equal to 2 * "
        "min_samples_at_leaf!");
  if (random_seed == 0) throw ForpyException("Random seed must be > 0!");
};

Tree::Tree(std::string filename) {
  std::ifstream fstream(filename);
  std::stringstream sstream;
  if (fstream) {
    std::string json_ending = ".json";
    bool json_mode = false;
    if (json_ending.size() < filename.size()) {
      if (std::equal(json_ending.rbegin(), json_ending.rend(),
                     filename.rbegin()))
        json_mode = true;
    }
    if (json_mode) {
      cereal::JSONInputArchive iar(fstream);
      uint serialized_forpy_version;
      iar(CEREAL_NVP(serialized_forpy_version));
      iar(*this);
    } else {
      if (!ends_with(filename, ".fpt"))
        throw ForpyException("Forpy trees must be stored in `.fpt` files.");
      cereal::PortableBinaryInputArchive iar(fstream);
      uint serialized_forpy_version;
      iar(CEREAL_NVP(serialized_forpy_version));
      iar(*this);
    }
    fstream.close();
  } else {
    throw ForpyException("Could not load tree from file: " + filename);
  }
};

void Tree::make_node(const IDataProvider *data_provider, Desk *desk) {
  auto &tc = ThreadControl::getInstance();
  auto &d = desk->t;
  if (d.marks.empty())
    throw ForpyException("Tried to process a node where none was left.");
  auto mark = std::move(d.marks.back());
  d.marks.pop_back();
  VLOG(10) << "Processing node with id " << mark.node_id << " at depth "
           << mark.depth << " with samples starting from " << mark.interv.first
           << " to " << mark.interv.second << " ("
           << mark.interv.second - mark.interv.first
           << "). Sample list size: " << mark.sample_ids->size() << ".";
  FASSERT(mark.interv.second > mark.interv.first);
  FASSERT(mark.interv.second <= mark.sample_ids->size());
  size_t n_samples = mark.interv.second - mark.interv.first;
  bool make_to_leaf = false;
  FASSERT(mark.node_id == 0 || n_samples >= this->min_samples_at_leaf);
  FASSERT(mark.depth < max_depth);
  if (n_samples < min_samples_at_node) {
    FASSERT(mark.depth == 0);
    VLOG(11) << "Making leaf (too few samples) with " << n_samples
             << " samples.";
    make_to_leaf = true;
  } else {
    VLOG(11) << "Optimizing decision node...";
    decider->make_node(mark, min_samples_at_leaf, *data_provider, desk);
    make_to_leaf = desk->d.make_to_leaf;
  }
  if (make_to_leaf) {
    VLOG(11) << "Making leaf...";
    leaf_manager->make_leaf(mark, *data_provider, desk);
    d.stored_in_leafs->operator+=(n_samples);
  } else {
    VLOG(11) << "Creating child node todos...";
    FASSERT(desk->d.left_int.second > desk->d.left_int.first);
    FASSERT(desk->d.right_int.second > desk->d.right_int.first);
    FASSERT(desk->d.left_int.second - desk->d.left_int.first >=
            this->min_samples_at_leaf);
    FASSERT((desk->d.right_int.second - desk->d.right_int.first >=
             this->min_samples_at_leaf));
    // Right child.
    d.tree_p->at(mark.node_id).second = desk->d.right_id;
    TodoMark mark_right(mark.sample_ids, desk->d.right_int, desk->d.right_id,
                        mark.depth + 1);
    if (desk->d.right_int.second - desk->d.right_int.first <
            min_samples_at_node ||
        mark_right.depth >= max_depth) {
      leaf_manager->make_leaf(mark_right, *data_provider, desk);
      d.stored_in_leafs->operator+=(desk->d.right_int.second -
                                    desk->d.right_int.first);
    } else {
      if (tc.get_idle() > 0 &&
          (desk->d.right_int.second - desk->d.right_int.first) *
                  sqrt(desk->d.annot_dim) * desk->d.input_dim >
              7000) {
        std::unique_lock<std::mutex> lck(fut_mtx);
        futures.emplace_back(tc.push_move(&Tree::DFS_and_store, this,
                                          std::move(mark_right), data_provider,
                                          ECompletionLevel::Complete));
      } else {
        d.marks.emplace_back(std::move(mark_right));
      }
    }
    // Left child.
    d.tree_p->at(mark.node_id).first = desk->d.left_id;
    TodoMark mark_left(mark.sample_ids, desk->d.left_int, desk->d.left_id,
                       mark.depth + 1);
    if (desk->d.left_int.second - desk->d.left_int.first <
            min_samples_at_node ||
        mark_left.depth >= max_depth) {
      leaf_manager->make_leaf(mark_left, *data_provider, desk);
      d.stored_in_leafs->operator+=(desk->d.left_int.second -
                                    desk->d.left_int.first);
    } else {
      if (false && tc.get_idle() > 0) {
        std::unique_lock<std::mutex> lck(fut_mtx);
        futures.emplace_back(tc.push_move(&Tree::DFS_and_store, this,
                                          std::move(mark_left), data_provider,
                                          ECompletionLevel::Complete));

      } else {
        d.marks.emplace_back(std::move(mark_left));
      }
    }
  }
};

void Tree::DFS(const IDataProvider *data_provider,
               const ECompletionLevel &completion, Desk *d) {
  auto start_size = d->t.marks.size();
  if (start_size == 0)
    throw ForpyException(
        "Called DFS on an empty marker set. Did you initialize the training by "
        "calling the tree's fit method?");
  switch (completion) {
    case ECompletionLevel::Complete:
      while (!d->t.marks.empty()) make_node(data_provider, d);
      break;
    case ECompletionLevel::Level:
      while (d->t.marks.size() > start_size - 1) make_node(data_provider, d);
      break;
    case ECompletionLevel::Node:
      make_node(data_provider, d);
      break;
    default:
      throw ForpyException("Unknown completion level used for DFS.");
  }
};

void Tree::DFS_and_store(Desk *d, TodoMark &mark, const IDataProvider *dprov,
                         const ECompletionLevel &comp) {
  VLOG(3) << "Starting DFSnstore task in thread " << d->thread_id
          << " with system id " << std::this_thread::get_id();
  VLOG(3) << "Processing node with node id " << mark.node_id << " at depth "
          << mark.depth << " for tree " << this;
  const auto &maps = decider->get_maps();
  d->setup(
      &stored_in_leafs, &next_id, &tree,
      const_cast<std::vector<size_t> *>(maps.first),
      const_cast<mu::variant<std::vector<float>, std::vector<double>,
                             std::vector<uint32_t>, std::vector<uint8_t>> *>(
          maps.second),
      const_cast<std::vector<Mat<float>> *>(leaf_manager->get_map()),
      random_seed);
  d->r.random_engine.seed(d->r.seed + mark.node_id);
  d->t.tree_p->push_back({0, 0});
  d->t.marks.push_back(std::move(mark));
  DFS(dprov, comp, d);
  d->reset();
}

void Tree::parallel_DFS(Desk * /*d*/, TodoMark &mark,
                        IDataProvider *data_provider, const bool &finalize) {
  auto &tc = ThreadControl::getInstance();
  VLOG(3) << "Initializing parallel DFS with " << tc.get_num()
          << " threads for the tree at " << this;
  const size_t &n_samples = data_provider->get_n_samples();
  // Total number of nodes in a full binary tree with 'n_samples' leaf nodes: 2
  // * n_samples - 1. Total number of nodes in a full binary tree with depth
  // 'max_depth': 2 ^ (max_depth + 1) - 1.
  float max_nodes_by_depth =  // Be safe against overflows here.
      powf(2.f, static_cast<float>(max_depth + 1.f)) - 1.f;
  size_t max_nodes_by_depth__size_t;
  if (max_nodes_by_depth >
      static_cast<float>(std::numeric_limits<size_t>::max()))
    max_nodes_by_depth__size_t = std::numeric_limits<size_t>::max();
  else
    max_nodes_by_depth__size_t = static_cast<size_t>(max_nodes_by_depth);
  size_t upper_bound = std::min<size_t>(
      2 * (n_samples / min_samples_at_leaf) - 1, max_nodes_by_depth__size_t);
  VLOG(3) << "Using upper bound for nodes " << std::to_string(upper_bound)
          << ", with " << std::to_string(n_samples) << " samples, "
          << std::to_string(min_samples_at_leaf) << " min samples at leaf and "
          << std::to_string(max_depth) << " maximum depth.";
  tree.resize(upper_bound);
  decider->ensure_capacity(upper_bound);
  leaf_manager->ensure_capacity(upper_bound);
  std::future<void> fut_store =
      tc.push_move(&Tree::DFS_and_store, this, std::move(mark), data_provider,
                   ECompletionLevel::Complete);
  if (finalize) {
    while (true) {
      fut_store.get();
      {
        std::unique_lock<std::mutex> lck(fut_mtx);
        if (futures.empty()) break;
        fut_store = std::move(futures.back());
        futures.pop_back();
      }
    }
    tree.resize(next_id);
    decider->finalize_capacity(next_id);
    leaf_manager->finalize_capacity(next_id);
  }
}

size_t Tree::get_depth() const {
  size_t depth = 0;
  if (!tree.empty()) {
    std::vector<std::pair<size_t, size_t>> to_check;
    to_check.push_back(std::make_pair(0, 0));
    while (!to_check.empty()) {
      const std::pair<size_t, size_t> to_process = to_check.back();
      to_check.pop_back();
      const size_t &node_id = to_process.first;
      const size_t &current_depth = to_process.second;
      if (current_depth > depth) depth = current_depth;
      FASSERT(node_id < tree.size());
      const auto &child_nodes = tree[node_id];
      if (child_nodes.first == 0) {
        if (child_nodes.second == 0) {
          // Leaf.
          continue;
        } else {
          to_check.push_back(
              std::make_pair(child_nodes.second, current_depth + 1));
        }
      } else {
        to_check.push_back(
            std::make_pair(child_nodes.first, current_depth + 1));
        if (child_nodes.second != 0) {
          to_check.push_back(
              std::make_pair(child_nodes.second, current_depth + 1));
        }
      }
    }
  }
  return depth;
}

Tree *Tree::fit(const Data<MatCRef> &data_v, const Data<MatCRef> &annotations_v,
                const size_t &n_threads, const bool &complete_dfs,
                const std::vector<float> &weights) {
  ThreadControl::getInstance().set_num(n_threads);
#ifdef WITHGPERFTOOLS
  ProfilerStart("forpy.profile.log");
#endif
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
                this->fit_dprov(data_provider, complete_dfs);
              } else {
                auto data_provider = std::make_shared<FastDProv>(
                    data_v, annotations_v,
                    weights.size() == 0
                        ? nullptr
                        : std::make_shared<std::vector<float>>(weights));
                this->fit_dprov(data_provider, complete_dfs);
              }
            },
            [&](const Empty &) { throw EmptyException(); });
      },
      [&](const Empty &) { throw EmptyException(); });
#ifdef WITHGPERFTOOLS
  ProfilerStop();
#endif
  return this;
}

Tree *Tree::fit_dprov(std::shared_ptr<IDataProvider> data_provider,
                      const bool &complete_dfs) {
  auto &tc = ThreadControl::getInstance();
  if (tc.get_num() == 0) tc.set_num(1);
  // Checks.
  if (tree.size() > 1 || is_initialized())
    throw ForpyException("This tree has been fitted already!");
  decider->get_threshopt()->check_annotations(data_provider.get());
  decider->set_data_dim(
      data_provider->get_feat_vec_dim());  // TODO add convenience method
                                           // copying the data to col major if
                                           // required and emitting a warning.
  decider->is_compatible_with(*data_provider);
  if (!leaf_manager->is_compatible_with(*data_provider))
    throw ForpyException("Leaf manager (" +  // TODO test exception
                         std::string(typeid(leaf_manager).name()) +
                         ") incompatible with the selected data provider (" +
                         std::string(typeid(data_provider).name()) + ")!");
  if (!leaf_manager->is_compatible_with(*(decider->get_threshopt())))
    throw ForpyException(
        "Leaf manager (" + std::string(typeid(leaf_manager).name()) +
        ") is incompatible with the selected threshold optimizer (" +
        std::string(typeid(decider->get_threshopt()).name()) + ")!");
  // Work.
  auto sample_ids = std::make_shared<std::vector<id_t>>(
      data_provider->get_initial_sample_list());
  FASSERT(sample_ids->size() > 0);
  TodoMark mark(sample_ids, interv_t(0, sample_ids->size()), next_id++, 0);
  is_initialized_for_training = true;
  if (complete_dfs) this->parallel_DFS(nullptr, mark, data_provider.get());
  return this;
};

id_t Tree::predict_leaf(const Data<MatCRef> &data, const id_t &start_node,
                        const std::function<void(void *)> &dptf) const {
  id_t current_node_id = start_node;
  while (true) {
    if (tree[current_node_id].first == 0 && tree[current_node_id].second == 0) {
      return current_node_id;
    } else {
      bool decision = decider->decide(current_node_id, data, dptf);
      current_node_id = (decision ? tree[current_node_id].first
                                  : tree[current_node_id].second);
    }
  }
};

Data<Mat> Tree::predict(const Data<MatCRef> &data_v, const int &num_threads,
                        const bool &use_fast_prediction_if_available,
                        const bool &predict_proba, const bool &for_forest) {
  if (num_threads == 0)
    throw ForpyException("The number of threads must be >0!");
  if (num_threads != 1) throw ForpyException("Unimplemented!");
  Data<Mat> result_v;
  data_v.match(
      [&](const auto &data) {
        Data<Mat> restype_v =
            leaf_manager->get_result_type(predict_proba, for_forest);
        size_t n_cols =
            leaf_manager->get_result_columns(1, predict_proba, for_forest);
        if (static_cast<size_t>(data.cols()) != this->decider->get_data_dim())
          throw ForpyException("Wrong array shape! Expecting " +
                               std::to_string(this->decider->get_data_dim()) +
                               " columns!");
        restype_v.match(
            [&](const auto &restype) {
              typedef typename get_core<decltype(restype.data())>::type RT;
              typedef typename get_core<decltype(data.data())>::type IT;
              result_v.set<Mat<RT>>(data.rows(), n_cols);
              auto &result = result_v.get_unchecked<Mat<RT>>();
              if (fast_tree.get() == nullptr &&
                  use_fast_prediction_if_available) {
                const auto *dec =
                    dynamic_cast<FastDecider const *>(this->decider.get());
                if (dec != nullptr) this->enable_fast_prediction();
              }
              Data<MatCRef> in_v;
              Data<MatRef> out_v;
              if (fast_tree.get() != nullptr) {
                VLOG(9) << "Using fast tree for predictions.";
                fast_tree->match([&](const auto &ftree) {
                  for (size_t i = 0; i < static_cast<size_t>(data.rows());
                       ++i) {
                    size_t node_id = 0;
                    while (std::get<2>(ftree[node_id]) != 0) {
                      if (data(i, std::get<0>(ftree[node_id])) <=
                          std::get<1>(ftree[node_id]))
                        node_id = std::get<2>(ftree[node_id]);
                      else
                        node_id = std::get<3>(ftree[node_id]);
                    }
                    in_v.set<MatCRef<IT>>(data.row(i));
                    out_v.set<MatRef<RT>>(result.row(i));
                    this->leaf_manager->get_result(node_id, out_v,
                                                   predict_proba, for_forest);
                  }
                });
              } else {
                for (size_t i = 0; i < static_cast<size_t>(data.rows()); ++i) {
                  in_v.set<MatCRef<IT>>(data.row(i));
                  out_v.set<MatRef<RT>>(result.row(i));
                  this->leaf_manager->get_result(this->predict_leaf(in_v),
                                                 out_v, predict_proba,
                                                 for_forest);
                }
              }
            },
            [](const Empty &) {});
      },
      [](const Empty &) {});
  return result_v;
};

Data<Mat> Tree::predict_proba(const Data<MatCRef> &data_v,
                              const int &num_threads,
                              const bool &use_fast_prediction_if_available) {
  return predict(data_v, num_threads, use_fast_prediction_if_available, true);
};

void Tree::enable_fast_prediction() {
  // Check that the tree is trained.
  if (!this->is_initialized_for_training && this->tree.size() > 0)
    throw ForpyException("Trying to unpack an untrained tree.");
  // Make sure the decider is a threshold decider.
  const auto *dec = dynamic_cast<FastDecider const *>(this->decider.get());
  if (dec == nullptr)
    throw ForpyException(
        "Unpacking can only be done with a threshold decider.");
  if (fast_tree.get() != nullptr)
    throw ForpyException("This tree has been unpacked before!");
  // Everything ok, start unpacking.
  auto maps = dec->get_maps();
  const auto &tree_map = *(maps.first);
  const auto &threshold_map_v = *(maps.second);
  VLOG(9) << "Unpacking " << tree.size() << " nodes for fast prediction.";
  threshold_map_v.match(
      [&](const auto &threshold_map) {
        this->fast_tree = std::make_unique<mu::variant<
            std::vector<std::tuple<size_t, float, size_t, size_t>>,
            std::vector<std::tuple<size_t, double, size_t, size_t>>,
            std::vector<std::tuple<size_t, uint32_t, size_t, size_t>>,
            std::vector<std::tuple<size_t, uint8_t, size_t, size_t>>>>();
        typedef
            typename std::remove_const<typename std::remove_reference<decltype(
                *threshold_map.data())>::type>::type thresh_t;
        FASSERT(tree_map.size() == threshold_map.size());
        // Find the maximum node id in the tree map.
        this->fast_tree
            ->set<std::vector<std::tuple<size_t, thresh_t, size_t, size_t>>>(
                tree.size(),
                std::make_tuple(0, static_cast<thresh_t>(0), 0, 0));
        auto &ftree = fast_tree->get_unchecked<
            std::vector<std::tuple<size_t, thresh_t, size_t, size_t>>>();
        for (size_t node_id = 0; node_id < tree.size(); ++node_id) {
          const auto &node_id_pair = tree[node_id];
          if (node_id_pair.first == 0 || node_id_pair.second == 0) {
            // Leaf. Leave zeros.
            continue;
          }
          ftree[node_id] =
              std::make_tuple(tree_map.at(node_id), threshold_map.at(node_id),
                              node_id_pair.first, node_id_pair.second);
        }
      },
      [](const Empty &) {
        throw ForpyException("Received empty threshold map!");
      });
  VLOG(9) << "Unpacking done.";
};

bool Tree::operator==(Tree const &rhs) const {
  bool eq_depth = max_depth == rhs.max_depth;
  bool eq_init = is_initialized_for_training == rhs.is_initialized_for_training;
  bool eq_min_samples = min_samples_at_node == rhs.min_samples_at_node;
  bool eq_min_samples_leaf = min_samples_at_leaf == rhs.min_samples_at_leaf;
  bool eq_weight = weight == rhs.weight;
  bool eq_dec = *decider == *(rhs.decider);
  bool eq_lm = *leaf_manager == *(rhs.leaf_manager);
  bool eq_tree = tree == rhs.tree;
  bool eq_nud = next_id == rhs.next_id;
  bool eq_rand = random_seed == rhs.random_seed;
  return (eq_depth && eq_init && eq_min_samples && eq_min_samples_leaf &&
          eq_weight && eq_dec && eq_lm && eq_tree && eq_nud && eq_rand);
}

void Tree::save(const std::string &filename) const {
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
      if (!ends_with(filename, ".fpt"))
        throw ForpyException("Forpy trees must be stored in `.fpt` files.");
      cereal::PortableBinaryOutputArchive oa(fstream);
      oa(cereal::make_nvp("serialized_forpy_version", FORPY_LIB_VERSION()));
      oa(*this);
    }
  }
  fstream.close();
}  // namespace forpy

ClassificationTree::ClassificationTree(
    const uint &max_depth, const uint &min_samples_at_leaf,
    const uint &min_samples_at_node, const uint &n_valid_features_to_use,
    const bool &autoscale_valid_features, const uint &random_seed,
    const size_t &n_thresholds, const float &gain_threshold)
    : Tree(max_depth, min_samples_at_leaf, min_samples_at_node,
           std::make_shared<FastDecider>(
               std::make_shared<FastClassOpt>(n_thresholds, gain_threshold),
               n_valid_features_to_use, autoscale_valid_features),
           std::make_shared<ClassificationLeaf>(), random_seed),
      params{{"max_depth", max_depth},
             {"min_samples_at_leaf", min_samples_at_leaf},
             {"min_samples_at_node", min_samples_at_node},
             {"n_valid_features_to_use", n_valid_features_to_use},
             {"autoscale_valid_features", autoscale_valid_features},
             {"random_seed", random_seed},
             {"n_thresholds", n_thresholds},
             {"gain_threshold", gain_threshold}} {};

RegressionTree::RegressionTree(
    const uint &max_depth, const uint &min_samples_at_leaf,
    const uint &min_samples_at_node, const uint &n_valid_features_to_use,
    const bool &autoscale_valid_features, const uint &random_seed,
    const size_t &n_thresholds, const float &gain_threshold,
    const bool &store_variance, const bool &summarize)
    : Tree(max_depth, min_samples_at_leaf, min_samples_at_node,
           std::make_shared<FastDecider>(
               std::make_shared<RegressionOpt>(n_thresholds, gain_threshold),
               n_valid_features_to_use, autoscale_valid_features),
           std::make_shared<RegressionLeaf>(store_variance, summarize),
           random_seed),
      params{{"max_depth", max_depth},
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
