#include <forpy/tree.h>

namespace forpy {

  Tree::Tree(const uint &max_depth,
             const uint &min_samples_at_leaf,
             const uint &min_samples_at_node,
             const std::shared_ptr<IDecider> &decider,
             const std::shared_ptr<ILeaf> &leaf_manager)
    : max_depth(max_depth),
      is_initialized_for_training(false),
      min_samples_at_node(min_samples_at_node),
      min_samples_at_leaf(min_samples_at_leaf),
      weight(1.0f),
      stored_in_leafs(0),
      decider(decider),
      leaf_manager(leaf_manager),
      tree(0),
      marks(0) {
    if (max_depth == 0) {
      throw Forpy_Exception("The max depth must be >0!");
    }
    if (min_samples_at_leaf == 0) {
      throw Forpy_Exception("The minimum number of samples at leafs "
        "must be >0!");
    }
    if (min_samples_at_node < 2 * min_samples_at_leaf) {
      throw Forpy_Exception("The minimum number of samples at a node "
        "must be >= 2*min_samples_at_leaf!");
    }
    // No need to initialize the marks, they're empty.
    // Initialize the root node.
    tree.push_back(node_id_pair_t(0, 0));
    if (min_samples_at_leaf < 1) {
      throw Forpy_Exception("min_samples_at_leaf must be greater 0!");
    }
    if (min_samples_at_node < 2 * min_samples_at_leaf) {
      throw Forpy_Exception("min_samples_at_node must be greater or "
        "equal to 2 * min_samples_at_leaf!");
    }
  };

  Tree::Tree(std::string filename)  {
    std::ifstream fstream(filename);
    std::stringstream sstream;
    if (fstream) {
      {
        cereal::PortableBinaryInputArchive iar(fstream);
        uint serialized_forpy_version;
        iar(CEREAL_NVP(serialized_forpy_version));
        iar(*this);
      }
      fstream.close();
    } else {
      throw Forpy_Exception("Could not load tree from file: " +
                            filename);
    }
  };

  void Tree::make_node(IDataProvider *data_provider,
                       const bool &append_on_different) {
    // Assert that there is a node left to process.
    if (marks.empty()) {
      throw Forpy_Exception("Tried to process a node where none was "
        "left.");
    }
    // Get the data to work with.
    elem_id_vec_t element_id_list;
    unsigned int node_depth;
    node_id_t node_id;
    std::tie(element_id_list, node_id, node_depth) = marks.front();
    VLOG(10) << "Processing node with id " << node_id << " at depth "
             << node_depth << " with " << element_id_list.size() << " samples.";
    marks.pop_front();
    // Assert that there are sufficiently many elements there.
    FASSERT(element_id_list.size() >= this->min_samples_at_leaf);
    FASSERT(node_depth <= max_depth);
    auto node_predictor_func = [this](const Data<MatCRef> &dta,
                                      const node_id_t start_node,
                                      const std::function<void(void*)> &tf) {
      return this -> predict_leaf(dta, start_node, tf);
    };
    data_provider -> optimize_set_for_node(node_id,
                                           node_depth,
                                           node_predictor_func,
                                           element_id_list);
    FASSERT(element_id_list.size() >= this -> min_samples_at_leaf);
    // Check min_samples_at_node and max_tree_depth.
    if (element_id_list.size() < min_samples_at_node ||
        node_depth >= max_depth) {
      VLOG(11) << "Making leaf (too few samples or too much depth) with "
               << element_id_list.size() << " samples.";
      data_provider -> load_samples_for_leaf(node_id,
                                             nullptr,
                                             &element_id_list);
      leaf_manager -> make_leaf(node_id, element_id_list, *data_provider);
      stored_in_leafs += element_id_list.size();
      return;
    }
    // Construct a classifier using them.
    bool make_to_leaf;
    elem_id_vec_t list_left, list_right;
    VLOG(11) << "Making decision node... ";
    std::tie(make_to_leaf, list_left, list_right) =
      decider -> make_node(node_id,
                           node_depth,
                           min_samples_at_leaf,
                           element_id_list,
                           *data_provider);

    // If the classifier manager couldn't find a sufficiently good split.
    if (make_to_leaf) {
      VLOG(11) << "Making leaf (no sufficiently good split found) with "
               << element_id_list.size() << " samples...";
      data_provider -> load_samples_for_leaf(node_id,
                                             node_predictor_func,
                                             &element_id_list);
      leaf_manager -> make_leaf(node_id, element_id_list, *data_provider);
      stored_in_leafs += element_id_list.size();
    } else {
      VLOG(11) << "Creating child node todos...";
      FASSERT(list_left.size() >= this -> min_samples_at_leaf);
      FASSERT(list_right.size() >= this -> min_samples_at_leaf);
      // Left child.
      node_id_t left_id = tree.size();
      tree.push_back(node_id_pair_t(0, 0));
      tree[node_id].first = left_id;
      if (append_on_different)
        marks.push_back(
            node_todo_tuple_t(std::move(list_left), left_id, node_depth + 1));
      else
        marks.push_front(
            node_todo_tuple_t(std::move(list_left), left_id, node_depth + 1));
      // Right child.
      node_id_t right_id = tree.size();
      tree.push_back(node_id_pair_t(0, 0));
      tree[node_id].second = right_id;
      if (append_on_different)
        marks.push_back(
           node_todo_tuple_t(std::move(list_right), right_id, node_depth + 1));
      else
        marks.push_front(
           node_todo_tuple_t(std::move(list_right), right_id, node_depth + 1));
      data_provider -> track_child_nodes(node_id, left_id, right_id);
    }
  };

  size_t Tree::DFS(IDataProvider *data_provider,
                   const ECompletionLevel &completion) {
    auto start_size = marks.size();
    if (start_size <= 0) {
      throw Forpy_Exception("Called DFS on an empty marker set. Did "
                            "you initialize the training by calling the tree's fit method?");
    }
    switch (completion) {
    case ECompletionLevel::Complete:
      while (!marks.empty())
        make_node(data_provider, false);
      break;
    case ECompletionLevel::Level:
      while (marks.size() > start_size - 1)
        make_node(data_provider, false);
      break;
    case ECompletionLevel::Node:
      make_node(data_provider, false);
      break;
    default:
      throw Forpy_Exception("Unknown completion level used for DFS.");
    }
    return marks.size();
  };

  size_t Tree::BFS(IDataProvider *data_provider,
                   const ECompletionLevel &completion) {
    auto start_size = marks.size();
    if (start_size <= 0) {
      throw Forpy_Exception("Called BFS on an empty marker set. Did "
                            "you initialize the training by calling the tree's fit method?");
    }
    switch (completion) {
    case ECompletionLevel::Complete:
      while (!marks.empty())
        make_node(data_provider, true);
      break;
    case ECompletionLevel::Level:
      {
        unsigned int depth = std::get<2>(marks.front());
        while (!marks.empty() &&
               std::get<2>(marks.front()) == depth)
          make_node(data_provider, true);
      }
      break;
    case ECompletionLevel::Node:
      make_node(data_provider, true);
      break;
    default:
      throw Forpy_Exception("Unknown completion level used for BFS.");
    }
    return marks.size();
  };

  size_t Tree::get_depth() const {
    size_t depth = 0;
    if (! tree.empty()) {
      std::vector<std::pair<size_t, size_t>> to_check;
      to_check.push_back(std::make_pair(0, 0));
      while(! to_check.empty()) {
        const std::pair<size_t, size_t> to_process = to_check.back();
        to_check.pop_back();
        const size_t &node_id = to_process.first;
        const size_t &current_depth = to_process.second;
        if (current_depth > depth)
          depth = current_depth;
        FASSERT (node_id < tree.size());
        const auto &child_nodes = tree[node_id];
        if (child_nodes.first == 0) {
          if (child_nodes.second == 0) {
            // Leaf.
            continue;
          } else {
            to_check.push_back(std::make_pair(child_nodes.second,
                                              current_depth+1));
          }
        } else {
          to_check.push_back(std::make_pair(child_nodes.first,
                                            current_depth+1));
          if (child_nodes.second != 0) {
            to_check.push_back(std::make_pair(child_nodes.second,
                                              current_depth+1));
          }
        }
      }
    }
    return depth;
  }

  void Tree::fit(const Data<MatCRef> &data_v,
                 const Data<MatCRef> &annotations,
                 const bool &complete_dfs) {
    auto data_provider = std::make_shared<PlainDataProvider>(data_v,
                                                             annotations);
    fit_dprov(data_provider.get(), complete_dfs);
  }

  void Tree::fit_dprov(IDataProvider *data_provider,
                       const bool complete_dfs) {
    if (tree.size() > 1) {
      throw Forpy_Exception("This tree has been fitted already!");
    }
    // Compatibility-check threshopt - data provider.
    decider->get_threshopt()->check_annotations(*data_provider);
    decider->set_data_dim(data_provider->get_feat_vec_dim());
    // Compatibility-check for leaf manager - data provider.
    if (!leaf_manager -> is_compatible_with(*data_provider)) {
      throw Forpy_Exception("Leaf manager incompatible with selected"
                                 "data provider!");
    }
    if (!leaf_manager -> is_compatible_with(*(decider->get_threshopt()))) {
      throw Forpy_Exception("Leaf manager is incompatible with threshold "
                            "optimizer!");
    }
    // Get the initial sample set for the root node.
    auto init_samples = data_provider -> get_initial_sample_list();
    elem_id_vec_t samples(init_samples.begin(),
                          init_samples.end());
    if (samples.size() <= 1) {
      throw Forpy_Exception("More than one example must arrive at the "
        "tree root node.");
    }
    marks.push_back(node_todo_tuple_t(std::move(samples), 0, 0));
    is_initialized_for_training = true;
    // Directly complete the DFS fitting for this tree if necessary.
    if (complete_dfs) DFS(data_provider, ECompletionLevel::Complete);
  };

  node_id_t Tree::predict_leaf(const Data<MatCRef> &data,
                               const node_id_t &start_node,
                               const std::function<void(void*)> &dptf)
   const {
    node_id_t current_node_id = start_node;
    while (true) {
      if (tree[ current_node_id ].first == 0 &&
          tree[ current_node_id ].second == 0) {
        return current_node_id;
      } else {
        bool decision = decider ->
          decide(current_node_id, data, dptf);
        current_node_id = (decision ?  tree[ current_node_id ].first :
                                       tree[ current_node_id ].second);
      }
    }
  };

  Data<Mat> Tree::predict(const Data<MatCRef> &data_v,
                          const int &num_threads)
    const {
    data_v.match([&](const auto &data) {
        if (static_cast<size_t>(data.cols()) !=
            this->decider->get_data_dim()) {
          throw Forpy_Exception("Wrong array shape! Expecting " +
                                std::to_string(this->decider->get_data_dim()) +
                                "columns!");
        }
      },
      [](const Empty &) {throw Forpy_Exception("Received empty data!");});
    // Check the shape of the incoming array.
    if (num_threads == 0) {
       throw Forpy_Exception("The number of threads must be >0!");
    }
    {
      Data<Mat> result_v;
      data_v.match([&](const auto &data) {
          // Predict the first one out-of-place to determine the matrix type.
          Data<Mat> firstresult_v = this->predict_leaf_result(data.row(0));
          firstresult_v.match([&](const auto &firstresult) {
              typedef typename get_core<decltype(firstresult.data())>::type RT;
              typedef typename get_core<decltype(data.data())>::type IT;
              result_v.set<Mat<RT>>(data.rows(), firstresult.cols());
              auto &result = result_v.get_unchecked<Mat<RT>>();
              result.row(0) = firstresult;
              Data<MatCRef> in_v;
              Data<MatRef> out_v;
              for (size_t i = 1; i < static_cast<size_t>(data.rows()); ++i) {
                in_v.set<MatCRef<IT>>(data.row(i));
                out_v.set<MatRef<RT>>(result.row(i));
                this->leaf_manager->get_result(this->predict_leaf(in_v),
                                               out_v,
                                               in_v);
              }
            },
            [](const Empty &){});
        },
        [](const Empty &){});
      return result_v;
    }
  };

  Data<Mat> Tree::predict_leaf_result(const Data<MatCRef> &data,
                                      const node_id_t &start_node,
                                      const std::function<void(void*)> &dptf)
    const {
    return leaf_manager -> get_result(predict_leaf(data,
                                                   start_node,
                                                   dptf),
                                      data,
                                      dptf);
  };

  Data<Mat> Tree::combine_leaf_results(
                                 const std::vector<Data<Mat>> &leaf_results,
                                 const Vec<float> &weights) const {
    return leaf_manager -> get_result(leaf_results, weights);
  }

  bool Tree::is_initialized() const { return is_initialized_for_training; };

  /**
   * \brief The tree weight.
   */
  float Tree::get_weight() const { return weight; };

  /**
   * \brief The number of tree nodes.
   */
  size_t Tree::get_n_nodes() const { return tree.size(); };

  /**
   * \brief Sets the tree weight.
   */
  void Tree::set_weight(const float &new_weight) { weight = new_weight; };

  /**
   * \brief The data dimension that is required by this tree.
   */
  size_t Tree::get_input_data_dimensions() const {
    return decider->get_data_dim();
  };

  /**
   * \brief The classifier manager used by this tree.
   */
  std::shared_ptr<const IDecider> Tree::get_decider() const {
    return decider;
  };

  std::shared_ptr<const ILeaf> Tree::get_leaf_manager() const {
    return leaf_manager;
  }

  /**
   * \brief The number of samples stored in leafs.
   */
  size_t Tree::get_samples_stored() const {
    return stored_in_leafs;
  }

  /**
   * Get the vector of marked nodes.
   */
  std::deque<node_todo_tuple_t> Tree::get_marks() const {
    return marks;
  }

  bool Tree::operator==(Tree const &rhs) const {
    bool eq_depth = max_depth == rhs.max_depth;
    bool eq_init = is_initialized_for_training == rhs.is_initialized_for_training;
    bool eq_min_samples = min_samples_at_node == rhs.min_samples_at_node;
    bool eq_min_samples_leaf = min_samples_at_leaf == rhs.min_samples_at_leaf;
    bool eq_weight = weight == rhs.weight;
    bool eq_dec = *decider == *(rhs.decider);
    bool eq_lm = *leaf_manager == *(rhs.leaf_manager);
    bool eq_tree = tree == rhs.tree;
    bool eq_marks = marks == rhs.marks;
    //std::cout << eq_depth << std::endl;
    //std::cout << eq_init << std::endl;
    //std::cout << eq_min_samples << std::endl;
    //std::cout << eq_min_samples_leaf << std::endl;
    //std::cout << eq_weight << std::endl;
    //std::cout << eq_dec << std::endl;
    //std::cout << eq_lm << std::endl;
    //std::cout << eq_tree << std::endl;
    //std::cout << eq_marks << std::endl;
    return (eq_depth &&
            eq_init &&
            eq_min_samples &&
            eq_min_samples_leaf &&
            eq_weight &&
            eq_dec &&
            eq_lm &&
            eq_tree &&
            eq_marks);
  }

  void Tree::save(const std::string &filename) const {
    std::ofstream fstream(filename);
    {
      cereal::PortableBinaryOutputArchive oa(fstream);
      oa(cereal::make_nvp("serialized_forpy_version", FORPY_LIB_VERSION()));
      oa(*this);
    }
    fstream.close();
  }

  Tree::Tree() {};

} // namespace forpy
