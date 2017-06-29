#include <forpy/features/featureselector.h>

namespace forpy {

  FeatureSelector::FeatureSelector() {};

  FeatureSelector::FeatureSelector(
    const size_t &n_selections_per_node,
    const size_t &selection_dimension,
    const size_t &how_many_available,
    size_t max_to_use,
    const uint &random_seed )
    : dimension(selection_dimension),
      how_many_per_node(n_selections_per_node),
      how_many_available(how_many_available),
      max_to_use(max_to_use),
      random_engine(std::make_shared<std::mt19937>(random_seed)) {
    //TODO: Remove the option to have `max_to_use` for a speed gain.
    // Sanity checks.
    if (how_many_per_node == 0) {
      throw Forpy_Exception("The number of selections per node must be >0!");
    }
    if (how_many_available == 0) {
      throw Forpy_Exception("The number of available dimensions must be >0!");
    }
    if (dimension == 0)
      throw Forpy_Exception("The number of selected dimensions per "
       "proposal by the features selector must be greater 0!");
    if (max_to_use == 0) {
      max_to_use = how_many_available;
      this->max_to_use = how_many_available;
    }
    if (dimension > how_many_available)
      throw Forpy_Exception("The number of available features must "
       "be greater than the number of features to select for one proposal!");
    int64_t combinations = ibinom(
      static_cast<int>(std::min(how_many_available, max_to_use)),
      static_cast<int>(dimension));
    if (((combinations < static_cast<int64_t>(how_many_per_node) * 2 &&
         combinations != -1) &&  // -1: Overflow detected: a LOT of comb.s ;)
         dimension != 1 && n_selections_per_node != 1) ||
        (dimension == 1 && how_many_per_node > std::min(max_to_use, how_many_available))) {
      throw Forpy_Exception("The standard feature selection "
        "provider has been initialized wrongly. In the "
        "case of n data features and k to choose per node being less "
        "than how many samples per node to provide / 2, "
        "the provided algorithm might be very slow! Use a different FeatureSelectionProvider.");
    }
    if (random_seed == 0) {
      throw Forpy_Exception("Choose a random seed >0!");
    }
    used_indices = std::make_shared<std::vector<size_t>>();
    used_index_markers = std::vector<bool>(how_many_available, false);
    available_indices = std::make_shared<std::vector<size_t>>(how_many_available);
    std::iota(available_indices -> begin(), available_indices -> end(), 0);
    std::shuffle(available_indices -> begin(),
                 available_indices -> end(), *random_engine);
  };

  size_t FeatureSelector::get_input_dimension() const {
    return how_many_available;
  };

  size_t FeatureSelector::get_selection_dimension() const {
    return dimension;
  };

  std::shared_ptr<IFeatureProposer> FeatureSelector::get_proposal_generator() {
    // The maximum amount of new indices that may be used in the generated
    // selection proposals.
    size_t new_to_include = max_to_use - used_indices -> size();
    size_t index_max = used_indices -> size() +
      std::min<size_t>(new_to_include, available_indices -> size()) - 1;

    return std::shared_ptr<FeatureProposer>(
            new FeatureProposer(dimension, index_max,
           how_many_per_node, used_indices, available_indices, random_engine));
  };

  proposal_set_t FeatureSelector::get_proposals() {
    auto gen = get_proposal_generator();
    // Generate the selections.
    auto ret_set = proposal_set_t();
    ret_set.reserve(how_many_per_node);

    for (uint i = 0; i < how_many_per_node; ++i) {
      ret_set.emplace(gen -> get_next());
    }

    return ret_set;
  };

  void FeatureSelector::register_used(const proposal_set_t &proposals) {
    // Speed optimization: in this case, the features don't have to be
    // tracked.
    if (max_to_use == how_many_available)
      return;

    // The erase operation is not very fast on vectors in general. However,
    // by the design of the two functions get_proposals and register_used,
    // elements are always pretty much at the end of the vector if they
    // are erased, making the operation reasonably fast. A list cannot
    // be used here, since the get_proposals method needs repeated fast
    // random access.
    for (const auto &proposal : proposals) {
      for (const size_t &index : proposal) {
        if (index >= used_index_markers.size()) {
          throw Forpy_Exception("Invalid index specified!");
        }
        if (!used_index_markers[ index ]) {
          used_index_markers[ index ] = true;
          used_indices -> push_back(index);
          available_indices -> erase(
            std::find(available_indices -> begin(),
                      available_indices -> end(), index));
        }
      }
    }
  }

  size_t FeatureSelector::get_max_to_use() const {
    return max_to_use;
  }

  bool FeatureSelector::operator==(const IFeatureSelector &rhs)
    const {
    const auto *rhs_c = dynamic_cast<FeatureSelector const *>(&rhs);
    if (rhs_c == nullptr)
      return false;
    else {
      bool eq_dim = dimension == rhs_c -> dimension;
      bool eq_hmpn = how_many_per_node == rhs_c -> how_many_per_node;
      bool eq_hmav = how_many_available == rhs_c -> how_many_available;
      bool eq_mtu = max_to_use == rhs_c -> max_to_use;
      bool eq_used = *used_indices == *(rhs_c -> used_indices);
      bool eq_mrks = used_index_markers == rhs_c -> used_index_markers;
      bool eq_av = *available_indices == *(rhs_c -> available_indices);
      bool eq_re = *random_engine == *(rhs_c -> random_engine);
      //std::cout << eq_dim << std::endl;
      //std::cout << eq_hmpn << std::endl;
      //std::cout << eq_hmav << std::endl;
      //std::cout << eq_mtu << std::endl;
      //std::cout << eq_used << std::endl;
      //std::cout << eq_mrks << std::endl;
      //std::cout << eq_av << std::endl;
      //std::cout << eq_re << std::endl;
      return eq_dim &&
             eq_hmpn &&
             eq_hmav &&
             eq_mtu &&
             eq_used &&
             eq_mrks &&
             eq_av &&
             eq_re;
    }
  };

} // namespace forpy
