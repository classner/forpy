#include <forpy/features/featureproposer.h>

namespace forpy {

  FeatureProposer::FeatureProposer(
      const size_t &dimension,
      const size_t &index_max,
      const size_t &how_many_per_node,
      std::shared_ptr<std::vector<size_t>> used_indices,
      std::shared_ptr<std::vector<size_t>> available_indices,
      std::shared_ptr<std::mt19937> random_engine)
    : dimension(dimension),
      index_max(index_max),
      how_many_per_node(how_many_per_node),
      used_indices(used_indices),
      available_indices(available_indices),
      random_engine(random_engine),
      generated(0) {
    if (dimension == 1) {
      sampler = std::unique_ptr<SamplingWithoutReplacement<size_t>>(
          new SamplingWithoutReplacement<size_t>(0, index_max, random_engine));
    } else {
      already_used = proposal_set_t();
      already_used.reserve(how_many_per_node);
    }
  }

  bool FeatureProposer::available() const {
    return generated < how_many_per_node;
  };

  size_t FeatureProposer::max_count() const {
    return how_many_per_node;
  }

  std::vector<size_t> FeatureProposer::get_next() {
    // Generate the selection.
    if (dimension == 1) {
      if (generated >= how_many_per_node)
        throw Forpy_Exception("Tried to generate more feature comb. "
          "for a node than there are available.");
      generated++;
      size_t next_preselection = sampler -> get_next();
      if (next_preselection < used_indices -> size()) {
        return std::vector<size_t>(1, used_indices -> at(next_preselection));
      } else {
        return std::vector<size_t>(1, available_indices -> at(
          available_indices -> size() -
          (next_preselection- used_indices -> size()) - 1));
      }
    } else {
      if (generated >= how_many_per_node)
        throw Forpy_Exception("Tried to generate more feature comb. "
          "for a node than there are available.");
      generated++;
      std::vector<size_t> selection(dimension), preselection;
        do {
          preselection = unique_indices<size_t>(dimension, 0, index_max, random_engine.get());
          for (size_t j = 0; j < static_cast<size_t>(dimension); ++j) {
              if (preselection[j] < used_indices -> size()) {
                selection[j] = used_indices -> at(preselection[j]);
              } else {
                selection[j] = available_indices -> at(
                                 available_indices -> size() -
                                 (preselection[j]- used_indices -> size()) - 1);
              }
            }
        } while (already_used.find(selection) != already_used.end());
        already_used.emplace(selection);
        return selection;
    }
  };

  bool FeatureProposer::operator==(const IFeatureProposer &rhs) const {
    const auto *rhs_c = dynamic_cast<FeatureProposer const *>(&rhs);
    if (rhs_c == nullptr) {
      return false;
    } else {
      bool eq_dim = dimension == rhs_c->dimension;
      bool eq_idm = index_max == rhs_c->index_max;
      bool eq_hmpn = how_many_per_node == rhs_c->how_many_per_node;
      bool eq_used_indices = used_indices == rhs_c->used_indices;
      bool eq_av_id = available_indices == rhs_c->available_indices;
      bool eq_smp = sampler == rhs_c->sampler;
      bool eq_re = random_engine == rhs_c->random_engine;
      bool eq_au = already_used == rhs_c->already_used;
      bool eq_gen = generated == rhs_c->generated;

      return eq_dim && eq_idm && eq_hmpn && eq_used_indices && eq_av_id &&
        eq_smp && eq_re && eq_au && eq_gen;
    }
  };

} // namespace forpy
