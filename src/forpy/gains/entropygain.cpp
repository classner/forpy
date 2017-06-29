#include <forpy/gains/entropygain.h>

namespace forpy {
  float EntropyGain::approx(const std::vector<float> &members_numbers_left,
                            const std::vector<float> &members_numbers_right) {
    FASSERT(safe_pos_sum_lessoe_than(members_numbers_left));
    FASSERT(safe_pos_sum_lessoe_than(members_numbers_right));
    float sum_left = static_cast<float>(
      std::accumulate(members_numbers_left.begin(),
                      members_numbers_left.end(),
                      static_cast<float>(0)));
    FASSERT(sum_left >= 0.f);
    float sum_right = static_cast<float>(
      std::accumulate(members_numbers_right.begin(),
                      members_numbers_right.end(),
                      static_cast<float>(0)));
    FASSERT(sum_right >= 0.f);
    FASSERT(std::numeric_limits<float>::max() - sum_left >= sum_right);

    auto sum_complete = sum_left + sum_right;
    if (sum_complete == 0.f)
      return 0.f;
    FASSERT(sum_complete > 0.f);
    return - (sum_left  / sum_complete * (*entropy_function)(members_numbers_left,  sum_left)
             +sum_right / sum_complete * (*entropy_function)(members_numbers_right, sum_right));
  };

  float EntropyGain::operator()(const float &current_entropy,
                                const std::vector<float> &members_numbers_left,
                                const std::vector<float> &members_numbers_right) {
    return current_entropy +
      approx(members_numbers_left, members_numbers_right);
  };

  float EntropyGain::operator()(
                   const std::vector<float> &members_numbers_left,
                   const std::vector<float> &members_numbers_right) {
    FASSERT(safe_pos_sum_lessoe_than(members_numbers_left,
                                     members_numbers_right));
    std::vector<float> combined;
    // OPTIMIZE: This is a clear hotspot: it takes about half of the
    // gain calculation time. It is avoided where possible anyway by
    // calling either approx or the other operator() overload.
    combined.reserve(members_numbers_left.size());
    std::transform(members_numbers_left.begin(),
                   members_numbers_left.end(),
                   members_numbers_right.begin(),
                   std::back_inserter(combined),
                   std::plus<float>());
    float entropy = (*entropy_function)(combined);

    return operator()(entropy, members_numbers_left, members_numbers_right);
  };

  bool EntropyGain::operator==(const IGainCalculator &rhs) const {
    const auto *rhs_c = dynamic_cast<EntropyGain const *>(&rhs);
    if (rhs_c == nullptr) {
      return false;
    } else {
      bool eq_ef = *entropy_function == *(rhs_c -> entropy_function);
      return eq_ef;
    }
  };

  std::shared_ptr<IEntropyFunction> EntropyGain::getEntropy_function() const {
    return entropy_function;
  }
} // namespace forpy
