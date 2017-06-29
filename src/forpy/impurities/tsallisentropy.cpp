#include <forpy/impurities/tsallisentropy.h>

namespace forpy {
  TsallisEntropy::TsallisEntropy() {};

  TsallisEntropy::TsallisEntropy(const float &q) : q(q) {
    if (q <= 0.f) {
      throw Forpy_Exception("q must be > 0.f.");
    }

    shannon_entropy = std::unique_ptr<ShannonEntropy>(new ShannonEntropy());
    induced_p = std::unique_ptr<InducedEntropy>(new InducedEntropy(q));
  };

  TsallisEntropy::~TsallisEntropy() {};

  float TsallisEntropy::operator()(
      const std::vector<float> &class_members_numbers,
      const float &fsum) const {
    if (q == 1.f) {
      return shannon_entropy -> operator()(class_members_numbers, fsum);
    }
    // In debug mode, run checks.
    FASSERT(static_cast<float>(
      std::accumulate(class_members_numbers.begin(),
                      class_members_numbers.end(),
                      static_cast<float>(0))) == fsum);
    FASSERT(safe_pos_sum_lessoe_than(class_members_numbers));

    // Deal with the special case quickly.
    if (fsum == 0.f)
      return 0.f;

    // Cast only once and save time.
    float entropy_sum = 1.f;
    float quot;
    if (ceilf(q) == q || floorf(q) == q) {
      // q is a whole number. Use a faster implementation.
      const uint whole_q = static_cast<uint>(q);
      // Calculate.
      for (const auto &member_number : class_members_numbers) {
        quot = static_cast<float>(member_number) / fsum;
        entropy_sum -= fpowi(quot, whole_q);
      }
    } else {
      // p is not a whole number.
      // Calculate.
      for (const auto &member_number : class_members_numbers) {
        quot = static_cast<float>(member_number) / fsum;
        entropy_sum -= powf(quot, q);
      }
    }
    return entropy_sum / (q - 1.f);
  };

  float TsallisEntropy::differential_normal(
      const float &det, const uint &dimension) const {
    if (det == 0.f)
      return 0.f;
    if (q == 1.f) {
      return shannon_entropy -> differential_normal(det, dimension);
    } else {
      return induced_p -> differential_normal(det, dimension) / (q - 1.f);
    }
  }

  bool TsallisEntropy::operator==(const IEntropyFunction &rhs) const {
    const auto *rhs_c = dynamic_cast<TsallisEntropy const *>(&rhs);
    if (rhs_c == nullptr) {
      return false;
    } else {
      bool eq_q = q == rhs_c -> q;
      return eq_q;
    }
  };

  float TsallisEntropy::get_q() const { return q; }
} // namespace forpy
