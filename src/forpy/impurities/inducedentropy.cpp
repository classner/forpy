#include <forpy/impurities/inducedentropy.h>

namespace forpy {
  InducedEntropy::InducedEntropy() {};

  InducedEntropy::InducedEntropy(const float &p) : p(p) {
    if (p <= 0.f) {
      throw Forpy_Exception("p must be > 0.f.");
    }
    tsallis_entropy = std::unique_ptr<TsallisEntropy>(
        new TsallisEntropy());
    tsallis_entropy -> q = p;
  }

  InducedEntropy::~InducedEntropy() {};

  float InducedEntropy::operator()(
      const std::vector<float> &class_members_numbers,
      const float &fsum) const {
    // In debug mode, run checks.
    FASSERT(static_cast<float>(
      std::accumulate(class_members_numbers.begin(),
                      class_members_numbers.end(),
                      static_cast<float>(0))) == fsum);
    FASSERT(safe_pos_sum_lessoe_than(class_members_numbers));

    // Deal with the special case quickly.
    if (fsum == 0.f)
      return 0.f;

    // T_2 has apparently "better" (more suited) numerical instabilities
    // leading to better scores. Both measures are theoretically equivalent
    // for q=2, so use this fact!
    if (p == 2.f)
      return tsallis_entropy -> operator()(class_members_numbers, fsum);

    // Cast only once and save time.
    float n_classes_f = static_cast<float>(class_members_numbers.size());
    float max_unorder_val = 1.f / n_classes_f;
    float entropy_sum;
    if (ceilf(p) == p || floorf(p) == p) {
      // p is a whole number. Use a faster implementation.
      const uint whole_p = static_cast<uint>(p);
      // Offset.
      entropy_sum = fpowi(1.f - max_unorder_val, whole_p) +
                    (n_classes_f - 1.f) * fpowi(max_unorder_val, whole_p);
      // Calculate.
      for (const auto &member_number : class_members_numbers) {
        float quot = static_cast<float>(member_number) / fsum;
        entropy_sum -= fpowi(fabs(quot - max_unorder_val), whole_p);
      }
    } else {
      // p is not a whole number.
      // Offset.
      entropy_sum = powf(1.f - max_unorder_val, p) +
                    (n_classes_f - 1.f) * powf(max_unorder_val, p);
      // Calculate.
      for (const auto &member_number : class_members_numbers) {
        float quot = static_cast<float>(member_number) / fsum;
        entropy_sum -= powf(fabs(quot - max_unorder_val), p);
      }
    }
    return entropy_sum;
  };

  float InducedEntropy::differential_normal(
      const float &det, const uint &dimension) const {
    if (det == 0.f) {
      return 0.f;
    }
    if (det < 0.f) {
      // This is complete bollocks.
      throw Forpy_Exception("Covariance matrix with negative "
                            "determinant occured.");
    }
    float det_clean = std::max(det, DIFFENTROPY_EPS);
    return (1.f-powf(p, -0.5f*static_cast<float>(dimension)) *
      powf(powf(TWO_PI, static_cast<float>(dimension)) * det_clean,
           -0.5f*(p-1.f))) -
      (1.f-powf(p, -0.5f*static_cast<float>(dimension)) *
       powf(powf(TWO_PI, static_cast<float>(dimension)) * DIFFENTROPY_EPS,
            -0.5f*(p-1.f)));
  };

  bool InducedEntropy::operator==(
      const IEntropyFunction &rhs) const {
    const auto *rhs_c = dynamic_cast<InducedEntropy const *>(&rhs);
    if (rhs_c == nullptr) {
      return false;
    } else {
      bool eq_p = p == rhs_c -> p;
      return eq_p;
    }
  };

  float InducedEntropy::get_p() const { return p; };
} // namespace forpy
