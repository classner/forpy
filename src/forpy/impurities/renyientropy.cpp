#include <forpy/impurities/renyientropy.h>

namespace forpy {
  RenyiEntropy::RenyiEntropy() {};

  RenyiEntropy::RenyiEntropy(const float &alpha) : q(alpha) {
    if (q <= 0.f) {
      throw Forpy_Exception("alpha must be > 0.f.");
    }
    shannon_entropy = std::unique_ptr<ShannonEntropy>(new ShannonEntropy());
    induced_p = std::unique_ptr<InducedEntropy>(new InducedEntropy(q));
    classification_error = std::unique_ptr<ClassificationError>(
         new ClassificationError());
  }
  RenyiEntropy::~RenyiEntropy() {};

  float RenyiEntropy::operator()(
      const std::vector<float> &class_members_numbers,
      const float &fsum) const {
    if (q == 1.f) {
      return shannon_entropy -> operator()(class_members_numbers, fsum);
    }
    if (q == std::numeric_limits<float>::infinity()) {
      return -logf(-(classification_error -> operator()(
          class_members_numbers, fsum)-1.f));
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
    float entropy_sum = 0.f;
    float quot;
    if (ceilf(q) == q || floorf(q) == q) {
      // q is a whole number. Use a faster implementation.
      const uint whole_q = static_cast<uint>(q);
      // Calculate.
      for (const auto &member_number : class_members_numbers) {
        quot = static_cast<float>(member_number) / fsum;
        entropy_sum += fpowi(quot, whole_q);
      }
    } else {
      // p is not a whole number.
      // Calculate.
      for (const auto &member_number : class_members_numbers) {
        quot = static_cast<float>(member_number) / fsum;
        entropy_sum += powf(quot, q);
      }
    }
    return logf(entropy_sum) / (1.f - q);
  };

  float RenyiEntropy::differential_normal(
      const float &det, const uint &dimension) const {
    if (det == 0.f) {
      return 0.f;
    }
    if (q == 1.f) {
      return shannon_entropy -> differential_normal(det, dimension);
    } else {
      float det_clean = std::max(det, DIFFENTROPY_EPS);
      if (q == std::numeric_limits<float>::infinity()) {
        return -logf(-((1.f - 1.f / sqrtf(fpowi(TWO_PI, dimension) *
                                          det_clean))-1.f)) +
          +logf(-((1.f - 1.f / sqrtf(fpowi(TWO_PI, dimension) *
                                     DIFFENTROPY_EPS))-1.f));
      } else {
        return logf(-((1.f-powf(q, -0.5f*static_cast<float>(dimension)) *
                       powf(powf(TWO_PI, static_cast<float>(dimension)) * det_clean,
                            -0.5f*(q-1.f)))-1.f))/(1.f-q) -
          logf(-((1.f-powf(q, -0.5f*static_cast<float>(dimension)) *
                  powf(powf(TWO_PI, static_cast<float>(dimension)) * DIFFENTROPY_EPS,
                       -0.5f*(q-1.f)))-1.f))/(1.f-q);
      }
    }
  };

  bool RenyiEntropy::operator==(const IEntropyFunction &rhs) const {
    const auto *rhs_c = dynamic_cast<RenyiEntropy const *>(&rhs);
    if (rhs_c == nullptr) {
      return false;
    } else {
      bool eq_q = q == rhs_c -> q;
      return eq_q;
    }
  };

  float RenyiEntropy::get_alpha() const { return q; };
} // namespace forpy
