#include <forpy/impurities/shannonentropy.h>
#include <forpy/threshold_optimizers/ithresholdoptimizer.h>

namespace forpy {
  ShannonEntropy::ShannonEntropy() {}
  ShannonEntropy::~ShannonEntropy() {}

  float ShannonEntropy::operator()(
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

    float entropy_sum = 0.f;
    float quot;
    // Calculate classical entropy.
    for (const auto &member_number : class_members_numbers) {
      // The value -0*log2(0) is defined to be 0, according to the limit
      // lim_{p->0}{p*log2(p)}.
      if (member_number == 0)
        continue;
      // All good, so calculate the normal parts.
      quot = static_cast<float>(member_number) / fsum;
      entropy_sum -= quot * logf(quot);
    }
    return entropy_sum / LOG2VAL;
  };

  float ShannonEntropy::differential_normal(
      const float &det, const uint &dimensions) const {
    FASSERT(dimensions > 0);
    float dimensionsf = static_cast<float>(dimensions);
    //if (det == 0.f) {
    //  return 0.f;
    //}
    //    if (det < 0.f) {
      // This is complete bollocks.
      //throw Forpy_Exception("Covariance matrix with negative "
      //                      "determinant occured.");
    //}
    //float det_clean = std::max(det, DIFFENTROPY_EPS);
    /*std::cout << 0.5f * (dimensionsf * logf(TWO_PI_E) + det) << std::endl;
    std::cout << logf(TWO_PI_E) << std::endl;
    std::cout << dimensionsf * logf(TWO_PI_E) << std::endl;
    std::cout << (dimensionsf * logf(TWO_PI_E) - 10.f * dimensionsf) << std::endl;
    std::cout << (0.5f * (dimensionsf * logf(TWO_PI_E) + (-10.f * dimensionsf))) << std::endl;*/
    return 0.5f * (dimensionsf * logf(TWO_PI_E) + det) -  \
      (0.5f * (dimensionsf * logf(TWO_PI_E) + (logf(ENTROPY_EPS) * dimensionsf)));
  }

  bool ShannonEntropy::operator==(const IEntropyFunction &rhs) const {
    const auto *rhs_c = dynamic_cast<ShannonEntropy const *>(&rhs);
    if (rhs_c == nullptr) {
      return false;
    } else {
      return true;
    }
  };
} // namespace forpy
