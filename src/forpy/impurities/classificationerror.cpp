#include <forpy/impurities/classificationerror.h>

namespace forpy {
  ClassificationError::ClassificationError() {}
  ClassificationError::~ClassificationError() {}

  float ClassificationError::operator()(
      const std::vector<float> &class_members_numbers,
      const float &fsum) const {
    // Deal with the special case quickly.
    if (fsum == 0.f)
      return 0.f;

    return 1.f - static_cast<float>(
        *std::max_element(class_members_numbers.begin(),
                          class_members_numbers.end())) / fsum;
  };

  float ClassificationError::differential_normal(
      const float &det, const uint &dimensions) const {
    FASSERT(dimensions > 0);
    if (det == 0.f) {
      return 0.f;
    }
    if (det < 0.f) {
      // This is complete bollocks.
      throw Forpy_Exception("Covariance matrix with negative "
                            "determinant occured.");
    }
    float det_clean = std::max(det, DIFFENTROPY_EPS);
    return (1.f - 1.f / sqrtf(fpowi(TWO_PI, dimensions) * det_clean)) -\
      (1.f - 1.f / sqrtf(fpowi(TWO_PI, dimensions) * DIFFENTROPY_EPS));
  }

  bool ClassificationError::operator==(const IEntropyFunction &rhs) const {
    const auto *rhs_c = dynamic_cast<ClassificationError const *>(&rhs);
    if (rhs_c == nullptr) {
      return false;
    } else {
      return true;
    }
  };
} // namespace forpy
