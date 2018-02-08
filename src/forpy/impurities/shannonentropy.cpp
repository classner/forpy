#include <forpy/impurities/shannonentropy.h>

namespace forpy {
ShannonEntropy::ShannonEntropy() {}
ShannonEntropy::~ShannonEntropy() {}

bool ShannonEntropy::operator==(const IEntropyFunction &rhs) const {
  const auto *rhs_c = dynamic_cast<ShannonEntropy const *>(&rhs);
  if (rhs_c == nullptr) {
    return false;
  } else {
    return true;
  }
};
}  // namespace forpy
