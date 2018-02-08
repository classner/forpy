#include <forpy/impurities/classificationerror.h>

namespace forpy {
ClassificationError::ClassificationError() {}
ClassificationError::~ClassificationError() {}

bool ClassificationError::operator==(const IEntropyFunction &rhs) const {
  const auto *rhs_c = dynamic_cast<ClassificationError const *>(&rhs);
  if (rhs_c == nullptr) {
    return false;
  } else {
    return true;
  }
};
}  // namespace forpy
