#include <forpy/impurities/classificationerror.h>
#include <forpy/impurities/inducedentropy.h>
#include <forpy/impurities/renyientropy.h>

namespace forpy {
RenyiEntropy::RenyiEntropy(const float &alpha) : q(alpha) {
  if (q <= 0.f) {
    throw ForpyException("alpha must be > 0.f.");
  }
  shannon_entropy = std::unique_ptr<ShannonEntropy>(new ShannonEntropy());
  induced_p = std::unique_ptr<InducedEntropy>(new InducedEntropy(q));
  classification_error =
      std::unique_ptr<ClassificationError>(new ClassificationError());
}

bool RenyiEntropy::operator==(const IEntropyFunction &rhs) const {
  const auto *rhs_c = dynamic_cast<RenyiEntropy const *>(&rhs);
  if (rhs_c == nullptr) {
    return false;
  } else {
    bool eq_q = q == rhs_c->q;
    return eq_q;
  }
};

float RenyiEntropy::get_alpha() const { return q; };
}  // namespace forpy
