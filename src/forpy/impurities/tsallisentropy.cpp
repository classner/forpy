#include <forpy/impurities/tsallisentropy.h>

namespace forpy {
TsallisEntropy::TsallisEntropy(){};

TsallisEntropy::TsallisEntropy(const float &q) : q(q) {
  if (q <= 0.f) {
    throw ForpyException("q must be > 0.f.");
  }

  shannon_entropy = std::unique_ptr<ShannonEntropy>(new ShannonEntropy());
  induced_p = std::unique_ptr<InducedEntropy>(new InducedEntropy(q));
};

TsallisEntropy::~TsallisEntropy(){};

bool TsallisEntropy::operator==(const IEntropyFunction &rhs) const {
  const auto *rhs_c = dynamic_cast<TsallisEntropy const *>(&rhs);
  if (rhs_c == nullptr) {
    return false;
  } else {
    bool eq_q = q == rhs_c->q;
    return eq_q;
  }
};

float TsallisEntropy::get_q() const { return q; }
}  // namespace forpy
