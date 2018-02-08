/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_IMPURITIES_SHANNONENTROPY_H_
#define FORPY_IMPURITIES_SHANNONENTROPY_H_

#include "../global.h"

#include "../util/serialization/basics.h"

#include <numeric>
#include <vector>

#include "../types.h"
#include "../util/checks.h"
#include "../util/exponentials.h"
#include "./ientropyfunction.h"

namespace forpy {
/**
 * \brief Computes the classical Shannon-Entropy.
 *
 * Works correctly up to a total sum of elements of
 * numeric_limits<float>::max().
 *
 * For classes \f$C={c_1, \ldots, c_n}\f$, the Shannon entropy is defined as
 * \f[-\sum_{c\in C}p_z\cdot \log_2 p_z.\f]
 *
 * The differential Shannon entropy for a normal distribution with covariance
 * matrix \f$\Sigma\f$ in \f$n\f$ dimensions is defined as
 * \f[\frac{1}{2}\log\left((2\pi e)^n\left|\Sigma\right|\right).\f]
 *
 * \ingroup forpyimpuritiesGroup
 */
class ShannonEntropy : public IEntropyFunction {
 public:
  ShannonEntropy();
  ~ShannonEntropy();

  inline virtual float operator()(const float *class_members_numbers,
                                  const size_t &n, const float &fsum) const {
    // In debug mode, run checks.
    FASSERT(static_cast<float>(std::accumulate(class_members_numbers,
                                               class_members_numbers + n,
                                               static_cast<float>(0))) == fsum);
    // Deal with the special case quickly.
    if (fsum == 0.f) return 0.f;

    float entropy_sum = 0.f;
    float quot;
    // Calculate classical entropy.
    for (size_t i = 0; i < n; ++i, class_members_numbers++) {
      // The value -0*log2(0) is defined to be 0, according to the limit
      // lim_{p->0}{p*log2(p)}.
      if (*class_members_numbers == 0.f) continue;
      // All good, so calculate the normal parts.
      quot = *class_members_numbers / fsum;
      entropy_sum -= quot * log2f(quot);
    }
    return entropy_sum;
  };

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const ShannonEntropy & /*self*/) {
    stream << "forpy::ShannonEntropy";
    return stream;
  };
  bool operator==(const IEntropyFunction &rhs) const;

 private:
  DISALLOW_COPY_AND_ASSIGN(ShannonEntropy);

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(cereal::make_nvp("base", cereal::base_class<IEntropyFunction>(this)));
  }
};
}  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::ShannonEntropy);
#endif  // FORPY_IMPURITIES_SHANNONENTROPY_H_
