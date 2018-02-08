/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_IMPURITIES_RENYIENTROPY_H_
#define FORPY_IMPURITIES_RENYIENTROPY_H_

#include "../global.h"

#include "../util/serialization/basics.h"

#include <limits>
#include <numeric>
#include <vector>

#include "../types.h"
#include "../util/checks.h"
#include "../util/exponentials.h"
#include "./classificationerror.h"
#include "./ientropyfunction.h"
#include "./inducedentropy.h"
#include "./shannonentropy.h"

namespace forpy {
/**
 * \brief Computes the Renyi entropy.
 *
 * Works correctly up to a total sum of elements of
 * numeric_limits<float>::max().
 *
 * This is the Renyi entropy, as introduced by Alfred Renyi
 * (see http://en.wikipedia.org/wiki/R%C3%A9nyi_entropy).
 *
 * \ingroup forpyimpuritiesGroup
 */
class RenyiEntropy : public IEntropyFunction {
 public:
  explicit RenyiEntropy(const float &alpha);
  inline ~RenyiEntropy(){};

  inline virtual float operator()(const float *class_members_numbers,
                                  const size_t &n, const float &fsum) const {
    if (q == 1.f) {
      return shannon_entropy->operator()(class_members_numbers, n, fsum);
    }
    if (q == std::numeric_limits<float>::infinity()) {
      return -logf(
          -(classification_error->operator()(class_members_numbers, n, fsum) -
            1.f));
    }
    // In debug mode, run checks.
    FASSERT(static_cast<float>(std::accumulate(class_members_numbers,
                                               class_members_numbers + n,
                                               static_cast<float>(0))) == fsum);
    // Deal with the special case quickly.
    if (fsum == 0.f) return 0.f;
    // Cast only once and save time.
    float entropy_sum = 0.f;
    float quot;
    if (ceilf(q) == q || floorf(q) == q) {
      // q is a whole number. Use a faster implementation.
      const uint whole_q = static_cast<uint>(q);
      for (size_t i = 0; i < n; ++i) {
        quot = *(class_members_numbers++) / fsum;
        entropy_sum += fpowi(quot, whole_q);
      }
    } else {
      // p is not a whole number.
      // Calculate.
      for (size_t i = 0; i < n; ++i) {
        quot = *(class_members_numbers++) / fsum;
        entropy_sum += powf(quot, q);
      }
    }
    return logf(entropy_sum) / (1.f - q);
  }

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const RenyiEntropy &self) {
    stream << "forpy::RenyiEntropy[alpha=" << self.get_alpha() << "]";
    return stream;
  };
  bool operator==(const IEntropyFunction &rhs) const;

  /** Returns the alpha value set in the constructor. */
  float get_alpha() const;

 private:
  /** \brief DON'T USE. Non-initializing constructor for serialization purposes.
   */
  inline RenyiEntropy(){};

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(cereal::make_nvp("base", cereal::base_class<IEntropyFunction>(this)),
       CEREAL_NVP(q), CEREAL_NVP(shannon_entropy), CEREAL_NVP(induced_p),
       CEREAL_NVP(classification_error));
  };

  float q;
  std::unique_ptr<ShannonEntropy> shannon_entropy;
  std::unique_ptr<InducedEntropy> induced_p;
  std::unique_ptr<ClassificationError> classification_error;
  DISALLOW_COPY_AND_ASSIGN(RenyiEntropy);
};
}  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::RenyiEntropy);
#endif  // FORPY_IMPURITIES_RENYIENTROPY_H_
