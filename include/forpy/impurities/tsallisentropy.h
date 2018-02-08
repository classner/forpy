/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_IMPURITIES_TSALLISENTROPY_H_
#define FORPY_IMPURITIES_TSALLISENTROPY_H_

#include "../global.h"
#include "../util/serialization/basics.h"

#include <numeric>
#include <vector>

#include "../types.h"
#include "../util/checks.h"
#include "../util/exponentials.h"
#include "./ientropyfunction.h"
#include "./inducedentropy.h"
#include "./shannonentropy.h"

namespace forpy {
/**
 * \brief Computes the Tsallis entropy.
 *
 * Works correctly up to a total sum of elements of
 * numeric_limits<float>::max().
 *
 * This is the Tsallis entropy, as introduced by Constantino Tsallis
 * (see http://en.wikipedia.org/wiki/Tsallis_entropy).
 *
 *
 * \ingroup forpyimpuritiesGroup
 */
class TsallisEntropy : public IEntropyFunction {
 public:
  /**
   * \param q float>0.f
   *   The entropy parameter.
   */
  explicit TsallisEntropy(const float &q);
  ~TsallisEntropy();

  inline float operator()(const float *class_members_numbers, const size_t &n,
                          const float &fsum) const {
    if (q == 1.f)
      return shannon_entropy->operator()(class_members_numbers, n, fsum);
    // In debug mode, run checks.
    FASSERT(static_cast<float>(std::accumulate(class_members_numbers,
                                               class_members_numbers + n,
                                               static_cast<float>(0))) == fsum);
    if (fsum == 0.f) return 0.f;
    float entropy_sum = 1.f;
    float quot;
    if (ceilf(q) == q || floorf(q) == q) {
      const uint whole_q = static_cast<uint>(q);
      // Calculate.
      for (size_t i = 0; i < n; ++i) {
        quot = *(class_members_numbers++) / fsum;
        entropy_sum -= fpowi(quot, whole_q);
      }
    } else {
      // Calculate.
      for (size_t i = 0; i < n; ++i) {
        quot = *(class_members_numbers++) / fsum;
        entropy_sum -= powf(quot, q);
      }
    }
    return entropy_sum / (q - 1.f);
  };

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const TsallisEntropy &self) {
    stream << "forpy::TsallisEntropy[q=" << self.get_q() << "]";
    return stream;
  };
  bool operator==(const IEntropyFunction &rhs) const;

  /** Returns q as set in the constructor. */
  float get_q() const;
  friend class InducedEntropy;
  friend class ShannonEntropy;

 private:
  /** \brief DON'T USE. Non-initializing constructor for serialization purposes.
   *
   * It is currently used by the Shannon entropy to break a dependency
   * circle.
   */
  TsallisEntropy();

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(cereal::make_nvp("base", cereal::base_class<IEntropyFunction>(this)),
       CEREAL_NVP(q), CEREAL_NVP(shannon_entropy), CEREAL_NVP(induced_p));
  };

  float q;
  std::unique_ptr<ShannonEntropy> shannon_entropy;
  std::unique_ptr<InducedEntropy> induced_p;
  DISALLOW_COPY_AND_ASSIGN(TsallisEntropy);
};
}  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::TsallisEntropy);
#endif  // FORPY_IMPURITIES_TSALLISENTROPY_H_
