/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_GAINS_ENTROPYGAIN_H_
#define FORPY_GAINS_ENTROPYGAIN_H_

#include "../global.h"

#include "../util/serialization/basics.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "../impurities/impurities.h"
#include "../util/checks.h"
#include "./igaincalculator.h"

namespace forpy {
/**
 * \brief Calculates the gain as difference of current entropy and the
 * weighted sum of subgroup entropies.
 *
 * Works correctly up to a total sum of elements of
 * min(numeric_limits<float>::max(), numeric_limits<input_dtype>::max())
 * and the limitations of the selected entropy function.
 * Speed optimized function that does no checks in release mode!
 *
 * \param input_dtype The datatype for counting class members. This might
 * be a float if sample weights are used.
 *
 * \ingroup forpygainsGroup
 */
class EntropyGain : public IGainCalculator {
 public:
  /**
   * \param entropy_function shared(IEntropyFunction)
   *   The entropy to use for gain calculation.
   */
  explicit EntropyGain(
      const std::shared_ptr<IEntropyFunction> &entropy_function)
      : entropy_function(entropy_function) {}

  /** Gets a gain approximation that can be used inside an `argmax` function. */
  float approx(const std::vector<float> &members_numbers_left,
               const std::vector<float> &members_numbers_right);

  /** Calculates the information gain. */
  float operator()(const float &current_entropy,
                   const std::vector<float> &members_numbers_left,
                   const std::vector<float> &members_numbers_right);

  /** Calculates the information gain. */
  float operator()(const std::vector<float> &members_numbers_left,
                   const std::vector<float> &members_numbers_right);

  bool operator==(const IGainCalculator &rhs) const;

  std::shared_ptr<IEntropyFunction> getEntropy_function() const;

 protected:
  EntropyGain() {}

 private:
  std::shared_ptr<IEntropyFunction> entropy_function;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(cereal::make_nvp("base", cereal::base_class<IGainCalculator>(this)),
       CEREAL_NVP(entropy_function));
  }
};
};  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::EntropyGain);
#endif  // FORPY_GAINS_ENTROPYGAIN_H_
