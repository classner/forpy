/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_GAINS_IGAINCALCULATOR_H_
#define FORPY_GAINS_IGAINCALCULATOR_H_

#include "../global.h"

#include "../util/serialization/basics.h"

#include <vector>

#include "../types.h"

namespace forpy {
/**
 * \brief Interface for a gain calculator class.
 *
 * A gain calculator must have an overloaded () operator with the
 * according parameters and an approx function that returns a fast
 * approximation of the gain (can return the original gain if no
 * approximation is available). The approximation is used to find the
 * best split position and only for that the actual gain is computed.
 *
 * \param counting_dtype The datatype for counting class members. This might
 * be a float if sample weights are used.
 *
 * \ingroup forpygainsGroup
 */
class IGainCalculator {
 public:
  virtual ~IGainCalculator() {}

  /** Calculates the exact gain for the two subsets. */
  virtual float operator()(const std::vector<float> &members_numbers_left,
                           const std::vector<float> &members_numbers_right)
      VIRTUAL(float);

  /** Calculates the exact gain for the two subsets and uses the
      provided `current_entropy`. */
  virtual float operator()(const float &current_entropy,
                           const std::vector<float> &members_numbers_left,
                           const std::vector<float> &members_numbers_right)
      VIRTUAL(float);

  /** Calculates an approximation for the gain of the two subsets that can be
      used inside an `argmax` function. */
  virtual float approx(const std::vector<float> &members_numbers_left,
                       const std::vector<float> &members_numbers_right)
      VIRTUAL(float);

  /**
   * Deep equality comparison.
   */
  virtual bool operator==(const IGainCalculator &rhs) const VIRTUAL(bool);

 protected:
  IGainCalculator() {}

 private:
  DISALLOW_COPY_AND_ASSIGN(IGainCalculator);

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &, const uint &) {}
};
};      // namespace forpy
#endif  // FORPY_GAINS_IGAINCALCULATOR_H_
