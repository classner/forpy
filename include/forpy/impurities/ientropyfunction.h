/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_IMPURITIES_IENTROPYFUNCTION_H_
#define FORPY_IMPURITIES_IENTROPYFUNCTION_H_

#include "../global.h"

#include "../util/serialization/basics.h"

#include <vector>

#include "../types.h"

namespace forpy {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
static const float ENTROPY_EPS = 1E-7;
#pragma clang diagnostic pop

/**
 * \brief Interface for an entropy calculation functor.
 *
 * \ingroup forpyimpuritiesGroup
 */
class IEntropyFunction {
 public:
  virtual ~IEntropyFunction();

  /**
   * \brief The interface function that must be implemented.
   *
   * Calculates the entropy from a given class distribution. For maximum
   * efficiency, the total weight of samples may be provided as float.
   *
   * \param class_members_numbers Class distribution histogram.
   * \param fsum The total number/weight of samples.
   * \return The calculated entropy value.
   */
  virtual float operator()(const std::vector<float> &class_members_numbers,
                           const float &fsum) const {
    return operator()(&class_members_numbers[0], class_members_numbers.size(),
                      fsum);
  };

  virtual float operator()(const float *class_members_numbers,
                                  const size_t &n, const float &fsum) const
      VIRTUAL(float);

  /**
   * \brief Classical entropy calculation function.
   *
   * Is implemented already and provides a shortcut for for the standard
   * function by calculating the sum of the class distribution.
   *
   * \param class_members_numbers Class distribution histogram.
   * \return The calculated entropy value.
   */
  virtual float operator()(
      const std::vector<float> &class_members_numbers) const;

  /**
   * Deep equality comparison.
   */
  virtual bool operator==(const IEntropyFunction &rhs) const VIRTUAL(bool);

 protected:
  IEntropyFunction();

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &, const uint &) {}

  DISALLOW_COPY_AND_ASSIGN(IEntropyFunction);
};
}  // namespace forpy
#endif  // FORPY_IMPURITIES_IENTROPYFUNCTION_H_
