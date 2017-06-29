/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_IMPURITIES_IENTROPYFUNCTION_H_
#define FORPY_IMPURITIES_IENTROPYFUNCTION_H_

#include <cereal/access.hpp>

#include <vector>

#include "../global.h"
#include "../types.h"

namespace forpy {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
  static const float DIFFENTROPY_EPS = 1E-5f;
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
     * efficiency, the number of total samples may be provided as float.
     *
     * \param class_member_numbers Class distribution histogram.
     * \param fsum The total number of samples.
     * \return The calculated entropy value.
     */
    virtual float operator()(
      const std::vector<float> &class_members_numbers, const float &fsum)
      const VIRTUAL(float);

    /**
     * \brief Classical entropy calculation function.
     *
     * Is implemented already and provides a shortcut for for the standard
     * function by calculating the sum of the class distribution.
     *
     * \param class_member_numbers Class distribution histogram.
     * \return The calculated entropy value.
     */
    virtual float operator()(
      const std::vector<float> &class_members_numbers) const;

    /**
     * \brief Differential entropy of the normal distribution.
     *
     * Calculates the differential entropy of a normal distribution with the
     * specified covariance matrix.
     */
    virtual float differential_normal(const MatCRef<float> &covar_matrix) const;

    /**
     * \brief Differential entropy of the normal distribution.
     *
     * Calculates the differential entropy of a normal distribution with the
     * specified determinant and dimension of the covariance matrix.
     */
    virtual float differential_normal(const float &det, const uint &dim)
      const VIRTUAL(float);

    /**
     * Deep equality comparison.
     */
    virtual bool operator==(const IEntropyFunction &rhs)
      const VIRTUAL(bool);

   protected:
    IEntropyFunction();

   private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive &, const uint &) {}

    DISALLOW_COPY_AND_ASSIGN(IEntropyFunction);
  };
}  // namespace forpy
#endif  // FORPY_IMPURITIES_IENTROPYFUNCTION_H_
