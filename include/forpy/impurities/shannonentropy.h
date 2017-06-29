/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_IMPURITIES_SHANNONENTROPY_H_
#define FORPY_IMPURITIES_SHANNONENTROPY_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/access.hpp>

#include <vector>
#include <numeric>

#include "../global.h"
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
   * \ingroup fertilizedimpuritiesGroup
   */
  class ShannonEntropy : public IEntropyFunction {
   public:
    ShannonEntropy();
    ~ShannonEntropy();

    /**
     * The method is supposed to be fast and hence no checking for the validity
     * of fsum is done in release mode!
     *
     * \param class_member_numbers Vector 
     *   The class member counts (class histogram).
     * \param fsum float 
     *   The total of the class_member_numbers Vector.
     * \return The calculated entropy value.
     */
    float operator()(const std::vector<float> &class_members_numbers,
                     const float &fsum) const;
    /**
     * \brief Differential shannon entropy of the normal distribution.
     *
     * Calculates the differential entropy of a normal distribution with the
     * specified determinant value of the covariance matrix.
     */
    float differential_normal(const float &det, const uint &dimensions) const;

    using IEntropyFunction::operator();
    using IEntropyFunction::differential_normal;

    bool operator==(const IEntropyFunction &rhs) const;
   private:
    DISALLOW_COPY_AND_ASSIGN(ShannonEntropy);

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar, const uint &) {
      ar(cereal::make_nvp("base",
                          cereal::base_class<IEntropyFunction>(this)));
    }
  };
}  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::ShannonEntropy);
#endif  // FORPY_IMPURITIES_SHANNONENTROPY_H_
