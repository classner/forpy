/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_IMPURITIES_INDUCEDENTROPY_H_
#define FORPY_IMPURITIES_INDUCEDENTROPY_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/access.hpp>

#include <vector>
#include <limits>

#include "../global.h"
#include "../util/checks.h"
#include "../util/exponentials.h"
#include "./ientropyfunction.h"

namespace forpy {
  // Forward declaration.
  class TsallisEntropy;

  /**
   * \brief Computes the induced p entropy.
   *
   * Works correctly up to a total sum of elements of
   * numeric_limits<float>::max().
   *
   * This is the induced p-metric of the vector of \f$n\f$ class probabilities
   * and the point of maximum unorder (the vector with all entries
   * \f$\frac{1}{n}\f$) in the n-dimensional space without applying the root.
   * It is equal to the Gini-measure for \f$p=2\f$.
   *
   * The definition for \f$c\f$ classes:
   * \f[\sum_{i=1}^{c} \left\Vert p_i - 0.5\right\Vert ^p\f].
   *
   * The differential entropy for a normal distribution with covariance
   * matrix \f$\Sigma\f$ in \f$n\f$ dimensions is defined as:
   * \f[\frac{1}{\sqrt{p^n}}\cdot\left(\sqrt{2\pi}^n\cdot\sqrt{\left|\Sigma\right|}\right)^{-(p-1)}\f]
   *
   * In the differential normal case, the most useful values for \f$p\f$ are
   * very close to 1 (e.g. 1.00001)! \f$p=2\f$ is already equivalent to the
   * infinite norm!
   *
   * \ingroup forpyimpuritiesGroup
   */
  class InducedEntropy : public IEntropyFunction {
   public:

    /**
     * \param p float>0.f
     *   The entropy parameter value.
     */
    explicit InducedEntropy(const float &p);
    ~InducedEntropy();

    /**
     * The method is supposed to be fast and hence no checking for the validity
     * of fsum is done in release mode!
     *
     * \param class_member_numbers Vector  The class member counts (class histogram).
     * \param fsum float The total of the class_member_numbers Vector.
     * \return The calculated entropy value.
     */
    float operator()(const std::vector<float> &class_members_numbers,
                     const float &fsum) const;

    /**
     * \brief Differential induced p entropy of the normal distribution.
     *
     * Calculates the differential entropy of a normal distribution with the
     * determinant value of the covariance matrix.
     */
    float differential_normal(const float &det, const uint &dimension) const;

    using IEntropyFunction::operator();
    using IEntropyFunction::differential_normal;

    bool operator==(const IEntropyFunction &rhs) const;

    /** Returns the parameter value as set in the constructor. */
    float get_p() const;

   private:
    InducedEntropy();

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive &ar, const uint &) {
      ar(cereal::make_nvp("base",
                          cereal::base_class<IEntropyFunction>(this)),
         CEREAL_NVP(p),
         CEREAL_NVP(tsallis_entropy));
    }

    std::unique_ptr<TsallisEntropy> tsallis_entropy;
    float p;
    DISALLOW_COPY_AND_ASSIGN(InducedEntropy);
  };
}  // namespace forpy


CEREAL_REGISTER_TYPE(forpy::InducedEntropy);
// This include must be made after the definition of the InducedEntropy
// to break a dependency circle.
#include "./tsallisentropy.h"
#endif  // FORPY_IMPURITIES_INDUCEDENTROPY_H_
