/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_IMPURITIES_TSALLISENTROPY_H_
#define FORPY_IMPURITIES_TSALLISENTROPY_H_

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

    /**
     * The method is supposed to be fast and hence no checking for the validity
     * of fsum is done in release mode!
     *
     * \param class_member_numbers Class distribution histogram.
     * \return The calculated entropy value.
     */
    float operator()(const std::vector<float> &class_members_numbers,
                     const float &fsum) const;

    /**
     * \brief Differential Tsallis entropy of the normal distribution.
     *
     * Calculates the differential entropy of a normal distribution with the
     * determinant value of the covariance matrix.
     */
    float differential_normal(const float &det, const uint &dimension) const;

    using IEntropyFunction::operator();
    using IEntropyFunction::differential_normal;

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
    template<class Archive>
    void serialize(Archive & ar, const uint &) {
      ar(cereal::make_nvp("base",
                          cereal::base_class<IEntropyFunction>(this)),
         CEREAL_NVP(q),
         CEREAL_NVP(shannon_entropy),
         CEREAL_NVP(induced_p));
    };

    float q;
    std::unique_ptr<ShannonEntropy> shannon_entropy;
    std::unique_ptr<InducedEntropy> induced_p;
    DISALLOW_COPY_AND_ASSIGN(TsallisEntropy);
  };
}  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::TsallisEntropy);
#endif  // FORPY_IMPURITIES_TSALLISENTROPY_H_
