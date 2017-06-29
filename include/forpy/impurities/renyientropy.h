/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_IMPURITIES_RENYIENTROPY_H_
#define FORPY_IMPURITIES_RENYIENTROPY_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/access.hpp>

#include <vector>
#include <numeric>
#include <limits>

#include "../global.h"
#include "../types.h"
#include "../util/checks.h"
#include "../util/exponentials.h"
#include "./ientropyfunction.h"
#include "./shannonentropy.h"
#include "./inducedentropy.h"
#include "./classificationerror.h"

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
   *
   * -----
   * Instantiations:
   * - float
   * - uint
   * .
   *
   * -----
   */
  class RenyiEntropy : public IEntropyFunction {
   public:
    /**
     * \param alpha float>0.f
     *   The entropy parameter.
     */
    explicit RenyiEntropy(const float &alpha);
    ~RenyiEntropy();

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
     * \brief Differential Renyi entropy of the normal distribution.
     *
     * Calculates the differential entropy of a normal distribution with the
     * determinant value of the covariance matrix.
     */
    float differential_normal(const float &det, const uint &dimension) const;

    using IEntropyFunction::operator();
    using IEntropyFunction::differential_normal;

    bool operator==(const IEntropyFunction &rhs) const;
 
    /** Returns the alpha value set in the constructor. */
    float get_alpha() const;

   private:
    /** \brief DON'T USE. Non-initializing constructor for serialization purposes. */
    RenyiEntropy();

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar, const uint &) {
      ar(cereal::make_nvp("base",
                          cereal::base_class<IEntropyFunction>(this)),
         CEREAL_NVP(q),
         CEREAL_NVP(shannon_entropy),
         CEREAL_NVP(induced_p),
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
