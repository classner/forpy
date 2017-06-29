/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_IMPURITIES_CLASSIFICATIONERROR_H_
#define FORPY_IMPURITIES_CLASSIFICATIONERROR_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/access.hpp>

#include <vector>
#include <limits>

#include "../global.h"

#include "../util/exponentials.h"
#include "./ientropyfunction.h"

namespace forpy {
  /**
   * \brief Computes the classification error as 1-\max(p_i).
   *
   * \ingroup forpyimpuritiesGroup
   */
  class ClassificationError : public IEntropyFunction {
   public:
    ClassificationError();
    ~ClassificationError();

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
     * \brief Classification error equivalent for the normal distribution.
     *
     * Calculates the differential entropy of a normal distribution with the
     * specified determinant value of the covariance matrix.
     */
    float differential_normal(const float &det, const uint &dimensions) const;

    using IEntropyFunction::operator();
    using IEntropyFunction::differential_normal;

    bool operator==(const IEntropyFunction &rhs) const;

   private:
    DISALLOW_COPY_AND_ASSIGN(ClassificationError);

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar, const uint &) {
      ar(cereal::make_nvp("base",
                          cereal::base_class<IEntropyFunction>(this)));
    }
  };
}  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::ClassificationError);
#endif  // FORPY_IMPURITIES_CLASSIFICATIONERROR_H_
