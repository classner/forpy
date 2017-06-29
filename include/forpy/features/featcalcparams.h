/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_FEATURES_FEATCALCPARAMS_H_
#define FORPY_FEATURES_FEATCALCPARAMS_H_

#include <cereal/access.hpp>

#include "../global.h"
#include "../types.h"

namespace forpy {
  /**
   * \brief Can be specialized to any necessary parameters used by a feature
   * calculator.
   *
   * This was planned as an implementable interface, however it must be a
   * single, plain POD object, since it might be transferred forth and back
   * between host and MIC device.
   */
  struct FeatCalcParams {
    float weights[9];
    float offsets[2];

    bool operator==(const FeatCalcParams &rhs) const;

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive &ar, const uint &) {
      ar(CEREAL_NVP(weights),
         CEREAL_NVP(offsets));
    };
  };
}  // namespace forpy
#endif  // FORPY_FEATURES_FEATCALCPARAMS_H_
