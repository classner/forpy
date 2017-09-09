/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_FEATURES_ALIGNEDSURFACECALCULATOR_H_
#define FORPY_FEATURES_ALIGNEDSURFACECALCULATOR_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/access.hpp>

#include <unordered_set>
#include <random>
#include <vector>
#include <algorithm>

#include "../global.h"
#include "../types.h"
#include "../util/sampling.h"
#include "./isurfacecalculator.h"
#include "./featcalcparams.h"

namespace forpy {
  /**
   * \brief Forwards the data as features.
   *
   * Does not require any parameters.
   *
   * \ingroup forpyfeaturesGroup
   */
  class AlignedSurfaceCalculator : public ISurfaceCalculator {
   public:
    AlignedSurfaceCalculator();

    std::vector<FeatCalcParams> propose_params();
    size_t required_num_features() const;

    FORPY_DECL(FORPY_ISURFCALC_CALC, ITFTEQ, , ;);

    FORPY_DECL(FORPY_ISURFCALC_CALCS, ITFTEQ, , ;);

    inline friend std::ostream &operator<<(std::ostream &stream,
                                           const AlignedSurfaceCalculator &/*self*/) {
      stream << "forpy::AlignedSurfaceCalculator";
      return stream;
    };

    bool operator==(const ISurfaceCalculator &rhs) const;

   private:
    FORPY_DECL_IMPL(FORPY_ISURFCALC_CALC, ITFTEQ, ;);

    FORPY_DECL_IMPL(FORPY_ISURFCALC_CALCS, ITFTEQ, ;);

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive &ar, const uint &) {
      ar(cereal::make_nvp("base",
                          cereal::base_class<ISurfaceCalculator>(this)));
    }
  };
}  // namespace forpy
CEREAL_REGISTER_TYPE(forpy::AlignedSurfaceCalculator);
#endif  // FORPY_FEATURES_ALIGNEDSURFACECALCULATOR_H_
