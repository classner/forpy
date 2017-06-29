/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_FEATURES_ISURFACECALCULATOR_H_
#define FORPY_FEATURES_ISURFACECALCULATOR_H_
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
#include "../util/macros.h"
#include "../data_providers/idataprovider.h"
#include "./featcalcparams.h"

namespace forpy {
  static const std::vector<size_t> FORPY_EMPTY_VEC(0);

  /**
   * \brief Calculates scalar values from a set of provided data dimensions.
   *
   * Must also propose parameter-sets for a random optimization of itself.
   *
   * \ingroup forpyfeaturesGroup
   */
  class ISurfaceCalculator {
   public:
    virtual ~ISurfaceCalculator();

    /**
     * \brief Propose a set of parameter configurations.
     *
     * Even if the feature calculator has no parameters, this method should
     * return a one-element vector with an empty set for the optimization
     * process.
     */
    virtual std::vector<FeatCalcParams> propose_params()
      VIRTUAL(std::vector<FeatCalcParams>);

    /**
     * \brief Checks compatibility to a specific data provider.
     */
    virtual bool is_compatible_to(const IDataProvider &dprov) const;

    /**
     * \brief Whether this data provider needs all inputs contiguously in one
     * array.
     *
     * The direct patch difference surface calculator does not require this.
     */
    virtual bool needs_elements_prepared() const;

#define FORPY_ISURFCALC_CALC_DOC 
    /**
      * \brief Get the scalar feature representations for a set of data
      * tuples.
      *
      * \param data_dimension_selection A selection of data dimensions to
      *         calculate the feature on.
      * \param parameter_set A parameter set to use.
      * \param data_provider A data provider that can provide the samples.
      * \param element_ids The ids of the samples of which the features
      *         should be calculated.
      * \return A vector of one value per sample.
      */
#define FORPY_ISURFCALC_CALC_RET(IT, FT) void
#define FORPY_ISURFCALC_CALC_NAME calculate
#define FORPY_ISURFCALC_CALC_PARAMNAMES data, retval, feature_selection, samples, element_ids, parameter_set
#define FORPY_ISURFCALC_CALC_PARAMTYPESNNAMES(IT, FT)  \
    const std::shared_ptr<const MatCM<IT>> &data,        \
    std::shared_ptr<const MatCM<FT>> &retval,          \
    const std::vector<size_t> &feature_selection,  \
    const SampleVec<Sample> &samples, \
    const elem_id_vec_t &element_ids, \
    const FeatCalcParams &parameter_set
#define FORPY_ISURFCALC_CALC_PARAMTYPESNNAMESNDEF(IT, FT) \
    const std::shared_ptr<const MatCM<IT>> &data,           \
    std::shared_ptr<const MatCM<FT>> &retval,             \
    const std::vector<size_t> &feature_selection=FORPY_EMPTY_VEC, \
    const SampleVec<Sample> &samples=SampleVec<Sample>(),\
    const elem_id_vec_t &element_ids=FORPY_EMPTY_VEC, \
    const FeatCalcParams &parameter_set=FeatCalcParams()
#define FORPY_ISURFCALC_CALC_MOD const
    FORPY_DECL(FORPY_ISURFCALC_CALC, ITFT, virtual, ;)
 
#define FORPY_ISURFCALC_CALCS_DOC 
    /**
     * \brief Get the scalar feature representation for one sample.
     */
#define FORPY_ISURFCALC_CALCS_RET(IT, FT) void
#define FORPY_ISURFCALC_CALCS_NAME calculate_pred
#define FORPY_ISURFCALC_CALCS_PARAMNAMES data, retval, feature_selection, parameter_set
#define FORPY_ISURFCALC_CALCS_PARAMTYPESNNAMES(IT, FT) \
    const MatCRef<IT> &data,                      \
    FT *retval, \
    const std::vector<size_t> &feature_selection, \
    const FeatCalcParams &parameter_set
#define FORPY_ISURFCALC_CALCS_PARAMTYPESNNAMESNDEF(IT, FT)          \
    const MatCRef<IT> &data,                                      \
    FT *retval, \
    const std::vector<size_t> &feature_selection=FORPY_EMPTY_VEC, \
    const FeatCalcParams &parameter_set=FeatCalcParams()
#define FORPY_ISURFCALC_CALCS_MOD const
    FORPY_DECL(FORPY_ISURFCALC_CALCS, ITFT, virtual, ;)

    /**
     * \brief Get the number of data dimensions required to calculate the
     * the feature.
     */
    virtual size_t required_num_features() const VIRTUAL(size_t);

    virtual bool operator==(const ISurfaceCalculator &rhs) const VIRTUAL(bool);

   protected:
    ISurfaceCalculator();

   private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive &, const uint &) {};
  };
}  // namespace forpy
#endif // FORPY_FEATURES_ISURFACECALCULATOR_H_
