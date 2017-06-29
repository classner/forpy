/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_THRESHOLD_OPTIMIZERS_ITHRESHOLDOPTIMIZER_H_
#define FORPY_THRESHOLD_OPTIMIZERS_ITHRESHOLDOPTIMIZER_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/access.hpp>

#include <tuple>
#include <utility>

#include "../global.h"
#include "../types.h"
#include "../data_providers/idataprovider.h"
#include "../util/macros.h"

namespace forpy {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
  const float GAIN_EPS = 1E-7;
  const float ENTROPY_EPS = 1E-7;
#pragma clang diagnostic pop
  const MatCRef<float> FORPY_ZERO_MATR(Mat<float>::Zero(0, 1));

  /**
   * \brief Finds an optimal threshold.
   *
   * Is classically used by the \ref ThresholdClassifier to optimize the
   * thresholds (\f$\tau\f$).
   *
   * \ingroup forpythreshold_optimizersGroup
   */
  class IThresholdOptimizer {
   public:
    virtual ~IThresholdOptimizer();

    /**
     * \brief Whether this threshold optimizer can take into account weights
     * during the optimization.
     */
    virtual bool supports_weights() const VIRTUAL(bool);

    /** \brief Validate annotations for usability with this optimizer. \
     * \
     * Checks all provided annotations for validity. \
     */
    virtual void check_annotations(const IDataProvider &dprov)
      VIRTUAL_VOID;

#define FORPY_ITHRESHOPT_EARLYSTOP_DOC \
    /** \
     * \brief Check for early stopping. \
     *\
     * If true is returned, a leaf is created without searching for a threshold.\
     */
#define FORPY_ITHRESHOPT_EARLYSTOP_RET(AT) bool
#define FORPY_ITHRESHOPT_EARLYSTOP_NAME check_for_early_stop
#define FORPY_ITHRESHOPT_EARLYSTOP_PARAMNAMES annotations, node_id
#define FORPY_ITHRESHOPT_EARLYSTOP_PARAMTYPESNNAMES(AT) \
      const MatCRef<AT> annotations, const node_id_t &node_id
#define FORPY_ITHRESHOPT_EARLYSTOP_PARAMTYPESNNAMESNDEF(AT) \
      FORPY_ITHRESHOPT_EARLYSTOP_PARAMTYPESNNAMES(AT)
#define FORPY_ITHRESHOPT_EARLYSTOP_MOD 
      FORPY_DECL(FORPY_ITHRESHOPT_EARLYSTOP, AT, virtual, ;)

    /**
     * \brief Prepares for the optimization routine of a specific node.
     *
     * Must be calles before optimization. Any kind of setup can be performed.
     */
    virtual void prepare_for_optimizing(const size_t &node_id,
                                        const int &num_threads);

#define FORPY_ITHRESHOPT_OPT_DOC  /** Optimize. */
    /** Optimizes a threshold decider for one node.
     *
     * \param selected_data MatCRef<input_dtype>, samples column-wise.
     *     The selected features of the input data in row major format.
     *     Required for regression.
     * \param annotations MatCRef<annotation_dtype>, samples row-wise.
     *     The data labels in consecutive memory. n_samplesx1
     * \param feature_values MatCRef<feature_dtype>
     *     The features in consecutive memory, n_samplesx1.
     * \param node_id node_id_t
     *     The tree node id to optimize for. Default: 0.
     * \param min_samples_at_leaf size_t
     *     Minimum number of samples per leaf. Can be used to early stop.
     *     Default: 0.
     * \param weights 2D MatRef<float>
     *     Sample weights. If no weights should be used, may have size 0x0,
     *     otherwise, 1xn_samples. Default: 0x0.
     * \param suggestion_index int
     *     The index of this optimization call. Can be used to ensure
     *     determinism for randomized optimizers by using it for seeding.
     *     Default: 0.
     * \returns optimized_split_tuple_t<feature_dtype>
     *     The tuple consists of a pair of thresholds for 'less_than' and
     *     'greater_than' criteria, the type of threshold optimized,
     *     number of elements going 'left', number of elements going 'right',
     *     calculated gain and whether a 'valid' split has been found.
     */
#define FORPY_ITHRESHOPT_OPT_RET(IT, FT, AT) optimized_split_tuple_t<FT>
#define FORPY_ITHRESHOPT_OPT_NAME optimize
#define FORPY_ITHRESHOPT_OPT_PARAMNAMES \
      selected_data_matr,\
      feature_values_matr,\
      annotations_matr,\
      node_id,\
      min_samples_at_leaf,\
      weights_matr,\
      suggestion_index
#define FORPY_ITHRESHOPT_OPT_PARAMTYPESNNAMES(IT, FT, AT)  \
      const MatCMCRef<IT> &selected_data_matr,                 \
      const VecCRef<FT> &feature_values_matr,\
      const MatCRef<AT> &annotations_matr,\
      const node_id_t &node_id,\
      const size_t &min_samples_at_leaf,        \
      const VecCRef<float> & weights_matr,\
      const int &suggestion_index
#define FORPY_ITHRESHOPT_OPT_PARAMTYPESNNAMESNDEF(IT, FT, AT)  \
      const MatCMCRef<IT> &selected_data_matr,                 \
      const VecCRef<FT> &feature_values_matr,                \
      const MatCRef<AT> &annotations_matr,                 \
      const node_id_t &node_id=0,                       \
      const size_t &min_samples_at_leaf=0,              \
      const VecCRef<float> & weights_matr=FORPY_ZERO_MATR,                 \
      const int &suggestion_index=0
#define FORPY_ITHRESHOPT_OPT_MOD 
    FORPY_DECL(FORPY_ITHRESHOPT_OPT, ITFTAT, virtual, ;)

    /**
     * \brief Returns the gain threshold to use for this node.
     */
    virtual float get_gain_threshold_for(const size_t &node_id) VIRTUAL(float);
    
    /**
     * Deep equality check.
     */
    virtual bool operator==(const IThresholdOptimizer &rhs) const VIRTUAL(bool);

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive &, const uint &) {}

   protected:
    IThresholdOptimizer();
  };
}  // namespace forpy
#endif  // FORPY_THRESHOLD_OPTIMIZERS_ITHRESHOLDOPTIMIZER_H_
