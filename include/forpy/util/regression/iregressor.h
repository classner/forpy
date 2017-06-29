/* Author: Moritz Einfalt, Christoph Lassner. */
#pragma once
#ifndef FORPY_UTIL_REGRESSION_IREGRESSOR_H_
#define FORPY_UTIL_REGRESSION_IREGRESSOR_H_

#include <cereal/access.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>

#include "../../global.h"
#include "../../types.h"
#include "../macros.h"
#include "../storage.h"

namespace forpy {
  const regint_t FORPY_FULL_INTERVAL(regint_t(-1, -1));
  // Forward declaration.
  class RegressionThresholdOptimizer;
  /**
   * \brief The interface for a regressor.
   *
   * Given a set of annotated samples, it calculates a model that explains the
   * dependency between the input variables and the response variables of the
   * sampels. This model is chosen as good as its parameters allow it to fit the
   * data. With this model, a prediction for the response variables on a single
   * sample can be calculated. Additionally, the probability distribution for a
   * specific output value given a single sample can be computed.
   *
   * Often, samples have to be incrementally added or removed from the sample
   * set. This can be done by initializing a IRegressor with the complete set of
   * samples and afterwards specify an interval as the index-range denoting
   * which sample are actually used for the regression model. Depending on the
   * type of regression calculator, this can lead to a major performance
   * increase. Only once the interval of samples to use is fixed (frozen), data
   * and annotations are released and only the model is stored.
   */
  class IRegressor {
   public:
    virtual ~IRegressor(){};

    /**
     * \brief Check if the regression calculator needs input data.
     *
     * \return Whether input data is needed or not.
     */
    virtual bool needs_input_data() const VIRTUAL(bool);

    /**
     * \brief Check if the regression calculator provides a constant prediciton
     * covariance.
     *
     * Returns, whether the regression calculator supports a constant prediction
     * covariance or each prediction covariance depends on the specific input.
     * Usually, this coincides wiht \ref needs_input_data.
     */
    virtual bool has_constant_prediction_covariance() const VIRTUAL(bool);

#define FORPY_REGRESSOR_INIT_DOC /** Initializes the regressor. */
    /**
     * \brief Initializes the regression calculator.
     *
     * In order to initialize a IRegressionCalulator, the complete set of
     * samples and their annotations has to be specified. Furthermore, an
     * initial interval \f$\left[a,b\right)$\f has to be provided. The samples
     * with indices in this interval are then used to calcualate a initial
     * regression model.
     *
     * The samples are potentially copied and transformed to homogeneous
     * coordinates, the annotation matrix is kept with a reference. If you know
     * what you're doing and can guarantee the existence of data and annotations
     * until you freeze the model, you can register your class as friend class,
     * *read the doc of the regressor you use and potentially homogenize your
     * data yourself* and use the `initialize_nocopy` method. See also
     * `needs_homogeneous_input_data`.
     *
     * \param sample_mat The complete set of samples. It needs to be valid only
     *    if \ref needs_input_data is true.
     * \param annotation_mat The complete set of annotations.
     * \param index_interval The initial index interval \f$\left[a,b\right)$\f
     *    on which the regression model is calculated.
     */
#define FORPY_REGRESSOR_INIT_RET(IT) void
#define FORPY_REGRESSOR_INIT_NAME initialize
#define FORPY_REGRESSOR_INIT_PARAMNAMES sample_mat, annotation_mat, index_interval
#define FORPY_REGRESSOR_INIT_PARAMTYPESNNAMES(IT)  \
    std::shared_ptr<const Mat<IT>> &sample_mat, \
    std::shared_ptr<const Mat<IT>> &annotation_mat, \
    const regint_t &index_interval
#define FORPY_REGRESSOR_INIT_PARAMTYPESNNAMESNDEF(IT)\
    std::shared_ptr<const Mat<IT>> &sample_mat,\
    std::shared_ptr<const Mat<IT>> &annotation_mat,\
    const regint_t &index_interval=FORPY_FULL_INTERVAL
#define FORPY_REGRESSOR_INIT_MOD
    FORPY_DECL(FORPY_REGRESSOR_INIT, ITR, , ;)

    /**
     * \brief Get the subset of samples used in the regression model.
     */
    virtual regint_t get_index_interval() const VIRTUAL(regint_t);

    /**
     * \brief Change the subset of samples used in the regression model.
     *
     * The provided interval specifies a new index range denoting the subset of
     * samples used in the regression model. The regression model is updated or
     * recomputed afterwards. The maximum interval is
     * \f$\left[0,n_samples\right)$\f
     * 
     * \return Whether the given interval is valid or not.
     */
    virtual bool set_index_interval(const regint_t & interval)
      VIRTUAL(bool);

    /**
     * \brief Check if a regression model is available.
     *
     * \return Whether a regression model is available based on the samples
     * specified by the current interval.
     */
    virtual bool has_solution() const VIRTUAL(bool);

#define FORPY_REGRESSOR_PREDICT_DOC 
    /**
     * \brief Predict the output for a given sample.
     *
     * The regression calculator predicts the output based on the given input.
     * The input is only used, if \ref needs_input_data is true. If necessary,
     * the data is homogenized before prediction.
     * 
     * \param input The input sample.
     * \param prediction_output The output for the prediction.
     */
#define FORPY_REGRESSOR_PREDICT_RET(IT) void
#define FORPY_REGRESSOR_PREDICT_NAME predict
#define FORPY_REGRESSOR_PREDICT_PARAMNAMES input, prediction_output
#define FORPY_REGRESSOR_PREDICT_PARAMTYPESNNAMES(IT) \
    const VecCRef<IT> &input,           \
    VecRef<IT> prediction_output
#define FORPY_REGRESSOR_PREDICT_PARAMTYPESNNAMESNDEF(IT) \
    FORPY_REGRESSOR_PREDICT_PARAMTYPESNNAMES(IT)
#define FORPY_REGRESSOR_PREDICT_MOD const
    FORPY_DECL(FORPY_REGRESSOR_PREDICT, ITR, virtual,;)

#define FORPY_REGRESSOR_PREDICT_NOCOPY_DOC 
      /**
       * \brief Predict the output for a given sample.
       *
       * The regression calculator predicts the output based on the given input.
       * The input is only used, if \ref needs_input_data is true. It is
       * assumed that the data is already homogeneous if required.
       * 
       * \param input The input sample.
       * \param prediction_output The output for the prediction.
       */
#define FORPY_REGRESSOR_PREDICT_NOCOPY_RET(IT) void
#define FORPY_REGRESSOR_PREDICT_NOCOPY_NAME predict_nocopy
#define FORPY_REGRESSOR_PREDICT_NOCOPY_PARAMNAMES input, prediction_output
#define FORPY_REGRESSOR_PREDICT_NOCOPY_PARAMTYPESNNAMES(IT)  \
        const VecCRef<IT> &input,                       \
        VecRef<IT> prediction_output
#define FORPY_REGRESSOR_PREDICT_NOCOPY_PARAMTYPESNNAMESNDEF(IT)  \
      FORPY_REGRESSOR_PREDICT_NOCOPY_PARAMTYPESNNAMES(IT)
#define FORPY_REGRESSOR_PREDICT_NOCOPY_MOD const
      FORPY_DECL(FORPY_REGRESSOR_PREDICT_NOCOPY, ITR, virtual,;)

#define FORPY_REGRESSOR_PREDICTCOVAR_DOC 
    /**
     * \brief Predicts the output and covariance for a given sample.
     *
     * The regression calculator predicts the output based on the given input.
     * The input is only used, if \ref needs_input_data is true. Additionally,
     * the covariance matrix of the predicted output specifying the uncertainty
     * of the prediction is calculated.
     * 
     * \param input The input sample.
     * \param prediction_output The output for the prediction.
     * \param covar_output The output for the covariance matrix.
     */
#define FORPY_REGRESSOR_PREDICTCOVAR_RET(IT) void
#define FORPY_REGRESSOR_PREDICTCOVAR_NAME predict_covar
#define FORPY_REGRESSOR_PREDICTCOVAR_PARAMNAMES input, prediction_output, covar_output
#define FORPY_REGRESSOR_PREDICTCOVAR_PARAMTYPESNNAMES(IT)  \
      const VecCRef<IT> &input,                               \
      VecRMRef<IT> prediction_output, \
      MatRef<IT> covar_output
#define FORPY_REGRESSOR_PREDICTCOVAR_PARAMTYPESNNAMESNDEF(IT)  \
      FORPY_REGRESSOR_PREDICTCOVAR_PARAMTYPESNNAMES(IT)
#define FORPY_REGRESSOR_PREDICTCOVAR_MOD const
      FORPY_DECL(FORPY_REGRESSOR_PREDICTCOVAR, ITR, virtual,;)

#define FORPY_REGRESSOR_PREDICTCOVAR_NOCOPY_DOC 
    /**
     * \brief Predicts the output and covariance for a given sample.
     *
     * The regression calculator predicts the output based on the given input.
     * The input is only used, if \ref needs_input_data is true. Additionally,
     * the covariance matrix of the predicted output specifying the uncertainty
     * of the prediction is calculated.
     * 
     * \param input The input sample.
     * \param prediction_output The output for the prediction.
     * \param covar_output The output for the covariance matrix.
     */
#define FORPY_REGRESSOR_PREDICTCOVAR_NOCOPY_RET(IT) void
#define FORPY_REGRESSOR_PREDICTCOVAR_NOCOPY_NAME predict_covar_nocopy
#define FORPY_REGRESSOR_PREDICTCOVAR_NOCOPY_PARAMNAMES input, prediction_output, covar_output
#define FORPY_REGRESSOR_PREDICTCOVAR_NOCOPY_PARAMTYPESNNAMES(IT)  \
      const VecCRef<IT> &input,                               \
      VecRef<IT> prediction_output, \
      MatRef<IT> covar_output
#define FORPY_REGRESSOR_PREDICTCOVAR_NOCOPY_PARAMTYPESNNAMESNDEF(IT)  \
      FORPY_REGRESSOR_PREDICTCOVAR_NOCOPY_PARAMTYPESNNAMES(IT)
#define FORPY_REGRESSOR_PREDICTCOVAR_NOCOPY_MOD const
      FORPY_DECL(FORPY_REGRESSOR_PREDICTCOVAR_NOCOPY, ITR, virtual,;)

#define FORPY_REGRESSOR_GETCONSTANTPREDCOV_DOC 
    /**
     * \brief Get the constant prediction covariance
     *
     * Returns the constant covariance matrix of the predictions. This is not
     * supported by all regression calculators. See \ref
     * has_constant_prediction_covariance
     *
     * \param covar_output The output for the covariance matrix.
     */
#define FORPY_REGRESSOR_GETCONSTANTPREDCOV_RET(IT) void
#define FORPY_REGRESSOR_GETCONSTANTPREDCOV_NAME get_constant_prediction_covariance
#define FORPY_REGRESSOR_GETCONSTANTPREDCOV_PARAMNAMES covar_output
#define FORPY_REGRESSOR_GETCONSTANTPREDCOV_PARAMTYPESNNAMES(IT) \
      MatRef<IT> covar_output
#define FORPY_REGRESSOR_GETCONSTANTPREDCOV_PARAMTYPESNNAMESNDEF(IT) \
      FORPY_REGRESSOR_GETCONSTANTPREDCOV_PARAMTYPESNNAMES(IT)
#define FORPY_REGRESSOR_GETCONSTANTPREDCOV_MOD const
      FORPY_DECL(FORPY_REGRESSOR_GETCONSTANTPREDCOV, ITR, virtual,;)
    
    /**
     * \brief Freezes the currently set index interval.
     *
     * The currently set index interval is locked. It can not be changed
     * afterwards. The solution based on this interval is preserved. The main
     * effect of this method is to release all unnecessary data. Call this
     * method before serializing!
     */
    virtual void freeze() VIRTUAL(void);

    /**
     * \brief Whether this regressor has a frozen interval.
     */
    virtual bool get_frozen() const VIRTUAL(bool);

    /**
     * \brief Get the dimensionality of the samples used in the regression
     * calculator.
     */
    virtual size_t get_input_dimension() const VIRTUAL(size_t);

    /**
     * \brief Get the dimensionality of the annotations used in the regression
     * calculator.
     */
    virtual size_t get_annotation_dimension() const VIRTUAL(size_t);

    /**
     * \brief Get the number of samples the model was built on.
     */
    virtual size_t get_n_samples() const VIRTUAL(size_t);

    virtual bool operator==(const IRegressor &rhs) const VIRTUAL(bool);

    virtual std::unique_ptr<IRegressor> empty_duplicate() const
      VIRTUAL(std::unique_ptr<IRegressor>);

#define FORPY_REGRESSOR_INIT_NOCOPY_DOC 
#define FORPY_REGRESSOR_INIT_NOCOPY_RET(IT) void
#define FORPY_REGRESSOR_INIT_NOCOPY_NAME initialize_nocopy
#define FORPY_REGRESSOR_INIT_NOCOPY_PARAMNAMES sample_mat, annotation_mat, index_interval
#define FORPY_REGRESSOR_INIT_NOCOPY_PARAMTYPESNNAMES(IT)  \
    const MatCRef<IT> &sample_mat,                        \
    const MatCRef<IT> &annotation_mat,                  \
    const regint_t &index_interval
#define FORPY_REGRESSOR_INIT_NOCOPY_PARAMTYPESNNAMESNDEF(IT)  \
    const MatCRef<IT> &sample_mat,                            \
    const MatCRef<IT> &annotation_mat,                      \
    const regint_t &index_interval=FORPY_FULL_INTERVAL
#define FORPY_REGRESSOR_INIT_NOCOPY_MOD
    FORPY_DECL(FORPY_REGRESSOR_INIT_NOCOPY, ITR, virtual,;)

    virtual float get_residual_error() const VIRTUAL(float);

    virtual size_t get_kernel_dimension() const VIRTUAL(size_t);

    virtual std::string get_name() const VIRTUAL(std::string);

   protected:
    IRegressor(){};
    RegStore<Mat> annotation_mat_data;
    RegStore<Mat> sample_mat_data;
    RegData<MatCRef> annotation_mat;
    RegData<MatCRef> sample_mat;

    /**
     * \brief Check if the regression calculator needs input data.
     *
     * \return Whether input data is needed or not.
     */
    virtual bool needs_homogeneous_input_data() const VIRTUAL(bool);

   private:
    FORPY_DECL_IMPL(FORPY_REGRESSOR_INIT, ITR, );
    FORPY_DECL_IMPL(FORPY_REGRESSOR_PREDICT, ITR, );
    FORPY_DECL_IMPL(FORPY_REGRESSOR_PREDICTCOVAR, ITR, );

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive &, const uint &) {};
  };
}; // namespace forpy
#endif // FORPY_REGRESSION_IREGRESSOR_H_
