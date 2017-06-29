/* Author: Moritz Einfalt, Christoph Lassner. */
#pragma once
#ifndef FORPY_UTIL_REGRESSION_LINEARREGRESSOR_H_
#define FORPY_UTIL_REGRESSION_LINEARREGRESSOR_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/access.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>

#include "../../global.h"
#include "../../types.h"
#include "../storage.h"
#include "../serialization/eigen.h"
#include "./iregressor.h"

namespace forpy {
  /**
   * \brief Calculator for linear regression.
   *
   * This regression calculator uses a linear combination of the input
   * dimensions to predict the output value. Therefore it does not provide a
   * constant prediction or a constant prediction covariance matrix. If there
   * are multiple output values to be predicted, each output is produced using
   * its own linear model.
   */
  class LinearRegressor : public IRegressor {
   public:
   /**
    * \brief Constructor for a LinearRegressionCalculator
    *
    * Costructs a LinearRegressionCalculator. If numberical stability is not
    * forced, the linear models in low dimensional cases are computed using a
    * closed form. This is faster but less accurate. Otherwise, always matrix
    * decomposition is used which provides more accurate and stable solutions.
    * In order to prevent numerical issues, a threshold can be specified to
    * denote the smallest number that is distinct to zero. Using the default
    * value -1, this threshold is determined automatically based on the data
    * samples.
    *
    * \returns A new LinearRegressionCalculator.
    * \param force_numerical_stability bool
    *   Whether to enforce numerical stability or allow instable solutions.
    *   Default: true.
    * \param numerical_zero_threshold DT >=0||-1
    *   Everything below this threshold is treated as zero. If set to -1.f, 
    *   use the value proposed by Eigen. Default: -1.f
    */
   LinearRegressor(const bool &force_numerical_stability=true,
                   const double &numerical_zero_threshold=-1);

    /** Returns true. */
    bool needs_input_data() const;

    /** Returns false. */
    bool has_constant_prediction_covariance() const;

    regint_t get_index_interval() const;

    /** Changes the interval if allowed. Returns success. */
    bool set_index_interval(const regint_t & interval);

    /** Whether the regression is well-defined for the data. */
    bool has_solution() const;

    FORPY_DECL(FORPY_REGRESSOR_PREDICT_NOCOPY, ITR, ,;)

    FORPY_DECL(FORPY_REGRESSOR_PREDICTCOVAR_NOCOPY, ITR, ,;)

    FORPY_DECL(FORPY_REGRESSOR_GETCONSTANTPREDCOV, ITR, ,;)

    /** Does not allow an interval change anymore. */
    void freeze();

    /**
     * \brief Whether this regressor has a frozen interval.
     */
    bool get_frozen() const;

    /** Returns the input dimension used. */
    size_t get_input_dimension() const;

   /** Returns the annotation dimensiton used. */
   size_t get_annotation_dimension() const;

   /**
    * \brief Get the number of samples the model was built on.
    */
   size_t get_n_samples() const;

   /** Returns the value as specified in the constructor. */
   bool forces_numerical_stability() const;

   /** Returns the threshold as specified in the constructor. */
   double get_numerical_zero_threshold() const;

   bool operator==(const IRegressor &rhs) const;

   std::unique_ptr<IRegressor> empty_duplicate() const;

   FORPY_DECL(FORPY_REGRESSOR_INIT_NOCOPY, ITR, , ;)

   protected:
    using IRegressor::sample_mat;
    using IRegressor::sample_mat_data;
    using IRegressor::annotation_mat;
    using IRegressor::annotation_mat_data;

    /** Returns false. */
    bool needs_homogeneous_input_data() const;

    float get_residual_error() const;

    size_t get_kernel_dimension() const;

    std::string get_name() const;

   private:
    FORPY_DECL_IMPL(FORPY_REGRESSOR_INIT, ITR, );
    FORPY_DECL_IMPL(FORPY_REGRESSOR_PREDICT_NOCOPY, ITR, );
    FORPY_DECL_IMPL(FORPY_REGRESSOR_PREDICTCOVAR_NOCOPY, ITR, );
    FORPY_DECL_IMPL(FORPY_REGRESSOR_GETCONSTANTPREDCOV, ITR, );
    FORPY_DECL_IMPL(FORPY_REGRESSOR_INIT_NOCOPY, ITR, );

    bool check_interval_valid(const regint_t & interval);

    template<typename IT>
    bool calc_solution(IT) {
      const auto &sample_mat = this->sample_mat.get_unchecked<MatCRef<IT>>();
      const auto &annotation_mat = this->annotation_mat.get_unchecked<MatCRef<IT>>();
      auto &error_vars = this->error_vars.get_unchecked<Vec<IT>>();
      auto &solution = this->solution.get_unchecked<Mat<IT>>();
      auto &param_covar_mat_template = this->param_covar_mat_template.get_unchecked<Mat<IT>>();
      IT numerical_zero_threshold = static_cast<IT>(this->numerical_zero_threshold);
      // Notation:
      // sample_input_mat: X
      // sample_annotation_mat : Y
      // w_mat: W = X' * X
      // w_inverse_mat: W^(-1)
      auto current_sample_mat = sample_mat.block(
          current_interval.first, 0,
          current_interval.second - current_interval.first,
          input_dim);
      auto current_annot_mat = annotation_mat.block(
          current_interval.first, 0,
          current_interval.second-current_interval.first, annot_dim);
      Mat<IT> w_inverse_mat(input_dim, input_dim);
      // Regression models with less than five parameters can be computed fast
      // without matrix decomposition.
      if ((!force_numerical_stability) && input_dim < 5) {
        IT numeric_threshold = (numerical_zero_threshold <= static_cast<IT>(0)) ?
          (Eigen::NumTraits<IT>::dummy_precision()) :
          (numerical_zero_threshold);
        IT determinant;
        bool invertible = false;
        // For small matrices, the inverse calculation in Eigen is very fast for fixed-size matrices only.
        if (input_dim == 1) {
          // Can't occur, because of homogeneous coordinates.
          Eigen::Matrix<IT,1,1,Eigen::RowMajor> w_mat;
          w_mat.noalias() = current_sample_mat.transpose() * current_sample_mat;
          w_mat.computeInverseAndDetWithCheck(w_inverse_mat,
                                              determinant,
                                              invertible,
                                              numeric_threshold);
        } else if (input_dim == 2) {
          Eigen::Matrix<IT,2,2,Eigen::RowMajor> w_mat;
          w_mat.noalias() = current_sample_mat.transpose() * current_sample_mat;
          w_mat.computeInverseAndDetWithCheck(w_inverse_mat,
                                              determinant,
                                              invertible,
                                              numeric_threshold);
        } else if (input_dim == 3) {
          Eigen::Matrix<IT,3,3,Eigen::RowMajor> w_mat;
          w_mat.noalias() = current_sample_mat.transpose() * current_sample_mat;
          w_mat.computeInverseAndDetWithCheck(w_inverse_mat,
                                              determinant,
                                              invertible,
                                              numeric_threshold);
        } else if (input_dim == 4) {
          Eigen::Matrix<IT,4,4,Eigen::RowMajor> w_mat;
          w_mat.noalias() = current_sample_mat.transpose() * current_sample_mat;
          w_mat.computeInverseAndDetWithCheck(w_inverse_mat,
                                              determinant,
                                              invertible,
                                              numeric_threshold);
        }
        if (! invertible) {
          return false;
        }
        // Calculate the least squares solution = W^(-1) * X' * Y
        solution.noalias() = w_inverse_mat * (current_sample_mat.transpose() *
                                              current_annot_mat);
      } else {
        // This is the case, where the regression model is calculated for more
        // than four input dimensions or a numerical stable solution is required
        // In order to check if a unique solution to the regression problem
        // exists, a rank-revealing QR-decomposition is used.
        // Calculate the decomposition X*P = Q*R, where
        // P is a permutation matrix and
        // R is upper triangular.
        Eigen::ColPivHouseholderQR<Mat<IT>> decomposer;
        if (numerical_zero_threshold <= static_cast<IT>(0)) {
          decomposer.setThreshold(Eigen::Default);
        } else {
          decomposer.setThreshold(numerical_zero_threshold);
        }
        decomposer.compute(current_sample_mat);
        /*
          The QR decomposition method will determine another rank than the
          FulPivLU. Doesn't matter, we did our job.
        size_t rank = decomposer.rank();
        if (rank < input_dim) {
          Eigen::FullPivLU<Mat<IT>> decomposer(sample_mat);
          if (numerical_zero_threshold <= static_cast<IT>(0)) {
            decomposer.setThreshold(Eigen::Default);
          } else {
            decomposer.setThreshold(numerical_zero_threshold);
          }
          throw Forpy_Exception("Internal error! Rank for LR lower than inputs: "
                                "Rank here: " + std::to_string(rank) +
                                " determined before: " + std::to_string(input_dim) +
                                " LU rank here: " + std::to_string(decomposer.rank()));
                                }*/
        // Get the model params using the built-in solver.
        solution = decomposer.solve(current_annot_mat);
        // In order to compute W^(-1) = (X' * X)^(-1), the equal expression P *
        // R^(-1) * R^(-1)' * P' is calculated. Since R is upper triangular,
        // R^(-1) is too and can directly be calculated.
        auto r_mat = decomposer.matrixR();
        Mat<IT> r_inverse_mat = Mat<IT>::Zero(input_dim, input_dim);
        for (Eigen::Index j = static_cast<Eigen::Index>(input_dim - 1);
             j >= 0 ; --j) {
          r_inverse_mat(j,j) = static_cast<IT>(1) / r_mat(j,j);
          for (Eigen::Index i = j-1; i >= 0; --i) {
            for (Eigen::Index k = i+1; k <= j; k++) {
              r_inverse_mat(i,j) -= r_mat(i,k) * r_inverse_mat(k,j) / r_mat(i,i);
            }
          }
        }
        w_inverse_mat.noalias() = decomposer.colsPermutation() *
          (r_inverse_mat * r_inverse_mat.transpose()) *
          decomposer.colsPermutation().transpose();
      }
      param_covar_mat_template = w_inverse_mat.eval();
      // Now, estimate the error variance (sigma^2) through the sum of squared
      // residuals for each annotation dimension.
      Mat<IT> predmat = current_sample_mat * solution;
      for (Eigen::Index i = 0;
           i < (current_interval.second - current_interval.first); ++i) {
        for (size_t j = 0; j < annot_dim; ++j) {
          if (i == 0)
            error_vars(j) = static_cast<IT>(0.f);
          IT residual = predmat(i,j) - current_annot_mat(i,j);
          error_vars(j) += (residual * residual) /
            static_cast<IT>(n_samples - input_dim);
          // Since input_dim is in homogeneous coordinates, this is
          // - degrees_of_freedom - 1.
        }
      }
      return true;
    };

    friend class cereal::access;
    template<class Archive>
    void save(Archive &ar, const uint &) const {
      if (! get_frozen()) {
        throw Forpy_Exception("This regressor has not been frozen before "
                              "serialization. This is unsupported.");
      }
      ar(cereal::make_nvp("base",
                          cereal::base_class<IRegressor>(this)),
         CEREAL_NVP(force_numerical_stability),
         CEREAL_NVP(numerical_zero_threshold),
         CEREAL_NVP(initialized));
      if (initialized) {
        ar(CEREAL_NVP(double_mode),
           CEREAL_NVP(rank_deficient),
           CEREAL_NVP(orig_input_dim),
           CEREAL_NVP(proj),
           CEREAL_NVP(input_dim),
           CEREAL_NVP(annot_dim),
           CEREAL_NVP(n_samples),
           CEREAL_NVP(solution_available));
        if (solution_available) {
          ar(CEREAL_NVP(solution),
             CEREAL_NVP(param_covar_mat_template),
             CEREAL_NVP(error_vars));
        }
      }
    }
    template<class Archive>
    void load(Archive &ar, const uint &) {
      ar(cereal::make_nvp("base",
                          cereal::base_class<IRegressor>(this)),
         CEREAL_NVP(force_numerical_stability),
         CEREAL_NVP(numerical_zero_threshold),
         CEREAL_NVP(initialized));
      if (initialized) {
        ar(CEREAL_NVP(double_mode),
           CEREAL_NVP(rank_deficient),
           CEREAL_NVP(orig_input_dim),
           CEREAL_NVP(proj),
           CEREAL_NVP(input_dim),
           CEREAL_NVP(annot_dim),
           CEREAL_NVP(n_samples),
           CEREAL_NVP(solution_available));
        if (solution_available) {
          ar(CEREAL_NVP(solution),
             CEREAL_NVP(param_covar_mat_template),
             CEREAL_NVP(error_vars));
        }
      }
      current_interval = regint_t(-1, -1);
      interval_frozen = true;
    };

    bool force_numerical_stability;
    double numerical_zero_threshold;
    bool initialized;
    bool double_mode;
    bool rank_deficient;
    size_t orig_input_dim;
    size_t input_dim;
    size_t annot_dim;
    size_t n_samples;
    regint_t current_interval;
    std::vector<Eigen::Index> proj;
    RegData<Mat> solution;
    RegData<Mat> param_covar_mat_template;
    RegData<Vec> error_vars;
    bool solution_available;
    bool interval_frozen;
  };
}; // namespace forpy

namespace cereal
{
  template <class Archive> 
    struct specialize<
    Archive,
    forpy::LinearRegressor,
    cereal::specialization::member_load_save> {};
}
CEREAL_REGISTER_TYPE(forpy::LinearRegressor);
#endif // FORPY_UTIL_REGRESSION_LINEARREGRESSOR_H_
