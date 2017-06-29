/* Author: Moritz Einfalt, Christoph Lassner. */
#pragma once
#ifndef FORPY_UTIL_REGRESSION_CONSTANTREGRESSOR_H_
#define FORPY_UTIL_REGRESSION_CONSTANTREGRESSOR_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/access.hpp>

#include "../../global.h"
#include "../../types.h"
#include "../macros.h"
#include "../serialization/eigen.h"
#include "../storage.h"
#include "./iregressor.h"

namespace forpy {
  /**
   * \brief Calculator for constant regression.
   *
   * This regression calculator uses a constant value to predict the output
   * value. Therefore, it provides a constant prediction and a constant
   * prediction covariance matrix.
   */
  class ConstantRegressor : public IRegressor {
   public:
    ConstantRegressor();

    /** Returns false. */
    bool needs_input_data() const;

    /** Returns true. */
    bool has_constant_prediction_covariance() const;

    FORPY_DECL(FORPY_REGRESSOR_INIT, ITR, , ;)

    regint_t get_index_interval() const;

    /** Adjusts the data interval and updates whether a solution is
     * available.
     */
    bool set_index_interval(const regint_t &interval);

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

    /** Always returns 0. */
    size_t get_input_dimension() const;

    /** Returns the annotation dimension. */
    size_t get_annotation_dimension() const;

    /**
     * \brief Get the number of samples the model was built on.
     */
    size_t get_n_samples() const;

    bool operator==(const IRegressor &rhs) const;

    std::unique_ptr<IRegressor> empty_duplicate() const;

    FORPY_DECL(FORPY_REGRESSOR_INIT_NOCOPY, ITR, , ;)

   protected:
    using IRegressor::annotation_mat_data;
    using IRegressor::sample_mat_data;

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
    bool calc_solution() {
      auto &annotation_mat = this->annotation_mat.get_unchecked<MatCRef<IT>>();
      auto &error_vars = this->error_vars.get_unchecked<Vec<IT>>();
      auto &solution = this->solution.get_unchecked<Vec<IT>>();
      ptrdiff_t current_n_sample = current_interval.second -
        current_interval.first;
      if (current_n_sample < 1)
        return false;
      solution = (annotation_mat.block(current_interval.first,
                                       0,
                                       current_interval.second - current_interval.first,
                                       annotation_mat.cols()).colwise()).mean();
      // calculate the error variances
      if (current_n_sample == 1) {
        error_vars.fill(static_cast<IT>(0.f));
      } else {
        for (ptrdiff_t i=current_interval.first;
             i<current_interval.second; ++i) {
          for (size_t j=0; j<annot_dim; ++j) {
            if (i==current_interval.first)
              error_vars(j) = static_cast<IT>(0.f);
            error_vars(j) = error_vars(j) +
              ((annotation_mat)(i,j) - solution(j)) *
              ((annotation_mat)(i,j) - solution(j)) /
              static_cast<IT>(current_n_sample);
          }
        }
      }
      return true;
    };

    template<typename IT>
    bool increment_right_interval_boundary () {
      auto &annotation_mat = this->annotation_mat.get_unchecked<MatCRef<IT>>();
      auto &error_vars = this->error_vars.get_unchecked<Vec<IT>>();
      auto &solution = this->solution.get_unchecked<Vec<IT>>();
      int added_index = current_interval.second;
      int old_n_samples = current_interval.second - current_interval.first;
      int new_n_samples = old_n_samples + 1;
      current_interval.second++;
      if (static_cast<size_t>(current_interval.second) > n_samples ||
          new_n_samples < 1)
          return false;
      if (old_n_samples < 1)
        return calc_solution<IT>();
      // Update the mean annotation and error variances.
      for (size_t j=0; j<annot_dim; ++j) {
        IT mean_new = solution(j) +
          (((annotation_mat)(added_index, j) - solution(j)) /
           (static_cast<IT>(new_n_samples)));
        error_vars(j) *= ((static_cast<IT>(old_n_samples)) /
                          (static_cast<IT>(old_n_samples + 1)));
        error_vars(j) += ((annotation_mat)(added_index, j) - solution(j)) *
                         ((annotation_mat)(added_index, j) - mean_new) /
                         (static_cast<IT>(old_n_samples + 1));
        solution(j) = mean_new;
      }
      return true;
    };

    template<typename IT>
    bool increment_left_interval_boundary () {
      auto &annotation_mat = this->annotation_mat.get_unchecked<MatCRef<IT>>();
      auto &error_vars = this->error_vars.get_unchecked<Vec<IT>>();
      auto &solution = this->solution.get_unchecked<Vec<IT>>();
      int removed_index = current_interval.first;
      int old_n_samples = current_interval.second - current_interval.first;
      int new_n_samples = old_n_samples - 1;
      current_interval.first ++;
      if (static_cast<size_t>(current_interval.first) >= n_samples ||
          new_n_samples < 1)
          return false;
      if (old_n_samples < 3)
        return calc_solution<IT>();
      // Update the mean annotation and error variances.
      for (size_t j=0; j<annot_dim; j++) {
        IT mean_new  = solution(j) +
          ((solution(j) - (annotation_mat)(removed_index,j)) /
           static_cast<IT>(new_n_samples));
        error_vars(j) *= ((static_cast<IT>(old_n_samples)) /
                          (static_cast<IT>(new_n_samples)));
        error_vars(j) -= ((annotation_mat)(removed_index, j) - solution(j)) *
                         ((annotation_mat)(removed_index, j) - mean_new) /
                         (static_cast<IT>(new_n_samples));
        solution(j) = mean_new;
        // Check for numerical error with var < 0.
        error_vars(j) = std::max(error_vars(j), static_cast<IT>(0));
      }
      return true;
    };

    friend class cereal::access;
    template<class Archive>
    void save(Archive & ar, const uint &) const {
      if (! get_frozen()) {
         throw Forpy_Exception("This regressor has not been frozen before "
                               "serialization. This is unsupported.");
       }
       ar(cereal::make_nvp("base",
                           cereal::base_class<IRegressor>(this)),
          CEREAL_NVP(initialized));
       if (initialized) {
         ar(CEREAL_NVP(input_dim),
            CEREAL_NVP(annot_dim),
            CEREAL_NVP(n_samples),
            CEREAL_NVP(use_double),
            CEREAL_NVP(solution_available));
         if (solution_available) {
           ar(CEREAL_NVP(solution),
              CEREAL_NVP(error_vars));
         }
       }
    }
    template<class Archive>
    void load(Archive & ar, const uint &) {
      ar(cereal::make_nvp("base",
                          cereal::base_class<IRegressor>(this)),
         CEREAL_NVP(initialized));
      if (initialized) {
        ar(CEREAL_NVP(input_dim),
           CEREAL_NVP(annot_dim),
           CEREAL_NVP(n_samples),
           CEREAL_NVP(use_double),
           CEREAL_NVP(solution_available));
        if (solution_available) {
          ar(CEREAL_NVP(solution),
             CEREAL_NVP(error_vars));
        }
      }
      current_interval = regint_t(-1, -1);
      interval_frozen = true;
    }

    /// Residual visitor. Calculates the averaged squared norm.
    struct ResVis {
      float result;
      const RegData<Vec> *solution;

      explicit inline ResVis(RegData<Vec> const * solution) :
        result(0.f), solution(solution) {};

      template <typename T>
      typename std::enable_if<std::is_same<T, Empty>::value, void>::type
      operator()(const T &) {
        throw Forpy_Exception("Tried to find residuals of an empty regressor.");
      }

      template <typename T>
      typename std::enable_if<!std::is_same<T, Empty>::value, void>::type
      operator()(const T &an) {
        typedef typename get_core<decltype(*(an.data()))>::type IT;
        const auto &sol = solution->get_unchecked<Vec<IT>>();
        result = (an.rowwise() -
                  sol.transpose()).rowwise().sum().squaredNorm() /
          static_cast<float>(an.rows());
      }
    };

    size_t input_dim;
    size_t annot_dim;
    size_t n_samples;
    regint_t current_interval;
    bool initialized;
    bool use_double;
    RegData<Vec> solution;
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
    forpy::ConstantRegressor,
    cereal::specialization::member_load_save> {};
}
CEREAL_REGISTER_TYPE(forpy::ConstantRegressor);
#endif // FORPY_REGRESSION_CONSTANTREGRESSOR_H_
