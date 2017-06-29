#include <forpy/util/regression/constantregressor.h>

namespace forpy {

  ConstantRegressor::ConstantRegressor()
    : input_dim(0),
      annot_dim(0),
      n_samples(0),
      current_interval(std::make_pair(-1,-1)),
      initialized(false),
      use_double(false),
      solution(),
      error_vars(),
      solution_available(false),
      interval_frozen(false) {};

  bool ConstantRegressor::needs_input_data() const { return false; };
  
  bool ConstantRegressor::has_constant_prediction_covariance() const { return true; };

  regint_t ConstantRegressor::get_index_interval() const { return current_interval; };

  bool ConstantRegressor::set_index_interval(const regint_t &interval) {
    if (! check_interval_valid(interval)) {
      throw Forpy_Exception("Invalid index interval!");
    };
    if (interval_frozen) {
      throw Forpy_Exception("This regressor has been frozen already!");
    }
    if (! initialized) {
      throw Forpy_Exception("This regressor has not been initialized yet!");
    }
    if (interval != current_interval) {
      if ((interval.second - interval.first) < 1) {
        this->solution_available = false;
        current_interval = interval;
        return false;
      } else {
        while(current_interval != interval) {
          if (solution_available) {
            // right interval increase
            if (current_interval.first == interval.first &&
                interval.second > current_interval.second) {
              if (use_double) {
                solution_available = increment_right_interval_boundary<double>();
              } else {
                solution_available = increment_right_interval_boundary<float>();
              }
              // left interval decrease
            } else if (current_interval.second == interval.second &&
                       current_interval.first < interval.first) {
              if (use_double) {
                solution_available = increment_left_interval_boundary<double>();
              } else {
                solution_available = increment_left_interval_boundary<float>();
              }
              // recalculation
            } else {
              current_interval = interval;
              if (use_double) {
                solution_available = calc_solution<double>();
              } else {
                solution_available = calc_solution<float>();
              }
            }
          } else {
            current_interval = interval;
            if (use_double) {
              solution_available = calc_solution<double>();
            } else {
              solution_available = calc_solution<float>();
            }
          }
        }
        return true;
      }
    } else {
      return (interval.second - interval.first) > 0;
    }
  };

  bool ConstantRegressor::has_solution() const { return solution_available; };

  FORPY_IMPL(FORPY_REGRESSOR_PREDICT_NOCOPY, ITR, ConstantRegressor) {
    if (static_cast<size_t>(prediction_output.rows()) !=
        get_annotation_dimension()) {
      throw Forpy_Exception("prediction_output must have " +
                            std::to_string(get_annotation_dimension()) +
                            " rows ( " +
                            "has " + std::to_string(prediction_output.rows()) +
                            ")!");
    }
    if (solution_available) {
      const auto &solution = this->solution.get_unchecked<Vec<IT>>();
      prediction_output = solution;
    } else {
      throw Forpy_Exception("No solution available! Check this before "
                            "predicting by using `get_solution_available`!");
    }
  };

  FORPY_IMPL(FORPY_REGRESSOR_PREDICTCOVAR_NOCOPY, ITR, ConstantRegressor) {
    if (solution_available) {
      predict(input, prediction_output);
      get_constant_prediction_covariance(covar_output);
    } else {
      throw Forpy_Exception("No solution available! Check this before "
                            "predicting by using `get_solution_available`!");
    }
  };

  FORPY_IMPL(FORPY_REGRESSOR_GETCONSTANTPREDCOV, ITR, ConstantRegressor) {
    const auto &solution = this->solution.get_unchecked<Vec<IT>>();
    const auto &error_vars = this->error_vars.get_unchecked<Vec<IT>>();
    if (static_cast<size_t>(covar_output.rows()) != annot_dim) {
      throw Forpy_Exception("covar_output must have " +
                            std::to_string(annot_dim) + " rows!");
    }
    if (static_cast<size_t>(covar_output.cols()) != annot_dim) {
      throw Forpy_Exception("covar_output must have " +
                            std::to_string(solution.cols()) + " cols!");
    }
    if (solution_available) {
      covar_output.fill(static_cast<IT>(0.f));
      for (size_t i=0; i<annot_dim; i++) {
        covar_output(i,i) = error_vars(i);
      }
    } else {
      throw Forpy_Exception("No solution available! Check this before "
                            "predicting by using `get_solution_available`!");
    }
  };

  void ConstantRegressor::freeze() {
    interval_frozen = true;
    current_interval = regint_t(-1, -1);
    this->annotation_mat = Empty();
    this->sample_mat = Empty();
    mu::apply_visitor(VReset(), this->annotation_mat_data);
    mu::apply_visitor(VReset(), this->sample_mat_data);
  };

  bool ConstantRegressor::get_frozen() const { return interval_frozen; };

  size_t ConstantRegressor::get_input_dimension() const {
    if (! initialized) {
      throw Forpy_Exception("Regressor not initialized!");
    }
    return input_dim;
  };

  size_t ConstantRegressor::get_annotation_dimension() const {
    if (! initialized) {
      throw Forpy_Exception("Regressor not initialized!");
    }
    return annot_dim;
  };

  size_t ConstantRegressor::get_n_samples() const { return n_samples; };

  FORPY_IMPL(FORPY_REGRESSOR_INIT_NOCOPY, ITR, ConstantRegressor) {
    this->annotation_mat = MatCRef<IT>(annotation_mat);
    initialized = true;
    if (typeid(IT) == typeid(float)) {
      use_double = false;
    } else {
      use_double = true;
    }
    this->input_dim = sample_mat.cols();
    this->annot_dim = annotation_mat.cols();
    this->n_samples = annotation_mat.rows();
    interval_frozen = false;
    regint_t real_interval;
    if (index_interval.first == -1 && index_interval.second == -1) {
      real_interval = std::make_pair(0, n_samples);
    } else {
      real_interval = index_interval;
    }
    solution.set<Vec<IT>>(annot_dim);
    error_vars.set<Vec<IT>>(annot_dim);
    current_interval = regint_t(-1, -1);
    set_index_interval(real_interval);
  };

  bool ConstantRegressor::needs_homogeneous_input_data() const { return false; };

  bool ConstantRegressor::check_interval_valid(const regint_t & interval) {
    return (interval.second >= interval.first &&
            interval.first >= 0 &&
            static_cast<size_t>(interval.second) <= n_samples);
  };

  bool ConstantRegressor::operator==(const IRegressor &rhs) const {
    const auto *rhs_c = dynamic_cast<ConstantRegressor const *>(&rhs);
    if (rhs_c == nullptr) {
      return false;
    } else {
      bool eq_ind = input_dim == rhs_c -> input_dim;
      bool eq_init = initialized == rhs_c -> initialized;
      bool eq_mode = use_double == rhs_c -> use_double;
      bool eq_adim = annot_dim == rhs_c -> annot_dim;
      bool eq_amat = sample_mat == rhs_c -> sample_mat;
      bool eq_nsamples = n_samples == rhs_c -> n_samples;
      bool eq_int = current_interval == rhs_c -> current_interval;
      bool eq_sol = solution == rhs_c -> solution;
      bool eq_ev = error_vars == rhs_c -> error_vars;
      bool eq_av = solution_available == rhs_c -> solution_available;
      bool eq_frz = interval_frozen == rhs_c -> interval_frozen;
      if (! interval_frozen || ! rhs_c -> interval_frozen) {
        if (! initialized) {
          return eq_init;
        } else {
          return eq_adim && eq_nsamples && eq_int && eq_sol && eq_ev &&
            eq_av && eq_frz && eq_amat && eq_mode && eq_ind;
        }
      } else {
        if (! initialized) {
          return eq_init;
        } else {
          return eq_adim && eq_nsamples && eq_sol && eq_ev && eq_av && eq_frz &&
            eq_mode && eq_ind;
        }
      }
    }
  };

  std::unique_ptr<IRegressor> ConstantRegressor::empty_duplicate() const {
    return std::unique_ptr<IRegressor>(new ConstantRegressor());
  }

  float ConstantRegressor::get_residual_error() const {
    if (! initialized) {
      throw Forpy_Exception("Can't get the error of an uninitialized regressor!");
    }
    if (use_double) {
      return error_vars.get_unchecked<Vec<double>>().mean();
    } else {
      return error_vars.get_unchecked<Vec<float>>().mean();
    }
  };

  size_t ConstantRegressor::get_kernel_dimension() const {
    if (! initialized) {
      throw Forpy_Exception("Regressor not initialized!");
    }
    return 0;
  }

  std::string ConstantRegressor::get_name() const {
    return "ConstantRegressor";
  }
} // namespace forpy
