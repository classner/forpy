#include <forpy/util/regression/linearregressor.h>

namespace forpy {

  LinearRegressor::LinearRegressor(
     const bool &force_numerical_stability,
     const double &numerical_zero_threshold)
      : force_numerical_stability(force_numerical_stability),
        numerical_zero_threshold(numerical_zero_threshold),
        initialized(false),
        double_mode(false),
        rank_deficient(false),
        orig_input_dim(0),
        input_dim(0),
        annot_dim(0),
        n_samples(0),
        current_interval(std::make_pair(-1,-1)),
        proj(0),
        solution(),
        param_covar_mat_template(),
        error_vars(),
        solution_available(false),
        interval_frozen(false) {
      if (numerical_zero_threshold != -1. &&
          numerical_zero_threshold < 0.) {
        throw Forpy_Exception("Invalid numerical zero threhsold.");
      }
    };

  bool LinearRegressor::needs_input_data() const { return true; };

  bool LinearRegressor::has_constant_prediction_covariance() const {
    return false; };

  regint_t LinearRegressor::get_index_interval() const {
    return current_interval; };

  bool LinearRegressor::set_index_interval(const regint_t & interval) {
    if (! initialized) {
      throw Forpy_Exception("This regressor has not been initialized!");
    }
    if (! check_interval_valid(interval)) {
      throw Forpy_Exception("Invalid index interval!");
    };
    if (interval_frozen) {
      throw Forpy_Exception("This regressor has been frozen already!");
    }
    if (interval != current_interval) {
      current_interval = interval;
      if (static_cast<size_t>(current_interval.second - current_interval.first) <
          (input_dim + 1)) {
        this->solution_available = false;
        return false;
      } else {
        if (double_mode) {
          solution_available = calc_solution(double());
        } else {
          solution_available = calc_solution(float());
        }
        return true;
      }
    } else {
      return (! (static_cast<size_t>(current_interval.second - current_interval.first) <
                 (input_dim + 1)));
    }
  };

  bool LinearRegressor::has_solution() const { return solution_available; };

  FORPY_IMPL(FORPY_REGRESSOR_PREDICT_NOCOPY, ITR, LinearRegressor) {
    const auto &solution = this->solution.get_unchecked<Mat<IT>>();
    if (static_cast<size_t>(input.rows()) != orig_input_dim) {
      throw Forpy_Exception("input must have " +
                            std::to_string(orig_input_dim - 1) + " rows!");
    }
    if (prediction_output.rows() != solution.cols()) {
      throw Forpy_Exception("prediction_output must have " +
                            std::to_string(solution.cols()) + " rows!");
                            }
    if (solution_available) {
      if (rank_deficient) {
        Vec<IT> proj_input(input_dim);
        for (size_t i = 0; i < input_dim; ++i) {
          proj_input(i) = input(proj[i]);
        }
        prediction_output.noalias() = (proj_input.transpose() * solution)
          .transpose();
      } else {
        prediction_output.noalias() = (input.transpose() *
                                       solution).transpose();
      }
    } else {
      throw Forpy_Exception("No solution available! Check this before "
                            "predicting by using `get_solution_available`!");
    }
  };

  FORPY_IMPL(FORPY_REGRESSOR_PREDICTCOVAR_NOCOPY, ITR, LinearRegressor) {
    const auto &error_vars = this->error_vars.get_unchecked<Vec<IT>>();
    const auto &param_covar_mat_template = (
        this->param_covar_mat_template.get_unchecked<Mat<IT>>());
    if (static_cast<size_t>(covar_output.rows()) != annot_dim) {
      throw Forpy_Exception("covar_output must have " +
                            std::to_string(annot_dim) + " rows!");
    }
    if (static_cast<size_t>(covar_output.cols()) != annot_dim) {
      throw Forpy_Exception("covar_output must have " +
                            std::to_string(annot_dim) + " cols!");
    }
    if (solution_available) {
      predict_nocopy(input, prediction_output);
      covar_output.fill(static_cast<IT>(0.f));
      Vec<IT> proj_input(input_dim);
      if (rank_deficient) {
        for (size_t i = 0; i < input_dim; ++i) {
          proj_input(i) = input(proj[i]);
        }
      } else {
        proj_input = input;
      }
      for (size_t i=0; i < annot_dim; ++i) {
        covar_output(i,i) = (proj_input.transpose() *
                             param_covar_mat_template *
                             proj_input)(0,0) * error_vars(i);
      }
    } else {
      throw Forpy_Exception("No solution available! Check this before "
                            "predicting by using `get_solution_available`!");
    }
  };

  FORPY_IMPL(FORPY_REGRESSOR_GETCONSTANTPREDCOV, ITR, LinearRegressor) {
    throw Forpy_Exception("No constant covariance available! Check this "
                          "before predicting by using "
                          "`has_constant_prediction_covariance`!");
  };

  void LinearRegressor::freeze() {
    interval_frozen = true;
    current_interval = regint_t(-1, -1);
    this->annotation_mat = Empty();
    this->sample_mat = Empty();
    mu::apply_visitor(VReset(), this->annotation_mat_data);
    mu::apply_visitor(VReset(), this->sample_mat_data);
  };

  bool LinearRegressor::get_frozen() const { return interval_frozen; };

  size_t LinearRegressor::get_input_dimension() const {
    if (! initialized) {
      throw Forpy_Exception("Regressor not initialized!");
    }
    return orig_input_dim - 1;
  };

  size_t LinearRegressor::get_annotation_dimension() const {
    if (! initialized) {
      throw Forpy_Exception("Regressor not initialized!");
    }
    return annot_dim;
  };

  size_t LinearRegressor::get_n_samples() const { return n_samples; };

  bool LinearRegressor::forces_numerical_stability() const {
    return force_numerical_stability;
  };

  double LinearRegressor::get_numerical_zero_threshold() const {
    return numerical_zero_threshold;
  };

  bool LinearRegressor::operator==(const IRegressor &rhs) const {
     const auto *rhs_c = dynamic_cast<LinearRegressor const *>(&rhs);
     if (rhs_c == nullptr) {
       return false;
     } else {
       bool eq_rdf = (rank_deficient == rhs_c->rank_deficient);
       bool eq_orig_input_dim = (orig_input_dim == rhs_c->orig_input_dim);
       bool eq_proj = (proj == rhs_c->proj);
       bool eq_frc = (force_numerical_stability ==
                      rhs_c -> force_numerical_stability);
       bool eq_init = (initialized == rhs_c -> initialized);
       bool eq_mode = (double_mode == rhs_c -> double_mode);
       bool eq_thresh = (numerical_zero_threshold ==
                         rhs_c -> numerical_zero_threshold);
       bool eq_inp = input_dim == rhs_c -> input_dim;
       bool eq_adm = annot_dim == rhs_c -> annot_dim;
       bool eq_ind = sample_mat == rhs_c -> sample_mat;
       bool eq_and = annotation_mat == rhs_c -> annotation_mat;
       bool eq_nsamples = n_samples == rhs_c -> n_samples;
       bool eq_int = current_interval == rhs_c -> current_interval;
       bool eq_sol = solution == rhs_c -> solution;
       bool eq_cmt = (param_covar_mat_template ==
                      rhs_c -> param_covar_mat_template);
       bool eq_ev = error_vars == rhs_c -> error_vars;
       bool eq_av = solution_available == rhs_c -> solution_available;
       bool eq_frz = interval_frozen == rhs_c -> interval_frozen;
       if (! interval_frozen || !rhs_c -> interval_frozen) {
         if (! initialized) {
           return eq_frc && eq_init && eq_thresh;
         } else {
           return eq_init && eq_mode && eq_frc && eq_thresh && eq_inp &&
             eq_adm && eq_nsamples &&
             eq_int && eq_sol && eq_cmt && eq_ev && eq_av && eq_frz && eq_ind &&
             eq_and && eq_rdf && eq_orig_input_dim && eq_proj;
         }
       } else {
         if (! initialized) {
           return eq_frc && eq_init && eq_thresh;
         } else {
           return eq_mode && eq_frc && eq_thresh && eq_inp && eq_adm &&
             eq_nsamples &&
             eq_sol && eq_cmt && eq_ev && eq_av && eq_rdf && eq_orig_input_dim &&
             eq_proj;
         }
       }
     }
   };

  FORPY_IMPL(FORPY_REGRESSOR_INIT_NOCOPY, ITR, LinearRegressor) {
    this->sample_mat = MatCRef<IT>(sample_mat);
    this->annotation_mat = MatCRef<IT>(annotation_mat);
    this->input_dim = sample_mat.cols();
    this->orig_input_dim = this->input_dim;
    this->annot_dim = annotation_mat.cols();
    this->n_samples = sample_mat.rows();
    if (this->n_samples < this->input_dim - 1 + 2) {
      throw Forpy_Exception("Number of samples (" +
                            std::to_string(this->n_samples) +
                            ") must be at least input_dim (" +
                            std::to_string(this->input_dim - 1) + ") + 2!");
    }
    // Check for rank deficiency.
    Eigen::FullPivLU<Mat<IT>> decomposer(sample_mat);
    if (numerical_zero_threshold <= static_cast<IT>(0)) {
      decomposer.setThreshold(Eigen::Default);
    } else {
      decomposer.setThreshold(numerical_zero_threshold);
    }
    size_t rank = decomposer.rank();
    if (rank < input_dim) {
      if (rank == 0) {
        proj.resize(std::max<size_t>(1, 1));
        proj[0] = 0;
        rank += 1;
      } else {
        proj.resize(std::max<size_t>(rank, 1));
        // Determine a basis.
        Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> pivots(rank);
        IT premultiplied_threshold = decomposer.maxPivot() *
          decomposer.threshold();
        Eigen::Index p = 0;
        for(Eigen::Index i = 0; i < decomposer.nonzeroPivots(); ++i)
          if(std::abs(decomposer.matrixLU().coeff(i,i)) > premultiplied_threshold)
            pivots.coeffRef(p++) = i;
        for(size_t i = 0; i < rank; ++i)
          proj[i] = decomposer.permutationQ().indices().coeff(pivots.coeff(i));
      }
      // It's necessary to ensure the homogeneous dimension is part of the used
      // ones.
      if (std::find(proj.begin(), proj.end(), 0) == proj.end()) {
        // Swap in our constant homogeneous dimension.
        // Check for other selected constant ones.
        Eigen::Index const_dim = -1;
        for (size_t i = 0; i < rank; ++i) {
          if (sample_mat.col(proj[i]).minCoeff() ==
              sample_mat.col(proj[i]).maxCoeff()) {
            const_dim = i;
            break;
          }
        }
        if (const_dim == -1) {
          // Replace any, so the first.
          proj[0] = 0;
        } else {
          // Replace the right one.
          proj[const_dim] = 0;
        }
      }
      auto storptr = std::make_shared<Mat<IT>>(sample_mat.rows(), rank);
      Mat<IT> &sample_mat_data = *storptr;
      for (size_t i = 0; i < rank; ++i) {
        sample_mat_data.col(i) = sample_mat.col(proj[i]);
      }
      this->sample_mat_data = storptr;
      this->sample_mat.set<MatCRef<IT>>(sample_mat_data);
      rank_deficient = true;
      input_dim = rank;
    } else {
      rank_deficient = false;
      proj.clear();
    }
    if (std::is_same<IT, float>::value) {
      double_mode = false;
    } else {
      double_mode = true;
    }
    interval_frozen = false;
    regint_t real_interval;
    if (index_interval.first == -1 && index_interval.second == -1) {
      real_interval = std::make_pair(0, n_samples);
    } else {
      real_interval = index_interval;
    }
    if (sample_mat.rows() != annotation_mat.rows()) {
      throw Forpy_Exception("Number of rows for samples and annotations "
                            "do not agree!");
    };
    error_vars.set<Vec<IT>>(Vec<IT>::Zero(annot_dim));
    param_covar_mat_template.set<Mat<IT>>(Mat<IT>::Zero(input_dim, input_dim));
    solution.set<Mat<IT>>(Mat<IT>::Zero(input_dim, annot_dim));
    current_interval = regint_t(-1, -1);
    initialized = true;

    set_index_interval(real_interval);
  };

  bool LinearRegressor::needs_homogeneous_input_data() const { return true; };

  bool LinearRegressor::check_interval_valid(const regint_t & interval) {
    return (interval.second >= interval.first &&
            interval.first >= 0 &&
            static_cast<size_t>(interval.second) <= n_samples);
  };

  std::unique_ptr<IRegressor> LinearRegressor::empty_duplicate() const {
    return std::unique_ptr<IRegressor>(new LinearRegressor(
        force_numerical_stability,
        numerical_zero_threshold));
  };

  size_t LinearRegressor::get_kernel_dimension() const {
    if (! initialized) {
      throw Forpy_Exception("Uninitialized regressor!");
    }
    return input_dim - 1;
  };

  float LinearRegressor::get_residual_error() const {
    if (! initialized) {
      throw Forpy_Exception("Uninitialized regressor!");
    }
    if (double_mode) {
      return error_vars.get_unchecked<Vec<double>>().mean();
    } else {
      return error_vars.get_unchecked<Vec<float>>().mean();
    }
  };

  std::string LinearRegressor::get_name() const {
    return "LinearRegressor";
  }

} // namespace forpy
