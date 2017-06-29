#include <forpy/util/regression/iregressor.h>

namespace forpy {

  FORPY_IMPL(FORPY_REGRESSOR_INIT, ITR, IRegressor) {
    if (needs_homogeneous_input_data()) {
      auto homogeneous = std::make_shared<Mat<IT>>(Mat<IT>::Ones(
          sample_mat->rows(), sample_mat->cols()+1));
      homogeneous->block(0, 1,
                         sample_mat->rows(),
                         sample_mat->cols()) = *sample_mat;
      sample_mat_data = homogeneous;
    } else {
      sample_mat_data = sample_mat;
    }
    annotation_mat_data = annotation_mat;
    initialize_nocopy(*sample_mat_data.get_unchecked<std::shared_ptr<const Mat<IT>>>(),
                      *annotation_mat_data.get_unchecked<std::shared_ptr<const Mat<IT>>>(),
                      index_interval);
  }; 

  FORPY_IMPL(FORPY_REGRESSOR_PREDICT, ITR, IRegressor) {
    if (needs_homogeneous_input_data()) {
      Vec<IT> homogeneous(Vec<IT>::Ones(input.rows() + 1));
      homogeneous.block(1, 0, input.rows(), 1) = input;
      predict_nocopy(homogeneous, prediction_output);
    } else {
      predict_nocopy(input, prediction_output);
    }
  }

  FORPY_IMPL(FORPY_REGRESSOR_PREDICTCOVAR, ITR, IRegressor) {
    if (needs_homogeneous_input_data()) {
      Vec<IT> homogeneous(Vec<IT>::Ones(input.rows() + 1));
      homogeneous.block(1, 0, input.rows(), 1) = input;
      predict_covar_nocopy(homogeneous, prediction_output, covar_output);
    } else {
      predict_covar_nocopy(input, prediction_output, covar_output);
    }
  }

  FORPY_IMPL_NOTAVAIL(FORPY_REGRESSOR_INIT_NOCOPY, ITR, IRegressor);

  FORPY_IMPL_NOTAVAIL(FORPY_REGRESSOR_PREDICT_NOCOPY, ITR, IRegressor);

  FORPY_IMPL_NOTAVAIL(FORPY_REGRESSOR_PREDICTCOVAR_NOCOPY, ITR, IRegressor);

  FORPY_IMPL_NOTAVAIL(FORPY_REGRESSOR_GETCONSTANTPREDCOV, ITR, IRegressor);

} // namespace forpy
