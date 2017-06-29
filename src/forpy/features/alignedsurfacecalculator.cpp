#include <forpy/features/alignedsurfacecalculator.h>

namespace forpy {

  AlignedSurfaceCalculator::AlignedSurfaceCalculator() {};

  std::vector<FeatCalcParams> AlignedSurfaceCalculator::propose_params() {
    return std::vector<FeatCalcParams>(1);
  };

  size_t AlignedSurfaceCalculator::required_num_features() const { return 1; }

  FORPY_IMPL(FORPY_ISURFCALC_CALC, ITFTEQ, AlignedSurfaceCalculator) {
    if (data->cols() != 1) {
      throw Forpy_Exception("This surface calculator only allows 1 dimension!");
    }
    retval = data;
  };

  FORPY_IMPL(FORPY_ISURFCALC_CALCS, ITFTEQ, AlignedSurfaceCalculator) {
    if (data.rows() != 1) {
      throw Forpy_Exception("This function only predicts 1 sample!");
    }
    if (feature_selection.size() != 1) {
      throw Forpy_Exception("A feature selection for exactly one dimension is "
                            "required!");
    }
    if (feature_selection[0] >= static_cast<size_t>(data.cols())) {
      throw Forpy_Exception("The selected feature is out of bounds of this data!");
    } 
    *retval = data(0, feature_selection[0]);
  };

  bool AlignedSurfaceCalculator::operator==(const ISurfaceCalculator &rhs) const {
    const auto *rhs_c = dynamic_cast<AlignedSurfaceCalculator const *>(&rhs);
    if (rhs_c == nullptr)
      return false;
    else
      return true;
  };

} // namespace forpy
