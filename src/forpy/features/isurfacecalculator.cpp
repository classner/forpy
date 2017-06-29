#include <forpy/features/isurfacecalculator.h>

namespace forpy {
  ISurfaceCalculator::ISurfaceCalculator() {};
  ISurfaceCalculator::~ISurfaceCalculator() {};

  bool ISurfaceCalculator::is_compatible_to(const IDataProvider &dprov) const {
    return true;
  };

  bool ISurfaceCalculator::needs_elements_prepared() const {
    return true;
  };

  FORPY_IMPL_NOTAVAIL(FORPY_ISURFCALC_CALC, ITFT, ISurfaceCalculator);

  FORPY_IMPL_NOTAVAIL(FORPY_ISURFCALC_CALCS, ITFT, ISurfaceCalculator);

} // namespace forpy
