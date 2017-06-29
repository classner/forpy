#include <forpy/features/featcalcparams.h>

namespace forpy {

  bool FeatCalcParams::operator==(const FeatCalcParams &rhs) const {
    bool equiv = true;
    for (int i = 0; i < 9; ++i) {
      if (weights[i] != rhs.weights[i]) {
        equiv = false;
        break;
      }
    }
    for (int i = 0; i < 2; ++i) {
      if (offsets[i] != rhs.offsets[i]) {
        equiv = false;
        break;
      }
    }
    return equiv;
  };

} // namespace forpy
