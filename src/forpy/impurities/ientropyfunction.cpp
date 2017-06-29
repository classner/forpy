#include <forpy/impurities/ientropyfunction.h>

#include <numeric>

namespace forpy {
  IEntropyFunction::IEntropyFunction() {}

  IEntropyFunction::~IEntropyFunction() {}

  float IEntropyFunction::operator()(
      const std::vector<float> &class_members_numbers) const {
    return operator()(class_members_numbers,
        static_cast<float>(std::accumulate(class_members_numbers.begin(),
                                           class_members_numbers.end(),
                                           static_cast<float>(0))));
  };

  float IEntropyFunction::differential_normal(const MatCRef<float> &covar_matrix)
    const {
    FASSERT(covar_matrix.rows() == covar_matrix.cols());
    float det = covar_matrix.determinant();
    return differential_normal(det, static_cast<uint>(covar_matrix.rows()));
  };
} // namespace forpy
