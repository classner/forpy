#include <forpy/impurities/ientropyfunction.h>

#include <numeric>

namespace forpy {
IEntropyFunction::IEntropyFunction() {}

IEntropyFunction::~IEntropyFunction() {}

float IEntropyFunction::operator()(
    const std::vector<float> &class_members_numbers) const {
  return operator()(class_members_numbers,
                    static_cast<float>(std::accumulate(
                        class_members_numbers.begin(),
                        class_members_numbers.end(), static_cast<float>(0))));
};

}  // namespace forpy
