/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_IMPURITIES_CLASSIFICATIONERROR_H_
#define FORPY_IMPURITIES_CLASSIFICATIONERROR_H_

#include "../global.h"

#include "../util/serialization/basics.h"

#include <limits>
#include <vector>

#include "../util/exponentials.h"
#include "./ientropyfunction.h"

namespace forpy {
/**
 * \brief Computes the classification error as \f$1-\max(p_i)\f$.
 *
 * \ingroup forpyimpuritiesGroup
 */
class ClassificationError : public IEntropyFunction {
 public:
  ClassificationError();
  ~ClassificationError();

  inline float operator()(const float *class_members_numbers, const size_t &n,
                          const float &fsum) const {
    // Deal with the special case quickly.
    if (fsum == 0.f) return 0.f;

    return 1.f -
           *std::max_element(class_members_numbers, class_members_numbers + n) /
               fsum;
  }

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const ClassificationError & /*self*/) {
    stream << "forpy::ClassificationError";
    return stream;
  };
  bool operator==(const IEntropyFunction &rhs) const;

 private:
  DISALLOW_COPY_AND_ASSIGN(ClassificationError);

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(cereal::make_nvp("base", cereal::base_class<IEntropyFunction>(this)));
  }
};
}  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::ClassificationError);
#endif  // FORPY_IMPURITIES_CLASSIFICATIONERROR_H_
