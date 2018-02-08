/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_IMPURITIES_INDUCEDENTROPY_H_
#define FORPY_IMPURITIES_INDUCEDENTROPY_H_

#include "../util/serialization/basics.h"

#include <limits>
#include <vector>

#include "../global.h"
#include "../util/checks.h"
#include "../util/exponentials.h"
#include "./ientropyfunction.h"

namespace forpy {

/**
 * \brief Computes the induced p entropy.
 *
 * Works correctly up to a total sum of elements of
 * numeric_limits<float>::max().
 *
 * This is the induced p-metric of the vector of \f$n\f$ class probabilities
 * and the point of maximum unorder (the vector with all entries
 * \f$\frac{1}{n}\f$) in the n-dimensional space without applying the root.
 * It is equal to the Gini-measure for \f$p=2\f$.
 *
 * The definition for \f$c\f$ classes:
 * \f[\sum_{i=1}^{c} \left\Vert p_i - \frac{1}{c}\right\Vert ^p\f].
 *
 * The differential entropy for a normal distribution with covariance
 * matrix \f$\Sigma\f$ in \f$n\f$ dimensions is defined as:
 * \f[\frac{1}{\sqrt{p^n}}\cdot\left(\sqrt{2\pi}^n\cdot\sqrt{\left|\Sigma\right|}\right)^{-(p-1)}\f]
 *
 * In the differential normal case, the most useful values for \f$p\f$ are
 * very close to 1 (e.g. 1.00001)! \f$p=2\f$ is already equivalent to the
 * infinite norm!
 *
 * \ingroup forpyimpuritiesGroup
 */
class InducedEntropy : public IEntropyFunction {
 public:
  inline explicit InducedEntropy(const float &p) : p(p) {
    if (p <= 0.f) {
      throw ForpyException("p must be > 0.f.");
    }
  };
  inline ~InducedEntropy(){};

  inline virtual float operator()(const float *class_members_numbers,
                                  const size_t &n, const float &fsum) const {
    FASSERT(static_cast<float>(std::accumulate(class_members_numbers,
                                               class_members_numbers + n,
                                               static_cast<float>(0))) == fsum);
    if (fsum == 0.f) return 0.f;
    if (p == 2.f) {
      float sq_left = 0.f;
      for (size_t i = 0; i < n; ++i, ++class_members_numbers)
        sq_left += *class_members_numbers * *class_members_numbers;
      return 1.f - sq_left / (fsum * fsum);
    } else {
      float n_classes_f = static_cast<float>(n);
      float max_unorder_val = 1.f / n_classes_f;
      float entropy_sum;
      if (ceilf(p) == p || floorf(p) == p) {
        const uint whole_p = static_cast<uint>(p);
        entropy_sum = fpowi(1.f - max_unorder_val, whole_p) +
                      (n_classes_f - 1.f) * fpowi(max_unorder_val, whole_p);
        for (size_t i = 0; i < n; ++i) {
          float quot = *(class_members_numbers++) / fsum;
          entropy_sum -= fpowi(fabs(quot - max_unorder_val), whole_p);
        }
      } else {
        entropy_sum = powf(1.f - max_unorder_val, p) +
                      (n_classes_f - 1.f) * powf(max_unorder_val, p);
        for (size_t i = 0; i < n; ++i) {
          float quot = *(class_members_numbers++) / fsum;
          entropy_sum -= powf(fabs(quot - max_unorder_val), p);
        }
      }
      return entropy_sum;
    }
  };

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const InducedEntropy &self) {
    stream << "forpy::InducedEntropy[p=" << self.get_p() << "]";
    return stream;
  };

  inline bool operator==(const IEntropyFunction &rhs) const {
    const auto *rhs_c = dynamic_cast<InducedEntropy const *>(&rhs);
    if (rhs_c == nullptr) {
      return false;
    } else {
      bool eq_p = p == rhs_c->p;
      return eq_p;
    }
  };

  inline float get_p() const { return p; };

 private:
  InducedEntropy(){};

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(cereal::make_nvp("base", cereal::base_class<IEntropyFunction>(this)),
       CEREAL_NVP(p));
  }

  float p;
  DISALLOW_COPY_AND_ASSIGN(InducedEntropy);
};
}  // namespace forpy

CEREAL_REGISTER_TYPE(forpy::InducedEntropy);
/*CEREAL_REGISTER_DYNAMIC_INIT(forpy::InducedEntropy);
  CEREAL_FORCE_DYNAMIC_INIT(forpy::InducedEntropy);*/
#endif  // FORPY_IMPURITIES_INDUCEDENTROPY_H_
