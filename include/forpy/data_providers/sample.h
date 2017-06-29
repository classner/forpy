/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_DATA_PROVIDERS_SAMPLE_H_
#define FORPY_DATA_PROVIDERS_SAMPLE_H_

#include "../global.h"
#include "../types.h"

namespace forpy {

  /**
   * \brief Stores information about one data sample.
   *
   * This form of storage is mainly important since samples may be generated
   * by data providers on the fly, so it cannot be assumed that data has a 
   * contiguous memory layout.
   *
   * The data vectors can have a \ref data_step, but the annotations cannot and
   * must have a contiguous memory layout.
   */
  template <typename IT, typename AT>
  struct Sample {
    const VecCMap<IT> data;
    const VecCMap<AT> annotation;
    float weight;
    std::shared_ptr<const Mat<IT>> parent_dt;
    std::shared_ptr<const Mat<AT>> parent_at;

    Sample() {};

    Sample(const VecCMap<IT> &data,
           const VecCMap<AT> &annotation,
           const float &weight=1.f,
           const std::shared_ptr<const Mat<IT>> &parent_dt=nullptr,
           const std::shared_ptr<const Mat<AT>> &parent_at=nullptr) :
        data(data),
        annotation(annotation),
        weight(weight),
        parent_dt(parent_dt),
        parent_at(parent_at) {}

    bool operator==(const Sample &rhs) const {
      if (weight != rhs.weight) return false;
      for (size_t i = 0; i < data.cols(); ++i) {
        if (data[i] != rhs.data[i]) return false;
      }
      for (size_t i = 0; i < annotation.cols(); ++i) {
        if (annotation[i] != rhs.annotation[i]) return false;
      }
      return true;
    }
  };
}  // namespace forpy
#endif  // FORPY_DATA_PROVIDERS_SAMPLE_H_
