/* Author: Christoph Lassner */
#pragma once
#ifndef FORPY_UTIL_CHECKS_H_
#define FORPY_UTIL_CHECKS_H_

#include <algorithm>
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>

#include "../global.h"

namespace forpy {

/**
 * \brief Tests whether the sum of all elements in vec is less than limit.
 */
template <typename T>
static bool safe_pos_sum_lessoe_than(const std::vector<T> &vec,
                                     const T &limit) {
  if (!std::is_arithmetic<T>()) {
    // This case should've been already detected at compile time because
    // of the use of std::numeric_limits in the function later on, but
    // better be safe.
    throw ForpyException(
        "The safe_pos_sum_lessoe_than function can "
        "only be called for arithmetic type vectors.");
  }
  T sum_so_far = static_cast<T>(0);
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] < static_cast<T>(0) || limit - vec[i] < sum_so_far) {
      return false;
    } else {
      sum_so_far += vec[i];
    }
  }
  return true;
};

/**
 * \brief Tests whether the sum of all elements in vec1 and vec2 is less than
 * limit.
 */
template <typename T>
static bool safe_pos_sum_lessoe_than(const std::vector<T> &vec1,
                                     const std::vector<T> &vec2,
                                     const T &limit) {
  std::vector<T> joined;
  joined.insert(joined.end(), vec1.begin(), vec1.end());
  joined.insert(joined.end(), vec2.begin(), vec2.end());
  return safe_pos_sum_lessoe_than(joined, limit);
};

/**
 * \brief Tests whether the sum of all elements in vec is less than the numeric
 * limit of its type.
 */
template <typename T>
static bool safe_pos_sum_lessoe_than(const std::vector<T> &vec) {
  return safe_pos_sum_lessoe_than(vec, std::numeric_limits<T>::max());
};

/**
 * \brief Tests whether the sum of all elements in vec1 and vec2 is less than
 * the numeric limit of their type.
 */
template <typename T>
static bool safe_pos_sum_lessoe_than(const std::vector<T> &vec1,
                                     const std::vector<T> &vec2) {
  return safe_pos_sum_lessoe_than(vec1, vec2, std::numeric_limits<T>::max());
};

/**
 * \brief Tests whether all element ids are valid.
 */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
static bool check_elem_ids_ok(const size_t &n_samples,
                              const std::vector<size_t> &elem_ids) {
  return elem_ids.end() ==
         std::find_if(elem_ids.begin(), elem_ids.end(),
                      std::bind2nd(std::greater_equal<size_t>(), n_samples));
}
#pragma clang diagnostic pop

};      // namespace forpy
#endif  // FORPY_UTIL_CHECKS_H_
