/* Author: Christoph Lassner */
#pragma once
#ifndef FORPY_UTIL_SAMPLING_H_
#define FORPY_UTIL_SAMPLING_H_

#include <cereal/types/polymorphic.hpp>
#include <cereal/access.hpp>

#include <vector>
#include <random>
#include <type_traits>
#include <cstdint>

#include "../global.h"
#include "./serialization/serialization.h"

namespace forpy {
  /**
   * \brief Integer binomial with overflow detection.
   *
   * The code here is based on the following short article:
   * http://etceterology.com/fast-binomial-coefficients. In the article, the
   * possibility to use a lookup-table is introduced. This is not done here,
   * since it is not particularly useful for the use-case. The code has been
   * thoroughly reviewed and tested.
   *
   * \return nChoosek or -1 if an overflow was detected.
   */
  static int64_t ibinom(const int &n, int k) {
    FASSERT(n >= 0 && k >= 0);
    int i;
    int64_t b;

    if (0 == k || n == k) return 1LL;
    if (k > n) return 0LL;

    if (k > (n - k)) k = n - k;
    if (1 == k) return static_cast<int64_t>(n);

    b = 1LL;
    for (i = 1; i <= k; ++i) {
      b *= (n - (k - i));
      if (b < 0) return -1LL; /* Overflow */
      b /= i;
    }
    return b;
  };

  /**
   * \brief A lazy evaluation sampling without replacement.
   *
   * Returns a set of num unique numbers in range [min, max].
   *
   * This needs to be stateful. Hence, it must store for each element
   * whether it has been used yet. This renders it inefficient for use
   * cases where only few instances need to be drawn.
   *
   * min and max are both inclusive.
   */
  template <typename T>
  class SamplingWithoutReplacement {
  public:
    SamplingWithoutReplacement(const T &min,
      const T &max,
      const std::shared_ptr<std::mt19937> &random_engine)
      : min(min), random_engine(random_engine) {
      FASSERT(max >= min);
      indices = std::vector<T>(max - min + 1);
      std::iota(indices.begin(), indices.end(), 0);
      index = 0;
      dist = std::uniform_int_distribution<size_t>(0, max - min);
    };

    /**
     * \brief Returns true if a sample can be drawn without raising an exception.
     */
    bool sample_available() const {
      return index < indices.size();
    };

    /**
     * \brief Gets the next sample.
     */
    T get_next() {
      T return_value;
      {
        if (index >= indices.size())
          throw Forpy_Exception("Tried to redraw without replacement "
          "from a limited set where the num of remaining examples was 0.");
        size_t rand_index = dist(*random_engine);
        std::swap(indices[index], indices[rand_index]);
        if (index != indices.size() - 1)
          dist.param(std::uniform_int_distribution<size_t>::param_type(dist.min() + 1, dist.max()));
        return_value = min + indices[index++];
      }
      return return_value;
    };

    inline friend std::ostream &operator<<(std::ostream &stream,
                                           const SamplingWithoutReplacement &self) {
      stream << "forpy::SamplingWithoutReplacement[" <<
        self.min << " (inc):" << (self.min + self.indices.size() - 1) << " (inc), " <<
        (self.indices.size() - self.index) << " available]";
      return stream;
    };

    bool operator==(const SamplingWithoutReplacement<T> &rhs) const {
      return (min == rhs.min &&
              random_engine == rhs.random_engine &&
              dist == rhs.dist &&
              indices == rhs.indices &&
              index == rhs.index);
    };

   private:
    SamplingWithoutReplacement() {};

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive &ar, const uint &) {
      ar(CEREAL_NVP(min),
         CEREAL_NVP(random_engine),
         CEREAL_NVP(dist),
         CEREAL_NVP(indices),
         CEREAL_NVP(index));
    };

    T min;
    std::shared_ptr<std::mt19937> random_engine;
    std::uniform_int_distribution<T> dist;
    std::vector<T> indices;
    size_t index;
  };

  /**
   * \brief Sampling without replacement.
   *
   * Returns a set of num unique numbers in range [min, max]. T must be an
   * integral datatype.
   *
   * This implementation does not need to be stateful, since the algorithm
   * completes in one go. VERY efficient in any case. It is inspired by
   * various algorithms from the below sources, but surpasses them in
   * terms of efficiency and distribution of the values. The algorithm it was
   * mainly inspired by iterates over the sample range once and picks
   * the next number by a random distribution. In the original version,
   * the random distribution is badly designed.
   *
   * See:
   * http://codegolf.stackexchange.com/questions/4772/random-sampling-without-replacement
   * http://www.cplusplus.com/reference/cstdlib/rand/
   * http://stackoverflow.com/questions/311703/algorithm-for-sampling-without-replacement
   *
   * \param num Number of examples to be selected from the range.
   * \param min Minimum of range (inclusive).
   * \param max Maximum of range (inclusive).
   * \param random_engine The random engine to use for random number generation.
   * \param return_sorted If true, returns the numbers sorted (no overhead),
   *    otherwise they will be shuffled (overhead).
   */
  template <typename T>
  static std::vector<T> unique_indices(T num,
                                       T min,
                                       const T &max,
                                       std::mt19937 *random_engine,
                                       bool return_sorted = false) {
    static_assert(std::is_integral<T>::value, "T must be integral!");
    if (max < min)
      throw Forpy_Exception("Invalid sample range.");
    if (num > max - min + 1)
      throw Forpy_Exception("Sample size larger than range.");
    std::vector<T> result(num);
    if (num == max - min + 1) {
      // The full range of numbers is required.
      std::iota(result.begin(), result.end(), min);
    } else {
      std::geometric_distribution<T> dist;
      if (num + 1 < max - min + 1) {
        // Draw unique samples by iterating once over the numbers.
        // There are only as many 'steps' as there should be numbers drawn.
        // The difference between two numbers uniquely drawn from the set is
        // distributed with a geometric distribution.
        // The mean difference between two numbers is (max-min+1)/(num+1).
        dist = std::geometric_distribution<T>(static_cast<float>(num + 1) /
                                              static_cast<float>(max - min + 1));
      }
      // A truncated geometric distribution must be used to guarantee that
      // all numbers can be drawn from the range. r is the truncation level.
      // i is the remaining number of elements to be drawn.
      T r, i = num;
      while (i--) {
        r = (max - min + 1 - i);
        if (r <= 1) {
          // The next value must be used, since there is no more 'space' for
          // additional numbers.
          result[i] = min;
        }
        else {
          // Do the next step and handle truncation.
          result[i] = min += std::min<T>(dist(*random_engine) + 1, r - 1);
        }
      }
    }
    // The numbers are now ready, sorted in the appropriate interval.
    if (!return_sorted) {
      std::shuffle(result.begin(), result.end(), *random_engine);
    }
    return result;
  };
};  // namespace forpy
#endif  // FORPY_UTIL_SAMPLING_H_
