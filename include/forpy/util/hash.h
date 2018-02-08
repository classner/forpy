#pragma once
#ifndef FORPY_UTIL_HASH_H_
#define FORPY_UTIL_HASH_H_

#include <cstddef>
#include <vector>

namespace forpy {
  /**
   * \brief Quick and easy implementation of 64-bit FNV 1a hash.
   *
   * The FNV 1a is easy to implement and has still good enough characteristics
   * to be used for this application.
   *
   * See http://www.isthe.com/chongo/tech/comp/fnv/index.html and for
   * comparisons and more information
   * http://eternallyconfuzzled.com/tuts/algorithms/jsw_tut_hashing.aspx and
   * http://burtleburtle.net/bob.
   */
  static size_t hash_fnv_1a(const unsigned char *key, const size_t &len) {
    // hash = offset_basis
    size_t h = 14695981039346656037ULL;
    // for each octet_of_data to be hashed
    for (size_t i = 0; i < len; ++i) {
      // hash = hash xor octet_of_data
      // hash = hash * FNV_prime
      h = (h ^ key[ i ]) * 1099511628211;
    }
    // return hash
    return h;
  };

  /** \brief A simple vector<size_t> hasher. */
  struct vector_hasher {
    /** Hash function. */
    size_t operator()(const std::vector<size_t> &t) const {
      if (t.empty()) return 0;
      return hash_fnv_1a(reinterpret_cast<const unsigned char *>(&t[0]),
                         t.size() * sizeof(size_t) / sizeof(unsigned char));
    };
  };
} // namespace forpy
#endif // FORPY_UTIL_HASH_H_

