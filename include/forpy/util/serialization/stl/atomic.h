#pragma once
#ifndef FORPY_UTIL_SERIALIZATION_STL_ATOMIC_H_
#define FORPY_UTIL_SERIALIZATION_STL_ATOMIC_H_

#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>

#include <atomic>

namespace cereal {
template <class Archive, typename E>
void save(Archive &archive, const std::atomic<E> &value,
          const unsigned int & /*version*/) {
  archive(value.load());
};

template <class Archive, typename E>
void load(Archive &archive, std::atomic<E> &value,
          const unsigned int & /*version*/) {
  E tmp;
  archive(tmp);
  value.store(tmp);
};
}  // namespace cereal

#endif  // FORPY_UTIL_SERIALIZATION_STL_ATOMIC_H_
