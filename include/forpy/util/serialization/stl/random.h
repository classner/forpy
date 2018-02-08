#pragma once
#ifndef FORPY_UTIL_SERIALIZATION_STL_RANDOM_H_
#define FORPY_UTIL_SERIALIZATION_STL_RANDOM_H_

#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>

#include <random>
#include <sstream>

namespace cereal {
  template<class Archive, class E, size_t K>
  void save(Archive & archive, const std::shuffle_order_engine<E, K> &value,
            const unsigned int &/*version*/) {
    // Serialize via a string.
    std::stringstream ss;
    ss << value;
    std::string state = ss.str();
    archive(state);
  };

  template<class Archive, class E, size_t K>
    void load(Archive & archive, std::shuffle_order_engine<E, K> &value,
              const unsigned int &/*version*/) {
    std::string tmp;
    archive(tmp);
    std::stringstream ss(tmp);
    ss >> value;
  };

  template<class Archive>
  void save(Archive & archive, const std::mt19937 &value,
            const unsigned int &/*version*/) {
    // Serialize via a string.
    std::stringstream ss;
    ss << value;
    std::string state = ss.str();
    archive(state);
  };

  template<class Archive>
  void load(Archive & archive, std::mt19937 &value,
            const unsigned int &/*version*/) {
    std::string tmp;
    archive(tmp);
    std::stringstream ss(tmp);
    ss >> value;
  };

  template<class Archive, typename T>
    void save(Archive & archive, const std::uniform_int_distribution<T> &value,
              const unsigned int &/*version*/) {
    // Serialize via a string.
    std::stringstream ss;
    ss << value;
    std::string state = ss.str();
    archive(state);
  };

  template<class Archive, typename T>
  void load(Archive & archive, std::uniform_int_distribution<T> &value,
            const unsigned int &/*version*/) {
    std::string tmp;
    archive(tmp);
    std::stringstream ss(tmp);
    ss >> value;
  };

} // namespace cereal

#endif // FORPY_UTIL_SERIALIZATION_STL_RANDOM_H_
