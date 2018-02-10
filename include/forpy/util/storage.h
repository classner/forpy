#pragma once
#ifndef FORPY_UTIL_STORAGE_H_
#define FORPY_UTIL_STORAGE_H_

#ifdef __GNUC__
#ifndef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#endif
#endif
#include <mapbox/variant_cast.hpp>
#ifdef __GNUC__
#ifndef __clang__
#pragma GCC diagnostic pop
#endif
#endif
#include "./serialization/variant.h"

namespace forpy {

//////////////////////////////////////////////////////////////////////////////
/// Design concepts behind this storage.
///
/// Aims are:
/// * easy and serializable storage for internal types,
/// * encoding of multiple possible return types,
/// * high efficiency,
/// * no hassle for library users,
/// * remain as compatible with the standard as possible (c.f., std::variant
///   in C++ 17).
///
/// Since it should be possible to use this library with large datasets that
/// shouldn't have to be copied when library functions are called, this
/// implies some additional constraints.
///
/// The standard says that std::variant can't hold pointer types, which is
/// necessary for this to work. The currently provided mu::variant can do
/// this.
///
/// To not influence library user experience and not create problems when
/// feeding the library with data, all external methods accept either
/// std::shared_ptr<Eigen::Mat>s if ownership of the data must be set.
/// That's why the `store` suffixed variants contain shared_ptr's (otherwise
/// the data would have to be copied).
///
/// In all other cases, the variants contain the data or MatRefs to it
/// directly for internal library use.
///
/// The ptr_variant is a subclass of the variant for which an automatic return
/// type conversion to Python is applied. `Empty` is similar to
/// `std::monostate`. The VReset visitor can be used to clear `ptr_variant`s.
//////////////////////////////////////////////////////////////////////////////

template <typename... Ts>
struct ptr_variant : mu::variant<Ts...> {
  using Base = mu::variant<Ts...>;
  using Base::Base;
};

/**
 * \brief A struct to represent an empty variant.
 *
 * This is necessary because the first variant type must be default
 * constructible. For types for which this is not possible, this 'empty' type
 * can be used.
 */
struct Empty {
  template <typename Archive>
  void serialize(Archive &, const uint &){};
  bool operator==(const Empty &) const { return true; };
  float *data() const {
    throw ForpyException("Trying to access an empty data storage.");
    return nullptr; };
  size_t &innerStride() const {
    throw ForpyException("Trying to access an empty data storage.");
  };
  size_t &outerStride() const {
    throw ForpyException("Trying to access an empty data storage.");
  };
  friend std::ostream &operator<<(std::ostream &stream, const Empty &) {
    stream << "forpy::Empty";
    return stream;
  };
};

/**
 * \brief Comparison visitor.
 *
 * Compares to Eigen::Matrix variants for element-wise approximate equality.
 */
struct MatEqVis {
  template <typename T, typename U>
  bool operator()(const T &, const U &) {
    return false;
  };
  template <typename T>
  bool operator()(const T &lhs, const T &rhs) {
    return lhs.isApprox(rhs);
  };
  inline bool operator()(const Empty &, const Empty &) { return true; };
};

/**
 * \brief Call the reset operation on a pointer variant.
 */
struct VReset {
  template <class T>
  void operator()(T &pointer) const {
    pointer.reset();
  }
};

/**
 * \brief Variant for storing shared_ptrs to the stored data matrix type.
 */
template <template <typename> class STOT>
using DataStore = typename mu::variant<
    std::shared_ptr<const STOT<float>>, std::shared_ptr<const STOT<double>>,
    std::shared_ptr<const STOT<uint>>, std::shared_ptr<const STOT<uint8_t>>>;

/**
 * \brief Storing a variant of the provided data container type.
 */
template <template <typename> class STOT>
using Data = typename mu::variant<Empty, STOT<float>, STOT<double>, STOT<uint>,
                                  STOT<uint8_t>>;

using DataV = typename mu::variant<std::vector<float>, std::vector<double>,
                                   std::vector<uint>, std::vector<uint8_t>>;

/**
 * \brief Get the core datatype with removed pointer, reference and const
 * modifiers.
 */
template <typename T>
struct get_core {
  typedef typename std::remove_pointer<T>::type _tmp;
  typedef typename std::remove_reference<_tmp>::type __tmp;
  typedef typename std::remove_cv<__tmp>::type type;
};

template <typename V, class... VarArgs>
V GetWithDefVar(
    const std::unordered_map<std::string, mu::variant<VarArgs...>> &m,
    std::string const &key, const V &defval) {
  typename std::unordered_map<std::string,
                              mu::variant<VarArgs...>>::const_iterator it =
      m.find(key);
  if (it == m.end()) return defval;
  return mu::static_variant_cast<V>(it->second);
}

}  // namespace forpy
#endif  // FORPY_UTIL_STORAGE_H_
