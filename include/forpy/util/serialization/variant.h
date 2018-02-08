#pragma once
#ifndef FORPY_UTIL_VARIANT_H_
#define FORPY_UTIL_VARIANT_H_

#include "../../global.h"

#include "./basics.h"

#include <mapbox/variant.hpp>
#include "./serialization.h"

namespace mu = mapbox::util;

namespace cereal {

template <class Archive>
struct variant_save_visitor {
  explicit variant_save_visitor(Archive &ar) : m_ar(ar){};
  template <typename T>
  void operator()(T const &value) const {
    m_ar(CEREAL_NVP(value));
  };

 private:
  Archive &m_ar;
};

template <class Archive, typename... Ts>
void save(Archive &ar, const mu::variant<Ts...> &v, const uint &) {
  int which = v.which();
  ar(CEREAL_NVP(which));
  variant_save_visitor<Archive> visitor(ar);
  mu::apply_visitor(visitor, v);
};

template <typename T>
struct remove_first_type {};
template <typename T, typename... Ts>
struct remove_first_type<std::tuple<T, Ts...>> {
  typedef std::tuple<Ts...> type;
};

template <class S>
struct variant_impl {
  struct load_null {
    template <class Archive, class V>
    static void invoke(Archive &, int, V &, const unsigned int &) {}
  };

  struct load_impl {
    template <class Archive, class V>
    static void invoke(Archive &ar, int which, V &v,
                       const unsigned int &version) {
      if (which == 0) {
        // note: A non-intrusive implementation (such as this one) necessarily
        // has to copy the value. This wouldn't be necessary with an
        // implementation that de-serialized to the address of the aligned
        // storage included in the variant.
        using head_type = typename std::tuple_element<0, S>::type;
        head_type value;
        ar(CEREAL_NVP(value));
        v.template set<head_type>(value);
      }
      typedef typename remove_first_type<S>::type type;
      variant_impl<type>::load(ar, which - 1, v, version);
    }
  };

  template <class Archive, class V>
  static void load(Archive &ar, int which, V &v, const unsigned int version) {
    typedef typename std::conditional<std::tuple_size<S>::value == 0, load_null,
                                      load_impl>::type typex;
    typex::invoke(ar, which, v, version);
  }
};

template <class Archive, typename... Ts>
void load(Archive &ar, mu::variant<Ts...> &v, const unsigned int &version) {
  int which;
  ar(CEREAL_NVP(which));
  if (which >= static_cast<int>(sizeof...(Ts))) {
    throw forpy::ForpyException("Unsupported library version.");
  }
  variant_impl<std::tuple<Ts...>>::load(ar, which, v, version);
};
}  // namespace cereal

#endif  // FORPY_UTIL_VARIANT_H_
