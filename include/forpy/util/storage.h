#pragma once
#ifndef FORPY_UTIL_STORAGE_H_
#define FORPY_UTIL_STORAGE_H_

#include "./variant.h"

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

  template<typename... Ts>
  struct ptr_variant : mu::variant<Ts...>{
    using Base = mu::variant<Ts...>;
    using Base::Base;
  };

  struct Empty{
    template<typename Archive>
    void serialize(Archive &, const uint &) {};
    bool operator==(const Empty &) const {return true;};
    float *data() const {return nullptr;};
    friend std::ostream &operator<<(std::ostream &stream, const Empty &) {
      stream << "forpy::Empty";
      return stream;
    };
  };

  struct MatEqVis {
    template<typename T, typename U>
    bool operator()(const T &, const U &) {
      return false;
    };
    template<typename T>
    bool operator()(const T &lhs, const T &rhs) {
      return lhs.isApprox(rhs);
    };
    inline bool operator()(const Empty &, const Empty &) {
      return true;
    };
  };

  struct VReset {
    template<class T>
    void operator()(T &pointer) const {
      pointer.reset();
    }
  };

  template<template<typename> class STOT>
  struct RegStore : mu::variant<std::shared_ptr<const STOT<float>>,
                                std::shared_ptr<const STOT<double>>> {
    using Base = mu::variant<std::shared_ptr<const STOT<float>>,
                             std::shared_ptr<const STOT<double>>>;
    using Base::Base;
  };

  template<template<typename> class STOT>
  struct RegData : mu::variant<Empty, STOT<float>, STOT<double>> {
    using Base = mu::variant<Empty, STOT<float>, STOT<double>>;
    using Base::Base;
  };

  template<template<typename> class STOT>
  using DataStore = typename mu::variant<std::shared_ptr<const STOT<float>>,
                                         std::shared_ptr<const STOT<double>>,
                                         std::shared_ptr<const STOT<uint>>,
                                         std::shared_ptr<const STOT<uint8_t>>>;

  template<template<typename> class STOT>
  using Data = typename mu::variant<Empty,
                                    STOT<float>,
                                    STOT<double>,
                                    STOT<uint>,
                                    STOT<uint8_t>>;

  template<typename T>
  struct get_core {
    typedef typename std::remove_pointer<T>::type _tmp;
    typedef typename std::remove_reference<_tmp>::type __tmp;
    typedef typename std::remove_cv<__tmp>::type type;
  };

  // Must be symmetric and the product of InpStore and AnnStore variant types.
  template<template<typename, typename> class STOT>
  using SampleVec = mu::variant<std::vector<STOT<float, float>>,
                                std::vector<STOT<float, double>>,
                                std::vector<STOT<float, uint8_t>>,
                                std::vector<STOT<float, uint>>,

                                std::vector<STOT<double, float>>,
                                std::vector<STOT<double, double>>,
                                std::vector<STOT<double, uint8_t>>,
                                std::vector<STOT<double, uint>>,

                                std::vector<STOT<uint8_t, float>>,
                                std::vector<STOT<uint8_t, double>>,
                                std::vector<STOT<uint8_t, uint8_t>>,
                                std::vector<STOT<uint8_t, uint>>,

                                std::vector<STOT<uint, float>>,
                                std::vector<STOT<uint, double>>,
                                std::vector<STOT<uint, uint8_t>>,
                                std::vector<STOT<uint, uint>>>;

  template<template<typename, typename> class STOT>
  using SampleCVec = mu::variant<const std::vector<STOT<float, float>>,
                                 const std::vector<STOT<float, double>>,
                                 const std::vector<STOT<float, uint8_t>>,
                                 const std::vector<STOT<float, uint>>,
                                 
                                 const std::vector<STOT<double, float>>,
                                 const std::vector<STOT<double, double>>,
                                 const std::vector<STOT<double, uint8_t>>,
                                 const std::vector<STOT<double, uint>>,

                                 const std::vector<STOT<uint8_t, float>>,
                                 const std::vector<STOT<uint8_t, double>>,
                                 const std::vector<STOT<uint8_t, uint8_t>>,
                                 const std::vector<STOT<uint8_t, uint>>,

                                 const std::vector<STOT<uint, float>>,
                                 const std::vector<STOT<uint, double>>,
                                 const std::vector<STOT<uint, uint8_t>>,
                                 const std::vector<STOT<uint, uint>>>;
} // namespace fop
#endif // FORPY_UTIL_STORAGE_H_
