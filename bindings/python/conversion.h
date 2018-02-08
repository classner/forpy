#pragma once
#ifndef FORPY_BINDINGS_PYTHON_CONVERSION_H_
#define FORPY_BINDINGS_PYTHON_CONVERSION_H_

#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <mapbox/variant.hpp>
#include <forpy/util/storage.h>

/// This header file must be included in all compilation units requiring
/// conversions (see https://github.com/pybind/pybind11/issues/903).

namespace py = pybind11;
namespace mu = mapbox::util;

namespace pybind11 { namespace detail {

    template <typename... Ts>
    struct type_caster<mapbox::util::variant<Ts...>> :
      variant_caster<mapbox::util::variant<Ts...>> {};

    // Specifies the function used to visit the variant -- `apply_visitor` instead of `visit`
    template <>
    struct visit_helper<mapbox::util::variant> {
      template <typename... Args>
      static auto call(Args &&...args)
        -> decltype(mapbox::util::apply_visitor(std::forward<Args>(args)...)) {
        return mapbox::util::apply_visitor(std::forward<Args>(args)...);
      }
    };

    //////////////////////////////
    /// Specializations for the pointer variants.
    /// Visit a variant and cast any found type to Python
    struct ptr_variant_caster_visitor {
      return_value_policy policy;
      handle parent;
      
      template <typename IT>
        handle operator()(const std::shared_ptr< IT >&src) const {
        return make_caster<IT>::cast(std::forward<IT>(*src), policy, parent);
      }
    };

    /// Helper class which abstracts away variant's `visit` function.
    /// `std::variant` and similar `namespace::variant` types which provide a
    /// `namespace::visit()` function are handled here automatically using
    /// argument-dependent lookup. Users can provide specializations for other
    /// variant-like classes, e.g. `boost::variant` and
    /// `boost::apply_visitor`.
    template <>
    struct visit_helper<forpy::ptr_variant> {
      template <typename... Args>
      static auto call(Args &&...args) -> decltype(mu::apply_visitor(std::forward<Args>(args)...)) {
        return mu::apply_visitor(std::forward<Args>(args)...);
      }
    };

    /// Generic variant caster
    template <typename Variant> struct ptr_variant_caster;
    
    template <typename E, template<typename> class F>
    struct ptrextract {};
    
    template <typename E>
    struct ptrextract<std::shared_ptr<E>, std::shared_ptr> {
      typedef E value;
    };
    
    template <template<typename...> class V, typename... Ts>
    struct ptr_variant_caster<V<Ts...>> {
      static_assert(sizeof...(Ts) > 0, "Variant must consist of at least one alternative.");

      /*template <typename U, typename... Us>
        bool load_alternative(handle src, bool convert, type_list<U, Us...>) {
        std::cerr << "in extract" << std::endl;
        
        typedef typename ptrextract<U, std::shared_ptr>::value inner;
        //auto caster = make_caster<inner>();
      //if (caster.load(src, convert)) {
      //value = std::make_shared<inner>(cast_op<inner>(caster));
      return true;
      //}
      //return load_alternative(src, convert, type_list<Us...>{});
      }
      
      bool load_alternative(handle, bool, type_list<>) { return false; }
      
      bool load(handle src, bool convert) {
      // Do a first pass without conversions to improve constructor resolution.
        // E.g. `py::int_(1).cast<variant<double, int>>()` needs to fill the `int`
        // slot of the variant. Without two-pass loading `double` would be filled
        // because it appears first and a conversion is possible.
        std::cerr << "in load" << std::endl;
        //if (convert && load_alternative(src, false, type_list<Ts...>{}))
        //     return true;
        // return load_alternative(src, convert, type_list<Ts...>{});
        return true;
      }*/

      template <typename Variant>
      static handle cast(Variant &&src, return_value_policy policy, handle parent) {
        return visit_helper<V>::call(ptr_variant_caster_visitor{policy, parent},
                                     std::forward<Variant>(src));
      }
      
      using Type = V<Ts...>;
      PYBIND11_TYPE_CASTER(Type, _("PtrUnion"));
    };

    template <typename... Ts>
    struct type_caster<forpy::ptr_variant<Ts...>> :
    ptr_variant_caster<forpy::ptr_variant<Ts...>> { };
  }
} // namespace pybind11::detail
#endif // FORPY_BINDINGS_PYTHON_VARIANT_RESULTS_H_
