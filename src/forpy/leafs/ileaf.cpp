#include <forpy/leafs/ileaf.h>
#include <mapbox/variant_io.hpp>

namespace forpy {

  ILeaf::ILeaf() {};

  ILeaf::~ILeaf() {};

  bool ILeaf::is_compatible_with(const IDataProvider &data_provider) {
    return true;
  };

  Data<Mat> ILeaf::get_result(
      const node_id_t &node_id,
      const Data<MatCRef> &data,
      const std::function<void(void*)> &dptf) const {
    Data<Mat> ret;
    ret.set<Mat<float>>(Mat<float>::Zero(1, get_result_columns(1)));
    Data<MatRef> dref = MatRef<float>(ret.get_unchecked<Mat<float>>());
    get_result(node_id,
               dref,
               data,
               dptf);
    return ret;
  };

  Data<Mat> ILeaf::get_result(const std::vector<Data<Mat>> &leaf_results,
                              const Vec<float> &weights) const {
    Data<Mat> ret;
    ret.set<Mat<float>>(Mat<float>::Zero(1,
                                         get_result_columns(leaf_results.size())));
    Data<MatRef> dref = MatRef<float>(ret.get_unchecked<Mat<float>>());
    get_result(leaf_results, dref, weights);
    return ret;
  };

  bool ILeaf::needs_data() const { return false; };

} // namespace forpy
