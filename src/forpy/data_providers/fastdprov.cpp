#include <forpy/data_providers/fastdprov.h>

namespace forpy {

FastDProv::FastDProv(
    const DataStore<Mat> &data_store, const DataStore<Mat> &annotation_store,
    const std::shared_ptr<std::vector<float> const> &weights_store)
    : data_store(data_store),
      annotation_store(annotation_store),
      weights_store(weights_store) {
  if (weights_store != nullptr && weights_store->size() == 0)
    this->weights_store = nullptr;
  data_store.match([](const Empty &) { throw EmptyException(); },
                   [&, this](const auto &data) {
                     annotation_store.match(
                         [](const Empty &) { throw EmptyException(); },
                         [&, this](const auto &annotations) {
                           this->data = *data;
                           this->annotations = *annotations;
                           checks(*data, *annotations);
                           init_from_arrays();
                         });
                   });
};

FastDProv::FastDProv(
    const Data<MatCRef> &data, const Data<MatCRef> &annotations,
    const std::shared_ptr<std::vector<float> const> &weights_store)
    : data(data), annotations(annotations), weights_store(weights_store) {
  if (weights_store != nullptr && weights_store->size() == 0)
    this->weights_store = nullptr;
  checks(data, annotations);
  init_from_arrays();
};

FastDProv::FastDProv(
    const Data<MatCRef> &data, const Data<MatCRef> &annotations,
    const std::shared_ptr<std::vector<float> const> &weights_store,
    std::shared_ptr<std::vector<id_t>> &training_ids)
    : data(data),
      annotations(annotations),
      weights_store(weights_store),
      training_ids(training_ids) {
  if (weights_store != nullptr && weights_store->size() == 0)
    this->weights_store = nullptr;
  checks(data, annotations);
  data.match(
      [&, this](const auto &data) {
        annotations.match(
            [&, this](const auto &annotations) {
              this->feat_vec_dim = data.rows();
              this->annot_vec_dim = annotations.cols();
            },
            [](const Empty &) { throw EmptyException(); });
      },
      [](const Empty &) { throw EmptyException(); });
  VLOG(22) << "Created FastDProv for " << get_n_samples() << " samples with "
           << feat_vec_dim << " features and " << annot_vec_dim
           << " annotations.";
}

void FastDProv::checks(const Data<MatCRef> &data,
                       const Data<MatCRef> &annotations) const {
  DLOG(INFO) << "Running DProv checks...";
  DLOG(INFO) << weights_store.get();
  size_t data_cols;
  data.match(
      [](const Empty &) { throw EmptyException(); },
      [&](const auto &data) {
        if (data.cols() == 0)
          throw ForpyException(
              "Tried to create a data provider for feature dimension 0.");
        if (data.rows() == 0)
          throw ForpyException(
              "Tried to create a data provider for 0 samples.");
        data_cols = data.cols();
        if (data.innerStride() != 1)
          throw ForpyException("The data array has an inner stride != 1 (" +
                               std::to_string(data.innerStride()) +
                               ")! A stride of 1 is required.");
      });
  annotations.match(
      [](const Empty &) { throw EmptyException(); },
      [&](const auto &annotations) {
        if (data_cols != static_cast<size_t>(annotations.rows()))
          throw ForpyException("Data and annotation counts don't match!");
        if (annotations.cols() == 0)
          throw ForpyException(
              "Tried to create a data provider for annotation dimension 0!");
        if (annotations.innerStride() != 1)
          throw ForpyException(
              "The annotation array has an inner stride != 1 (" +
              std::to_string(annotations.innerStride()) +
              ")! A stride of 1 is required!");
        if (weights_store != nullptr) {
          if (weights_store->size() != static_cast<size_t>(annotations.rows()))
            throw ForpyException(
                "Non-matching number of weights (" +
                std::to_string(annotations.rows()) + " samples and " +
                std::to_string(weights_store->size()) + " weights).");
          for (const auto &weight : *weights_store)
            if (weight < 0.f)
              throw ForpyException("Negative weight detected (" +
                                   std::to_string(weight) + ")!");
        }
      });
};

void FastDProv::init_from_arrays() {
  /* This assumes that checks have been performed! */
  this->training_ids = std::make_shared<std::vector<id_t>>();
  data.match(
      [&, this](const auto &data) {
        annotations.match(
            [&, this](const auto &annotations) {
              this->feat_vec_dim = data.rows();
              this->annot_vec_dim = annotations.cols();
              training_ids->resize(data.cols());
              std::iota(training_ids->begin(), training_ids->end(), 0);
            },
            [](const Empty &) { throw EmptyException(); });
      },
      [](const Empty &) { throw EmptyException(); });
  VLOG(22) << "Created FastDProv for " << get_n_samples() << " samples with "
           << feat_vec_dim << " features and " << annot_vec_dim
           << " annotations.";
}

std::vector<std::shared_ptr<IDataProvider>> FastDProv::create_tree_providers(
    usage_map_t &usage_map) {
  std::vector<std::shared_ptr<IDataProvider>> retvec;
  for (size_t i = 0; i < usage_map.size(); ++i) {
    if (!check_elem_ids_ok(get_n_samples(), *usage_map[i].first)) {
      throw ForpyException(
          "Wrong sample usage map with a too high element "
          "ID!");
    }
    if (!check_elem_ids_ok(get_n_samples(), *usage_map[i].first)) {
      throw ForpyException(
          "Wrong sample usage map with a too high element "
          "ID!");
    }
    retvec.emplace_back(new FastDProv(data, annotations, usage_map[i].second,
                                      usage_map[i].first));
  }
  return retvec;
}

Data<VecCMap> FastDProv::get_feature(const size_t &feat_idx) const {
  Data<VecCMap> ret_dta;
  data.match(
      [&, this](const auto &data) {
        typedef typename get_core<decltype(data.data())>::type IT;
        ret_dta.set<VecCMap<IT>>(data.data() + feat_idx * data.outerStride(),
                                 data.cols(),
                                 Eigen::InnerStride<Eigen::Dynamic>(1));
      },
      [](const Empty &) { throw EmptyException(); });
  return ret_dta;
}

bool FastDProv::operator==(const IDataProvider &rhs) const {
  const auto *rhs_c = dynamic_cast<FastDProv const *>(&rhs);
  if (rhs_c == nullptr) {
    return false;
  } else {
    bool eq_fvd = feat_vec_dim == rhs_c->feat_vec_dim;
    bool eq_avd = annot_vec_dim == rhs_c->annot_vec_dim;
    bool eq_data = mu::apply_visitor(MatEqVis(), data, rhs_c->data);
    bool eq_annotations =
        mu::apply_visitor(MatEqVis(), annotations, rhs_c->annotations);
    bool eq_ids = *training_ids == *rhs_c->training_ids;
    return eq_fvd && eq_avd && eq_data && eq_annotations && eq_ids;
  }
}
}  // namespace forpy
