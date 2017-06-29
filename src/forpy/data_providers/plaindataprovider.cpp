#include <forpy/data_providers/plaindataprovider.h>

namespace forpy {

  PlainDataProvider::PlainDataProvider()
    : n_samples(0), column_wise(false), step(0) {};

  PlainDataProvider::PlainDataProvider(const DataStore<Mat> &data_store,
                                       const DataStore<Mat> &annotation_store) :
    data_store(data_store),
    annotation_store(annotation_store) {
    data_store.match(
      [](const Empty &) { throw EmptyException(); },
      [&, this](const auto &data) {
        annotation_store.match(
          [](const Empty &) { throw EmptyException(); },
          [&, this](const auto &annotations) {
            PlainDataProvider(*data, *annotations);
          });
      });
  };

  PlainDataProvider::PlainDataProvider(const Data<MatCRef> &data,
                                       const Data<MatCRef> &annotations,
                                       const std::shared_ptr<SampleVec<Sample>> &samples,
                                       const elem_id_vec_t &training_ids) :
    data(data),
    annotations(annotations),
    samples(samples),
    training_ids(training_ids),
    column_wise(false),
    step(1) {
    data.match(
      [](const Empty &) { throw EmptyException(); },
      [&, this](const auto &data) {
        annotations.match(
          [](const Empty &) { throw EmptyException(); },
          [&, this](const auto &annotations) {
            this->checks(data, annotations);
            n_samples = data.rows();
            feat_vec_dim = data.cols();
            annot_vec_dim = annotations.cols();
          });
      });
  }

  PlainDataProvider::PlainDataProvider(const Data<MatCRef> &data,
                                       const Data<MatCRef> &annotations) :
    data(data),
    annotations(annotations) {
    checks(data, annotations);
    init_from_arrays();
  };

  void PlainDataProvider::checks(const Data<MatCRef> &data,
                                 const Data<MatCRef> &annotations) const {
    size_t data_rows;
    data.match(
        [](const Empty &) { throw EmptyException(); },
        [&](const auto &data) {
          if (data.cols() == 0)
            throw Forpy_Exception("Tried to create a data provider for "
                                  "feature dimension 0.");
          if (data.rows() == 0)
            throw Forpy_Exception("Tried to create a data provider for "
                                  "0 samples.");
          data_rows = data.rows();
        });
    annotations.match(
        [](const Empty &) { throw EmptyException(); },
        [&](const auto &annotations) {
          if (data_rows != static_cast<size_t>(annotations.rows()))
            throw Forpy_Exception("Data and annotation counts don't match!");
          if (annotations.cols() == 0)
            throw Forpy_Exception("Tried to create a data provider for "
                                  "annotation dimension 0!");
        });
  };

  void PlainDataProvider::init_from_arrays() {
    this->column_wise = false;
    this->step = 1;
    this->training_ids.clear();
    samples = std::make_shared<SampleVec<Sample>>();
    data.match([&, this](const auto &data) {
        annotations.match([&, this](const auto &annotations) {
            typedef typename get_core<decltype(data.data())>::type IT;
            typedef typename get_core<decltype(annotations.data())>::type AT;
            samples->set<std::vector<Sample<IT, AT>>>();
            auto &sample_vec = samples->get_unchecked<
              std::vector<Sample<IT, AT>>>();
            this->n_samples = data.rows();
            this->feat_vec_dim = data.cols();
            this->annot_vec_dim = annotations.cols();
            std::shared_ptr<const Mat<IT>> datap;
            if (data_store.is<std::shared_ptr<const Mat<IT>>>()) {
              datap = data_store.get_unchecked<std::shared_ptr<const Mat<IT>>>();
            }
            std::shared_ptr<const Mat<AT>> annotationp;
            if (annotation_store.is<std::shared_ptr<const Mat<AT>>>()) {
              annotationp = annotation_store.get_unchecked<std::shared_ptr<const Mat<AT>>>();
            }
            // The annotations may have a stride. This is known unfortunately
            // only in the python converting Eigen::Map and I didn't find a
            // better way to retrieve it:
            ptrdiff_t stride;
            if (annotations.cols() > 1) {
              const auto block_tmp = annotations.block(0, 1, 1, 1);
              stride = block_tmp.data() - annotations.data();
            } else {
              stride = 1;
            }
            // Only the outer dimension may have a stride, so from here on we're
            // safe. Check that for the data the stride is 1.
            ptrdiff_t datastride;
            if (data.cols() > 1) {
              const auto block_tmp = data.block(0, 1, 1, 1);
              datastride = block_tmp.data() - data.data();
            } else {
              datastride = 1;
            }
            for (size_t i = 0; i < n_samples; ++i) {
              sample_vec.emplace_back(
                  VecCMap<IT>(data.block(i, 0, 1, 1).data(),
                              1, data.cols(),
                              Eigen::InnerStride<>(datastride)),
                  VecCMap<AT>(annotations.block(i, 0, 1, 1).data(),
                              1, annotations.cols(),
                              Eigen::InnerStride<>(stride)),
                  1.f,
                  datap,
                  annotationp);
              training_ids.push_back(i);
            }
          },
          [](const Empty &) { throw EmptyException(); });
      },
      [](const Empty &) { throw EmptyException(); });
  }

  std::vector<std::shared_ptr<IDataProvider>> PlainDataProvider::create_tree_providers(
      const usage_map_t &usage_map) {
    std::vector<std::shared_ptr<IDataProvider>> retvec;
    for (size_t i = 0; i < usage_map.size(); ++i) {
      if (! check_elem_ids_ok(n_samples, usage_map[i])) {
        throw Forpy_Exception("Wrong sample usage map with a too high element "
                              "ID!");
      }
      if (!check_elem_ids_ok(n_samples, usage_map[i])) {
        throw Forpy_Exception("Wrong sample usage map with a too high element "
                              "ID!");
      }
      retvec.emplace_back(new PlainDataProvider(data,
                                                annotations,
                                                samples,
                                                usage_map[i]));
    }
    return retvec;
  }

  void PlainDataProvider::optimize_set_for_node(
                             const node_id_t &node_id,
                             const uint &depth,
                             const node_predf &node_predictor,
                             const elem_id_vec_t &element_list) {};

  const elem_id_vec_t &PlainDataProvider::get_initial_sample_list() const  {
    return training_ids;
  };

  const SampleVec<Sample> &PlainDataProvider::get_samples() const {
    return *samples;
  };

  size_t PlainDataProvider::get_n_samples() const {
    return n_samples;
  };

  /** Gets whether the data is stored column wise. So far, always returns false. */
  bool PlainDataProvider::get_column_wise() const {
    return column_wise;
  };

  /** Does nothing. */
  void PlainDataProvider::track_child_nodes(node_id_t node_id,
                                            node_id_t left_id,
                                            node_id_t right_id) {};

  bool PlainDataProvider::operator==(const IDataProvider &rhs) const {
    const auto *rhs_c = dynamic_cast<PlainDataProvider const *>(&rhs);
    if (rhs_c == nullptr) {
      return false;
    } else {
      bool eq_fvd = feat_vec_dim == rhs_c->feat_vec_dim;
      bool eq_avd = annot_vec_dim == rhs_c->annot_vec_dim;
      bool eq_data = mu::apply_visitor(MatEqVis(), data, rhs_c->data);
      bool eq_annotations = mu::apply_visitor(MatEqVis(),
                                              annotations,
                                              rhs_c->annotations);
      bool eq_samples = *samples == *(rhs_c->samples);
      bool eq_ids = training_ids == rhs_c->training_ids;
      bool eq_n_samples = n_samples == rhs_c->n_samples;
      bool eq_step = step == rhs_c->step;
      bool eq_cw = column_wise == rhs_c->column_wise;
      return eq_fvd && eq_avd && eq_data && eq_annotations && eq_samples &&
        eq_ids && eq_n_samples && eq_step && eq_cw;
    }
  }
}
