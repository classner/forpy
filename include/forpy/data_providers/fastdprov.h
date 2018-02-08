/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_DATA_PROVIDERS_FASTDPROV_H_
#define FORPY_DATA_PROVIDERS_FASTDPROV_H_

#include <functional>
#include <vector>
#include "../global.h"
#include "../types.h"
#include "../util/checks.h"
#include "./idataprovider.h"

namespace forpy {
/**
 * \brief Use the provided data plain throughout the training.
 *
 * \ingroup forpydata_providersGroup
 */
class FastDProv : public IDataProvider {
 public:
  /**
   * \brief Standard constructor. Takes shared ownership on data and
   * annotations.
   *
   * \param data Matrix in column major order with shape (n_samples x
   *    n_features).
   * \param annotations Matrix in row major order with shape (n_samples x
   *    n_annots).
   * \param weights_store
   *    Storage for sample weights. If nullptr, weights are ignored.
   */
  FastDProv(const DataStore<Mat> &data, const DataStore<Mat> &annotations,
            const std::shared_ptr<std::vector<float> const> &weights_store);

  /**
   * \brief Non-ownership requiring constructor.
   *
   * It's your job to keep the data alive as long as this object exists. I'm
   * looking at you, Python interface!
   *
   * \param data Matrix in column major order with shape (n_samples x
   *    n_features).
   * \param annotations Matrix in row major order with shape (n_samples x
   *    n_annots).
   * \param weights_store Vector with shape (n_samples) with positive weights
   *    for each sample.
   */
  FastDProv(const Data<MatCRef> &data, const Data<MatCRef> &annotations,
            const std::shared_ptr<std::vector<float> const> &weights_store);

  //@{
  /// forpy::IDataProvider function implementation.
  inline std::vector<id_t> &get_initial_sample_list() { return *training_ids; };

  inline size_t get_n_samples() const { return training_ids->size(); };

  Data<VecCMap> get_feature(const size_t & /*feat_idx*/) const;

  inline Data<MatCRef> get_annotations() const { return annotations; };

  inline void set_annotations(const DataStore<Mat> &new_annotation_store) {
    annotation_store = new_annotation_store;
    annotation_store.match(
        [&](const auto &new_annotations) { annotations = *new_annotations; });
  };

  inline std::shared_ptr<const std::vector<float>> get_weights() const {
    return weights_store;
  }

  std::vector<std::shared_ptr<IDataProvider>> create_tree_providers(
      usage_map_t &usage_map);
  //@}

  inline friend std::ostream &operator<<(std::ostream &stream,
                                         const FastDProv &self) {
    stream << "forpy::FastDProv[" << self.get_n_samples() << " samples, "
           << self.get_feat_vec_dim() << " -> " << self.get_annot_vec_dim()
           << "]";
    return stream;
  };
  bool operator==(const IDataProvider &rhs) const;

 private:
  /** \brief Constructor for deserialization. */
  inline FastDProv(){};

  /**
   * \brief Constructor for creating a 'proxy' data provider for trees.
   *
   * This just maps the data of an existing data provider.
   */
  FastDProv(const Data<MatCRef> &data, const Data<MatCRef> &annotations,
            const std::shared_ptr<std::vector<float> const> &weights_store,
            std::shared_ptr<std::vector<id_t>> &training_ids);

  /** \brief Perform all necessary checks before constructing an instance. */
  void checks(const Data<MatCRef> &data,
              const Data<MatCRef> &annotations) const;

  /** \brief Perform the initialization once the FastDProv::data and
   * FastDProv::annotations have been set. */
  void init_from_arrays();

  using IDataProvider::annot_vec_dim;
  using IDataProvider::feat_vec_dim;
  /// Data storage. If ownership of the data can't be shared, it is copied here.
  DataStore<Mat> data_store;
  /// Data storage. If ownership of the annotations can't be shared, they are
  /// copied here.
  DataStore<Mat> annotation_store;
  /// A reference to the data.
  Data<MatCRef> data;
  /// A reference to the annotations.
  Data<MatCRef> annotations;
  /// Weight storage.
  std::shared_ptr<std::vector<float> const> weights_store;
  /// A vector of the annotation indices that should be used out of the full
  /// data.
  std::shared_ptr<std::vector<id_t>> training_ids;
};
}  // namespace forpy
#endif  // FORPY_DATA_PROVIDERS_FASTDPROV_H_
