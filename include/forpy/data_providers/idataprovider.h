/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_DATA_PROVIDERS_IDATAPROVIDER_H_
#define FORPY_DATA_PROVIDERS_IDATAPROVIDER_H_

#include "../global.h"
#include "../util/serialization/basics.h"

#include <functional>
#include <vector>

#include "../types.h"
#include "../util/storage.h"

namespace forpy {

/**
 * \brief A data provider for the training of one tree.
 *
 * \ingroup forpydata_providersGroup
 */
class IDataProvider {
 public:
  inline virtual ~IDataProvider(){};

  /**
   * \brief Get a sample id list of samples for the root node.
   *
   * Not all samples have to be used for trees, so this list can be sparse
   * and/or unordered.
   */
  virtual std::vector<id_t> &get_initial_sample_list()
      VIRTUAL(std::vector<id_t>);

  /** \brief Get the number of samples. */
  virtual size_t get_n_samples() const VIRTUAL_VOID;

  /**
   * \brief Get the data for one feature from all samples, contiguously in
   * memory (stride 1).
   */
  virtual Data<VecCMap> get_feature(const size_t & /*feat_idx*/) const {
    throw ForpyException("`get_feature` not implemented!");
  };

  /**
   * \brief Get the full annotation data (must have inner stride 1).
   */
  virtual Data<MatCRef> get_annotations() const {
    throw ForpyException("`get_annotations` not implemented!");
  };

  /**
   * \brief Replace the annotations.
   *
   * Do not use this during ongoing training since there is no currently
   * implemented mechanism to update the pointers to annotations in the various
   * processing threads.
   *
   * The method is currently used to store reduced class number lists for
   * classification (see \ref forpy::ClassificationOpt::check_annotations).
   */
  virtual void set_annotations(const DataStore<Mat> &new_annotations)
      VIRTUAL_VOID;

  /**
   * \brief Get a pointer to the sample weights.
   *
   * Can be a nullptr, in that case no weights were provided.
   */
  virtual std::shared_ptr<const std::vector<float>> get_weights() const
      VIRTUAL_PTR;

  /**
   * \brief Get the feature vector dimension.
   */
  inline size_t get_feat_vec_dim() const { return feat_vec_dim; };

  /**
   * \brief Get the annotation vector dimension.
   */
  inline size_t get_annot_vec_dim() const { return annot_vec_dim; };

  /**
   * \brief Creates the data providers for each tree from the specified
   *        usage map.
   *
   * This brings some tricky issues concerning data ownership in: since
   * internally this method will construct other data providers not owning
   * their data, this must be communicated to the user who constructed this
   * object non-owning data. Hence, the resulting data providers must keep this
   * object `alive` if this method is called from Python, and all Python users
   * of a data provider must keep it alive as long as they use it.
   *
   * \param usage_map A vector with sample_id lists.
   */
  virtual std::vector<std::shared_ptr<IDataProvider>> create_tree_providers(
      usage_map_t &usage_map)
      VIRTUAL(std::vector<std::shared_ptr<IDataProvider>>);

  virtual bool operator==(const IDataProvider &rhs) const VIRTUAL(bool);

 protected:
  /**
   * \brief Standard constructor to use for inheriting classes.
   */
  explicit IDataProvider(const size_t &feature_dimension,
                         const size_t &annotation_dimension);

  /**
   * \brief Constructor solely for deserialization.
   */
  // cppcheck-suppress uninitVar
  inline IDataProvider(){};

  /** \brief The dimension of one feature vector. */
  size_t feat_vec_dim;
  /** \brief The dimension of one annotation vector. */
  size_t annot_vec_dim;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar, const uint &) {
    ar(CEREAL_NVP(feat_vec_dim), CEREAL_NVP(annot_vec_dim));
  }
};
}  // namespace forpy
#endif  // FORPY_DATA_PROVIDERS_IDATAPROVIDER_H_
