/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_DATA_PROVIDERS_PLAINDATAPROVIDER_H_
#define FORPY_DATA_PROVIDERS_PLAINDATAPROVIDER_H_

#include <vector>
#include <functional>
#include "../global.h"
#include "../types.h"
#include "../util/checks.h"
#include "../util/macros.h"
#include "./idataprovider.h"


namespace forpy {
  /**
   * \brief Uses the provided data plain throughout the training.
   *
   * \ingroup forpydata_providersGroup
   */
  class PlainDataProvider : public IDataProvider {
   public:

    /**
     * Standard constructor. Takes shared ownership on data and annotations.
     */
    PlainDataProvider(const DataStore<Mat> &data,
                      const DataStore<Mat> &annotations);

    /**
     * Non-ownership requiring constructor. It's your job to keep the data
     * alive as long as this object exists. I'm looking at you, Python
     * interface!
     */
    PlainDataProvider(const Data<MatCRef> &data,
                      const Data<MatCRef> &annotations);

    // TODO: create a shorthand way for trainings where weights are not used
    // and data is available in the matrices to access them directly and avoid
    // double referencing.

    /** \brief Does nothing. */
    void optimize_set_for_node(
      const node_id_t &node_id,
      const uint &depth,
      const node_predf &node_predictor,
      const elem_id_vec_t &element_list);

    /** \brief Returns a list of all sample ids. */
    const elem_id_vec_t &get_initial_sample_list() const;

    /** Gets the sample vector. */
    const SampleVec<Sample> &get_samples() const;

    /** Gets the number of samples. */
    size_t get_n_samples() const;

    /** Gets whether the data is stored column wise. So far, always returns false. */
    bool get_column_wise() const;

    /** Does nothing. */
    void track_child_nodes(node_id_t node_id,
                           node_id_t left_id,
                           node_id_t right_id);

    bool operator==(const IDataProvider &rhs) const;

    /**
     * \brief Creates the data providers for each tree from the specified
     *        usage map.
     *
     * This brings some tricky issues concerning data ownership in: since
     * internally this method will construct other data providers not owning
     * their data, this must be communicated to user who constructed this object
     * non-owning data. Hence, the resulting data providers must keep this
     * object `alive` if this method is called from Python, and all Python users
     * of a data provider must keep it alive as long as they use it.
     *
     * \param usage_map A vector with a pair of sample_id lists. Each vector
     *    element is for one tree. It contains a pair of training ids and
     *    validation ids of the samples to use.
     */
    std::vector<std::shared_ptr<IDataProvider>> create_tree_providers(
        const usage_map_t &usage_map);

   private:
    PlainDataProvider();

    PlainDataProvider(const Data<MatCRef> &data,
                      const Data<MatCRef> &annotations,
                      const std::shared_ptr<SampleVec<Sample>> &samples,
                      const elem_id_vec_t &training_ids);


    void checks(const Data<MatCRef> &data,
                const Data<MatCRef> &annotations) const;

    void init_from_arrays();

    using IDataProvider::feat_vec_dim;
    using IDataProvider::annot_vec_dim;
    DataStore<Mat> data_store;
    DataStore<Mat> annotation_store;
    Data<MatCRef> data;
    Data<MatCRef> annotations;
    std::shared_ptr<SampleVec<Sample>> samples;
    elem_id_vec_t training_ids;
    size_t n_samples;
    bool column_wise;
    size_t step;
  };
}  // namespace forpy
#endif  // FORPY_DATA_PROVIDERS_PLAINDATAPROVIDER_H_
