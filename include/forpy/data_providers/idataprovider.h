/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_DATA_PROVIDERS_IDATAPROVIDER_H_
#define FORPY_DATA_PROVIDERS_IDATAPROVIDER_H_

#include <cereal/access.hpp>

#include <functional>
#include <vector>

#include "../global.h"
#include "../types.h"
#include "./sample.h"
#include "../util/storage.h"

namespace forpy {
  /**
   * \brief The data provider for the training of one tree.
   *
   * \ingroup forpydata_providersGroup
   *
   * Data providers work with \ref Samples and sample ids. They are allowed to
   * add additional samples during training on the fly. The method
   * \ref optimize_set_for_node is called before a classifier is fitted for
   * a decision node, so that the data provider can alter the set of samples
   * that are used.
   */
  class IDataProvider {
   public:
    virtual ~IDataProvider();

    /**
      * \brief Optimizes a sample set.
      *
      * This method is called before classifier optimization at every node.
      *
      * \param node_id The current nodes id.
      * \param depth The node's depth.
      * \param node_predictor A function that takes a sample and returns the
      *                       node id of the node in the current tree at which
      *                       the sample would arrive.
      * \param element_list The id list of all elements that arrive at the
      *                     node.
      */
    virtual void optimize_set_for_node(
      const node_id_t &node_id,
      const uint &depth,
      const node_predf &node_predictor,
      const elem_id_vec_t &element_list) VIRTUAL_VOID;

    /**
      * \brief Get a sample id list of samples for the root node.
      */
    virtual const elem_id_vec_t &get_initial_sample_list() const
      VIRTUAL(elem_id_vec_t);

    /**
      * \brief Get all samples.
      */
    virtual const SampleVec<Sample> &get_samples() const
      VIRTUAL(SampleVec<Sample>);

    /**
     * \brief Some data providers might need this method to do cleanup actions
     * for efficient sample data management.
     */
    virtual void track_child_nodes(node_id_t node_id,
      node_id_t left_id, node_id_t right_id) VIRTUAL_VOID;

    /**
     * \brief Returns one feature vector dimension.
     */
    size_t get_feat_vec_dim() const;

    /**
     * \brief Returns one annotation vector dimension.
     */
    size_t get_annot_vec_dim() const;

    /** 
     * \brief Returns a coordinate transformation function for the dimension
     * selectors.
     *
     * The transformation is applied to the positions before the data is
     * extracted and passed to the IFeatureCalculator. If not overridden,
     * returns the nullptr (no transformation).
     */
    virtual std::function<void(void*)> get_decision_transf_func(
                                                         const element_id_t &)
      const;

    /**
     * \brief Gives the data provider the opportunity to load all samples
     * to construct a leaf.
     *
     * This method is especially important for the subsampling data providers:
     * only a certain amount of samples is used to find a good split, but
     * if the result is a leaf node, ALL samples should be used to create
     * a good leaf estimate.
     */
    virtual void load_samples_for_leaf(
        const node_id_t &node_id,
        const node_predf &node_predictor,
        elem_id_vec_t *element_list);

    /**
     * \brief Creates the data providers for each tree from the specified
     *        usage map.
     *
     * \param n How many data providers (trees) are needed.
     * \param usage_map A vector with a pair of sample_id lists. Each vector
     *    element is for one tree. It contains a pair of training ids and
     *    validation ids of the samples to use.
     */
    virtual std::vector<std::shared_ptr<IDataProvider>> create_tree_providers(
        const usage_map_t &usage_map)
      VIRTUAL(std::vector<std::shared_ptr<IDataProvider>>);

    virtual bool operator==(const IDataProvider &rhs) const VIRTUAL(bool);

   protected:
    explicit IDataProvider(const size_t &feature_dimension,
                           const size_t &annotation_dimension);

    // cppcheck-suppress uninitVar
    IDataProvider();

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar, const uint &) {
      ar(CEREAL_NVP(feat_vec_dim),
         CEREAL_NVP(annot_vec_dim));
    }

    /** \brief The dimension of one sample vector. */
    size_t feat_vec_dim;
    size_t annot_vec_dim;
  };
}  // namespace forpy
#endif  // FORPY_DATA_PROVIDERS_IDATAPROVIDER_H_
