/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_DECIDERS_IDECIDER_H_
#define FORPY_DECIDERS_IDECIDER_H_

#include <cereal/access.hpp>

#include <unordered_map>
#include <mutex>
#include <utility>
#include <vector>
#include <limits>
#include <typeinfo>

#include "../global.h"
#include "../types.h"
#include "../data_providers/idataprovider.h"
#include "../threshold_optimizers/ithresholdoptimizer.h"

namespace forpy {
  /**
   * \brief Interface for the decider. It does the
   * optimization of the deciding classifier for each node and stores the
   * parameters.
   *
   * \param input_dtype The datatype of the data to classify.
   * \param annotation_dtype The datatype of the annotations.
   * \param feature_dtype The datatype in which features are calculated.
   *
   * \ingroup forpydecidersGroup
   */
  class IDecider {
   public:
    virtual ~IDecider();

    /**
     * \brief Optimizes a classifier for the given data and stores the params.
     *
     * This method must either set make_to_leaf to true or assure that at least
     * the minimum amount of samples per leaf is contained in each of
     * element_list_left and element_list_right. In the case that make_leaf
     * is true, the list pointers may even be returned uninitialized.
     *
     * If it is necessary to enforce additional growing constraints for the
     * tree, this is the right place (e.g. have a minimum number of samples
     * per node). The classifier manager can take these constraints into
     * account and may return make_to_leaf accordingly.
     *
     * \param node_id The node id of the node for which the classifier should
     *                be optimized. The parameters must be stored for the id.
     * \param node_depth The depth of the node in the tree. The root has
     *                depth 0.
     * \param min_samples_at_leaf The minimum number of samples at a leaf.
     *                This information can be used during the optimization.
     * \param element_id_list The ids of the samples arriving at this node.
     * \param data_provider The data provider from which the samples can be
     *                      loaded.
     * \2param make_to_leaf If the optimization does not find a sufficiently
     *                     good split that all growing criteria are fulfilled,
     *                     the node must be converted to a leaf by the caller.
     * \param element_list_left The ids of the elements the classifier
     *                          sends to the left. Only must be initialized if
     *                          make_to_leaf is false.
     * \param element_list_right The ids of the elements the classifier sends
     *                           to the right. Only must be initialized if
     *                           make_to_leaf is false.
     */
    virtual std::tuple<bool, elem_id_vec_t, elem_id_vec_t> make_node(
      const node_id_t &node_id,
      const uint &node_depth,
      const uint &min_samples_at_leaf,
      const elem_id_vec_t &element_id_list,
      const IDataProvider &data_provider) VIRTUAL_VOID;

    /**
     * \brief Makes a decision for a node with already optimized parameters.
     *
     * The classifier parameters must have been optimized for the node_id
     * before this method is called.
     *
     * \param node_id The node id of the node for which the decision should
     *                be made.
     * \param data The input data.
     * \return true, if the decision goes to left, false otherwise.
     */
    virtual bool decide(
      const node_id_t &node_id,
      const Data<MatCRef> &data,
      const std::function<void(void*)> &dptf = nullptr)
     const VIRTUAL(bool);

    /**
     * \brief Whether this classifier manager supports sample weights
     * during training.
     */
    virtual bool supports_weights() const VIRTUAL(bool);

    /** Gets the input dimension of the feature selection provider. */
    virtual size_t get_data_dim() const VIRTUAL(size_t);

    virtual void set_data_dim(const size_t &val) VIRTUAL_VOID;

    virtual std::shared_ptr<IThresholdOptimizer> get_threshopt() const
      VIRTUAL_PTR;
    
    virtual bool operator==(const IDecider &rhs) const VIRTUAL(bool);

   protected:
    /** 
     * \brief Empty constructor to allow inheritance though
     * DISALLOW_COPY_AND_ASSIGN is applied.
     */
    IDecider();

   private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive &, const uint &) {};

    DISALLOW_COPY_AND_ASSIGN(IDecider);
  };

}  // namespace forpy
#endif  // FORPY_DECIDERS_IDECIDER_H_
