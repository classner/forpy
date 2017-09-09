/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_LEAFS_ILEAF_H_
#define FORPY_LEAFS_ILEAF_H_

#include <cereal/access.hpp>

#include <vector>

#include "../data_providers/idataprovider.h"
#include "../threshold_optimizers/ithresholdoptimizer.h"
#include "../global.h"
#include "../types.h"


namespace forpy {
  /**
   * \brief Stores and returns leaf values, and combines them to forest
   * results.
   *
   * \ingroup forpyleafsGroup
   */
  class ILeaf {
   public:
    virtual ~ILeaf();

    /**
     * \brief Checks compatibility with a certain \ref IDataProvider.
     *
     * This method is guaranteed to be called at the beginning of a training.
     */
    virtual bool is_compatible_with(const IDataProvider &data_provider);

    virtual bool is_compatible_with(const IThresholdOptimizer &threshopt)
      VIRTUAL(bool);

    virtual bool needs_data() const;

    /**
     * \brief Creates a leaf with the specified node_id and data.
     */
    virtual void make_leaf(
      const node_id_t &node_id,
      const elem_id_vec_t &element_list,
      const IDataProvider &data_provider) VIRTUAL_VOID;

    /** Gets the number of summary dimensions per sample. */
    virtual size_t get_result_columns(const size_t &n_trees=1) const
      VIRTUAL(size_t);

    /** Get the result data type (a 0x0 mat within in appropriate variant). */
    virtual Data<Mat> get_result_type() const VIRTUAL(Data<Mat>);

    /**
     * \brief Gets the leaf data for the leaf with the given id.
     */
    virtual Data<Mat> get_result(
      const node_id_t &node_id,
      const Data<MatCRef> &data=Data<MatCRef>(),
      const std::function<void(void*)> &dptf = nullptr) const;

    virtual void get_result(
      const node_id_t &node_id,
      Data<MatRef> &target,
      const Data<MatCRef> &data=Data<MatCRef>(),
      const std::function<void(void*)> &dptf = nullptr) const VIRTUAL_VOID;

    virtual Data<Mat> get_result(
        const std::vector<Data<Mat>> &leaf_results,
        const Vec<float> &weights=Vec<float>()) const;

    virtual void get_result(
      const std::vector<Data<Mat>> &leaf_results,
      Data<MatRef> &target,
      const Vec<float> &weights=Vec<float>()) const VIRTUAL_VOID;

    virtual bool operator==(const ILeaf &rhs) const VIRTUAL(bool);

   protected:
    ILeaf();

   private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive &, const uint &) {};

    DISALLOW_COPY_AND_ASSIGN(ILeaf);
  };
};  // namespace forpy
#endif  // FORPY_LEAFS_ILEAF_H_
