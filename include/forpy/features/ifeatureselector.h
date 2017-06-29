/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_FEATURES_IFEATURESELECTOR_H_
#define FORPY_FEATURES_IFEATURESELECTOR_H_

#include <cereal/access.hpp>

#include <unordered_set>
#include <random>
#include <vector>
#include <algorithm>

#include "../global.h"
#include "../types.h"
#include "../util/sampling.h"
#include "./ifeatureproposer.h"

namespace forpy {
  /**
   * \brief The selection provider generates index combination of data
   * dimensions.
   *
   * During the optimization, \f$\phi\f$ may select only a subset of the
   * provided data dimensions. The selection provider may "suggest" many
   * possible such selections during the optimization process.
   *
   * The method \ref register_used must be used after having selected a proposed
   * selection, so that it can be taken into account for the generation of
   * future proposals.
   */
  class IFeatureSelector {
   public:
    virtual ~IFeatureSelector();

    /** Gets the required input dimension. */
    virtual size_t get_input_dimension() const VIRTUAL(size_t);

    /** \brief Get the dimension of one selection proposal. */
    virtual size_t get_selection_dimension() const VIRTUAL(size_t);

    /** \brief Get a set of all proposals for one node. */
    virtual proposal_set_t get_proposals() VIRTUAL(proposal_set_t);

    /** \brief Get a proposal generator to generate proposals for one node on
     * demand.
     */
    virtual std::shared_ptr<IFeatureProposer> get_proposal_generator()
      VIRTUAL_PTR;

    /** \brief Register a set of proposal vectors as used. Each proposal must
     *  be unique in the set. */
    virtual void register_used(const proposal_set_t &proposals) VIRTUAL_VOID;

    /** Deep comparison. */
    virtual bool operator==(const IFeatureSelector &rhs)
      const VIRTUAL(bool);

   protected:
    /** \brief Empty constructor. */
    IFeatureSelector();

   private:

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive &, const uint &) {};

    DISALLOW_COPY_AND_ASSIGN(IFeatureSelector);
  };
}  // namespace forpy
#endif  // FORPY_FEATURES_IFEATURESELECTOR_H_
