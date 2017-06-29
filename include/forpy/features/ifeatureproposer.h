/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_FEATURES_IFEATUREPROPOSER_H_
#define FORPY_FEATURES_IFEATUREPROPOSER_H_

#include <cereal/access.hpp>

#include <vector>

#include "../global.h"

namespace forpy {
  /** Utility class used by the \ref IFeatureSelectors. */
  class IFeatureProposer {
  public:
    virtual ~IFeatureProposer();

    virtual bool available() const VIRTUAL(bool);

    virtual size_t max_count() const VIRTUAL(size_t);

    virtual std::vector<size_t> get_next() VIRTUAL(std::vector<size_t>);

    virtual bool operator==(const IFeatureProposer &rhs) const VIRTUAL(bool);

  protected:
    IFeatureProposer();

  private:
    DISALLOW_COPY_AND_ASSIGN(IFeatureProposer);
  };
}  // namespace forpy
#endif  // FORPY_FEATURES_IFEATUREPROPOSER_H_
