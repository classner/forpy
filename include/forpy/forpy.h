#pragma once
#ifndef FORPY_FORPY_H_
#define FORPY_FORPY_H_

#include "./global.h"
#include "./types.h"

/// \defgroup forpydata_providersGroup Data Providers
/// \brief Structures storing and providing data throughout the training.
#include "./data_providers/data_providers.h"

/// \defgroup forpydecidersGroup Deciders
/// \brief Decision functions that can be used in the inner nodes of a tree,
/// \f$h(v, \theta_j)\f$.
///
/// The objects are responsible for storing the parameters \f$\theta_j\f$
/// themselves. They provide a function IDecider::decide that provides the
/// corresponding result.
#include "./deciders/deciders.h"

/// \defgroup forpygainsGroup Gains
/// \brief Gain calculators for calculating \f$I\f$.
///
/// The only implementation as of now is the standard entropy gain
/// (forpy::EntropyGain).
#include "./gains/gains.h"

/// \defgroup forpyimpuritiesGroup Impurity Measures
/// \brief Impurity measures for calculating \f$H\f$.
///
/// They are currently only used for providing classification impurities. The
/// differential entropy has theoretical and practical disadvantages and has
/// been omitted so far in favor of direct MSE optimization.
#include "./impurities/impurities.h"

/// \defgroup forpyleafsGroup Leafs
/// \brief Leafs for storing the leaf distributions and estimation of joint
/// distributions.
#include "./leafs/leafs.h"

/// \defgroup forpythreshold_optimizersGroup Threshold Optimizers
/// \brief These objects encapsulate the optimization of split parameters
/// \f$\theta_j\f$.
#include "./threshold_optimizers/threshold_optimizers.h"

/// \defgroup forpydeskGroup Desk Implementations
///
/// \brief The 'desks' encapsulate thread local storage.
///
/// Object construction and destruction can be largely avoided and object
/// const-ness can be used as an automatic checker for memory access.
#include "./util/desk.h"

#include "./forest.h"
#include "./tree.h"

#include "./version.h"

#endif  // FORPY_FORPY_H_
