/* Author: Christoph Lassner. */
#pragma once
#ifndef FORPY_UTIL_SERIALINCLUDES_H_
#define FORPY_UTIL_SERIALINCLUDES_H_

#include <cereal/archives/portable_binary.hpp>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wexceptions"
#include <cereal/archives/json.hpp>
#pragma clang diagnostic pop
#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>

#endif  // FORPY_UTIL_SERIALINCLUDES_H_
