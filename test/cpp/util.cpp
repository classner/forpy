/* Author: Christoph Lassner. */
#include "gtest/gtest.h"

#include <forpy/util/exponentials.h>

// Test objects.
using forpy::fpowi;
using forpy::ipow;


TEST(fpowi, CorrCalculatesPow) {
  for (float i = 0.f; i < 8.f; i++) {
    for (unsigned int j = 0; j < 10; j++)
      ASSERT_EQ(fpowi(i, j),
                static_cast<int>(pow(static_cast<double>(i),
                                     static_cast<double>(j))));
  }
};

TEST(ipow, CorrCalculatesPow) {
  for (int i = 0; i < 8; i++) {
    for (unsigned int j = 0; j < 10; j++)
      ASSERT_EQ(ipow(i, j),
                static_cast<int>(pow(static_cast<double>(i),
                                     static_cast<double>(j))));
  }
};
