/* Author: Christoph Lassner. */
#include "gtest/gtest.h"

#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <forpy/impurities/impurities.h>
#include <forpy/types.h>

#include "./setup.h"
#include "./timeit.h"

// Test objects.
using forpy::ClassificationError;
using forpy::IEntropyFunction;
using forpy::InducedEntropy;
using forpy::RenyiEntropy;
using forpy::ShannonEntropy;
using forpy::TsallisEntropy;

using namespace forpy;

namespace {

class IEntropyTest : public ::testing::Test {
 public:
  /** Some vectors with class numbers. */
  std::vector<float> memberNumbers;
  std::vector<float> part_1;
  std::vector<float> part_2;
  /** A 2x2 zero matrix. */
  Eigen::Matrix<float, 2, 2, Eigen::RowMajor> mat;
  /** A 1x1 matrix with the value 4. */
  Eigen::Matrix<float, 1, 1, Eigen::RowMajor> mat_1x1_4;
  /** A 2x2 matrix with the values 2, 0, 0, 4. */
  Eigen::Matrix<float, 2, 2, Eigen::RowMajor> mat_2x2_r;
  /** A list of entropies to test. */
  std::vector<std::shared_ptr<IEntropyFunction>> to_test;
  /** A list of functions to calculate the max entropy values. */
  std::vector<std::function<float(unsigned int)>> max_calc;
  /** Names of the entropies for speed output. */
  std::vector<std::string> names;
  /** Equidistribution of ones. */
  std::vector<float> eq_dist_1;
  /** Equidistribution of 2s. */
  std::vector<float> eq_dist_2;

  IEntropyTest()
      : eq_dist_1(10, static_cast<float>(1)),
        eq_dist_2(10, static_cast<float>(2)) {
    // Initialize the first to elements to zero, the rest to arbitrary
    // values for the tests.
    memberNumbers = std::vector<float>(10);
    for (size_t i = 0; i < 10; ++i)
      memberNumbers[i] = static_cast<float>(i / 2);
    part_1 =
        std::vector<float>(memberNumbers.begin(), memberNumbers.begin() + 1);
    part_2 =
        std::vector<float>(memberNumbers.begin(), memberNumbers.begin() + 2);
    mat << 0.f, 0.f, 0.f, 0.f;
    mat_1x1_4 << 4.f;
    mat_2x2_r << 2.f, 0.f, 0.f, 4.f;
    // Setup test objects.
    names.push_back("shannon");
    to_test.push_back(std::make_shared<ShannonEntropy>());
    max_calc.push_back([](unsigned int n_classes) {
      return logf(static_cast<float>(n_classes)) / logf(2.f);
    });
    names.push_back("classification_error");
    to_test.push_back(std::make_shared<ClassificationError>());
    max_calc.push_back([](unsigned int n_classes) {
      return 1.f - 1.f / static_cast<float>(n_classes);
    });
    std::vector<float> pvals;
    pvals.push_back(2.);
    pvals.push_back(1.2);
    pvals.push_back(2.5);
    pvals.push_back(3.);
    pvals.push_back(4.);
    pvals.push_back(5.);
    for (float pval : pvals) {
      names.push_back("induced(" + std::to_string(pval) + ")");
      to_test.push_back(std::make_shared<InducedEntropy>(pval));
      max_calc.push_back([pval](unsigned int n_classes) {
        float n_classes_f = static_cast<float>(n_classes);
        return powf(1.f - 1.f / n_classes_f, pval) +
               (n_classes_f - 1.f) * powf(1.f / n_classes_f, pval);
      });
      names.push_back("renyi(" + std::to_string(pval) + ")");
      to_test.push_back(std::make_shared<RenyiEntropy>(pval));
      max_calc.push_back([pval](unsigned int n_classes) {
        float n_classes_f = static_cast<float>(n_classes);
        return logf(n_classes_f * powf(1.f / n_classes_f, pval)) /
               (1.f - static_cast<float>(pval));
      });
      names.push_back("tsallis(" + std::to_string(pval) + ")");
      to_test.push_back(std::make_shared<TsallisEntropy>(pval));
      max_calc.push_back([pval](unsigned int n_classes) {
        float n_classes_f = static_cast<float>(n_classes);
        return (1.f - n_classes_f * powf(1.f / n_classes_f, pval)) /
               (static_cast<float>(pval) - 1.f);
      });
    }
  };
};

TEST_F(IEntropyTest, CorrEntropyOfZerosIsZero) {
  for (auto tfunc : to_test) {
    ASSERT_EQ((*tfunc)(std::vector<float>()), static_cast<float>(0));
    ASSERT_EQ((*tfunc)(std::vector<float>(), static_cast<float>(0)), 0.f);
    ASSERT_EQ((*tfunc)(part_1), 0.f);
    ASSERT_EQ((*tfunc)(part_1, 0.f), 0.f);
    ASSERT_EQ((*tfunc)(part_2), 0.f);
    ASSERT_EQ((*tfunc)(part_2, 0.f), 0.f);
  }
}

TEST_F(IEntropyTest, CorrExtremeCases) {
  for (auto tfunc : to_test) {
    // Check for zero classes.
    ASSERT_EQ((*tfunc)(std::vector<float>()), 0.f);
    ASSERT_EQ((*tfunc)(std::vector<float>(), 0.f), 0.f);
    // Check for only one class.
    part_1[0] = 0.f;
    ASSERT_EQ((*tfunc)(part_1), 0.f);
    ASSERT_EQ((*tfunc)(part_1, 0.f), 0.f);
    part_1[0] = 1.f;
    ASSERT_EQ((*tfunc)(part_1), 0.f);
    ASSERT_EQ((*tfunc)(part_1, 1.f), 0.f);
    part_1[0] = 2.f;
    ASSERT_EQ((*tfunc)(part_1), 0.f);
    ASSERT_EQ((*tfunc)(part_1, 2.f), 0.f);
  }
}

TEST_F(IEntropyTest, CorrEntropyMax) {
  for (unsigned int ent_id = 0; ent_id < to_test.size(); ++ent_id) {
    for (unsigned int members = 2; members < 3; ++members) {
      float objective_value = max_calc[ent_id](members);
      EXPECT_NEAR((*to_test[ent_id])(std::vector<float>(
                      eq_dist_1.begin(), eq_dist_1.begin() + members)),
                  objective_value, 0.000001f)
          << "Failed for id " << ent_id << " and " << members << " members.";
      EXPECT_NEAR(
          (*to_test[ent_id])(std::vector<float>(eq_dist_1.begin(),
                                                eq_dist_1.begin() + members),
                             static_cast<float>(members)),
          objective_value, 0.000001f)
          << "Failed for id " << ent_id << " and " << members << " members.";
      EXPECT_NEAR((*to_test[ent_id])(std::vector<float>(
                      eq_dist_2.begin(), eq_dist_2.begin() + members)),
                  objective_value, 0.000001f)
          << "Failed for id " << ent_id << " and " << members << " members.";
      EXPECT_NEAR(
          (*to_test[ent_id])(std::vector<float>(eq_dist_2.begin(),
                                                eq_dist_2.begin() + members),
                             2.f * static_cast<float>(members)),
          objective_value, 0.000001f)
          << "Failed for id " << ent_id << " and " << members << " members.";
    }
  }
}

TEST_F(IEntropyTest, DISABLED_SpeedOK) {
  struct entropy_timer : public Utility::ITimefunc {
    entropy_timer(const std::vector<float> &numbers,
                  std::shared_ptr<IEntropyFunction> *ef)
        : numbers(numbers), ef(*ef) {}
    int operator()() { return static_cast<int>((*ef)(numbers)); }

   private:
    std::vector<float> numbers;
    std::shared_ptr<IEntropyFunction> ef;
  };

  double expected[] = {101.5, 10.5, 33.6, 50.2, 32.6, 738.5, 620.9, 590.9};
  for (unsigned int func_id = 0; func_id < 8; ++func_id) {
    // Get the measurement in ms.
    auto timer = entropy_timer(memberNumbers, &to_test[func_id]);
    float ent_time =
        Utility::timeit<std::chrono::nanoseconds>(&timer, false, 3, 2);
    std::cerr << "[          ] " << names[func_id] << ": "
              << std::to_string(ent_time) << "ns" << std::endl;
    EXPECT_LE(ent_time, expected[func_id] * 1.1);
  }
}
}  // namespace
