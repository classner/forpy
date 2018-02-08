#!/usr/bin/env python
"""Test the iris dataset."""
# pylint: disable=no-member
from __future__ import print_function

import os.path as path
import sys
import unittest

import numpy as np

sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))


class IrisTest(unittest.TestCase):
    """Test performance on the Iris dataset."""

    def test_iris(self):
        """Test forpy classification."""
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        np.random.seed(3)
        import sklearn.metrics as skm
        import forpy
        iris = load_iris()
        fpscores = []
        skscores = []
        for i in range(20):
            clf = forpy.ClassificationForest(random_seed=i + 1)
            clf = clf.fit(
                np.ascontiguousarray(iris.data[:125, :].T.astype(np.float32)),
                iris.target[:125].astype(np.uint32))
            fpsc = skm.accuracy_score(
                clf.predict(iris.data[125:, :]), iris.target[125:])
            clf = RandomForestClassifier()
            clf.fit(iris.data[:125, :], iris.target[:125])
            sksc = skm.accuracy_score(
                clf.predict(iris.data[125:, :]), iris.target[125:])
            fpscores.append(fpsc)
            skscores.append(sksc)
        print("forpy acc.: {}, sklearn acc.: {}".format(
            np.mean(fpscores), np.mean(skscores)))
        self.assertGreaterEqual(np.mean(fpscores) * 1.05, np.mean(skscores))
        fpscores = []
        skscores = []
        for i in range(20):
            clf = forpy.ClassificationForest(max_depth=2, random_seed=i + 1)
            clf = clf.fit(
                np.ascontiguousarray(iris.data[:125, :].T.astype(np.float32)),
                iris.target[:125].astype(np.uint32))
            fpsc = skm.accuracy_score(
                clf.predict(iris.data[125:, :]), iris.target[125:])
            clf = RandomForestClassifier(max_depth=2, bootstrap=True)
            clf.fit(iris.data[:125, :], iris.target[:125])
            sksc = skm.accuracy_score(
                clf.predict(iris.data[125:, :]), iris.target[125:])
            fpscores.append(fpsc)
            skscores.append(sksc)
        print("forpy acc.: {}, sklearn acc.: {}".format(
            np.mean(fpsc), np.mean(sksc)))
        self.assertGreaterEqual(fpsc * 1.05, sksc)

    def test_iris_reg(self):
        """Test forpy regression."""
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestRegressor
        import sklearn.metrics as skm
        np.random.seed(1)
        import forpy
        iris = load_iris()
        fpscores = []
        skscores = []
        for i in range(20):
            clf = forpy.RegressionForest(random_seed=i + 1)
            clf = clf.fit(
                np.ascontiguousarray(iris.data[:125, :].T.astype(np.float32)),
                iris.target[:125].astype(np.float32),
                bootstrap=True)
            fpsc = skm.mean_squared_error(
                clf.predict(iris.data[125:, :]), iris.target[125:])
            clf = RandomForestRegressor()
            clf.fit(iris.data[:125, :], iris.target[:125])
            sksc = skm.mean_squared_error(
                clf.predict(iris.data[125:, :]), iris.target[125:])
            fpscores.append(fpsc)
            skscores.append(sksc)
        print("forpy mse: ", np.mean(fpscores), ' sklearn mse: ',
              np.mean(skscores))
        self.assertLessEqual(np.mean(fpscores), np.mean(skscores) * 1.05)
        fpscores = []
        skscores = []
        for i in range(20):
            clf = forpy.RegressionForest(
                max_depth=2, random_seed=i + 2, gain_threshold=1E-7)
            clf = clf.fit(
                np.ascontiguousarray(iris.data[:125, :].T.astype(np.float32)),
                iris.target[:125, None].astype(np.float32),
                bootstrap=True)
            fpsc = skm.mean_squared_error(
                clf.predict(iris.data[125:, :]), iris.target[125:])
            clf = RandomForestRegressor(max_depth=2)
            clf.fit(iris.data[:125, :], iris.target[:125])
            sksc = skm.mean_squared_error(
                clf.predict(iris.data[125:, :]), iris.target[125:])
            fpscores.append(fpsc)
            skscores.append(sksc)
        print("forpy mse: ", np.mean(fpscores), ' sklearn mse: ',
              np.mean(skscores))
        self.assertLessEqual(np.mean(fpscores), np.mean(skscores) * 1.05)


if __name__ == '__main__':
    unittest.main()
