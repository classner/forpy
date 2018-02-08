#!/usr/bin/env python
"""Test for quality of the models."""
# pylint: disable=no-member, no-self-use
from __future__ import print_function

import os.path as path
import sys
import timeit
import unittest

import numpy as np

sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))


class QualityTestCase(unittest.TestCase):
    """Tests for quality of the created models."""

    def setUp(self):
        """Create the fixture."""
        np.random.seed(1)
        self.dta = np.random.normal(size=(10000, 1)).astype(np.float32)
        self.dta_t = np.ascontiguousarray(self.dta.T)
        self.annot = np.random.randint(
            low=0, high=2, size=(10000, 1)).astype(np.uint32)
        self.annot_r = np.random.randint(
            low=0, high=10, size=(10000, 1)).astype(np.float32)

    def test_plain_deep(self):
        """Test plain tree prediction quality."""
        import forpy
        import sklearn.metrics as skm

        def fit():
            """Closure for timing."""
            tree = forpy.ClassificationTree(max_depth=20)
            tree.fit(self.dta_t, self.annot)
            return tree

        print("forpy fitting time (deep): " + str(
            timeit.Timer(fit).timeit(int(1E1)) / 1E1))
        tree = fit()
        print("forpy depth: ", tree.depth)
        print("forpy accuracy: ", skm.accuracy_score(self.annot,
                                                     tree.predict(self.dta)))

    def test_plain_regr(self):
        """Test regr tree prediction quality."""
        import forpy
        import sklearn.metrics as skm

        def fit():
            """Closure for timing."""
            tree = forpy.RegressionTree(max_depth=20)
            tree.fit(self.dta_t, self.annot_r)
            return tree

        print("forpy fitting time (regr): " + str(
            timeit.Timer(fit).timeit(int(1E1)) / 1E1))
        tree = fit()
        print("forpy depth: ", tree.depth)
        print("forpy MSE: ", skm.mean_squared_error(
            self.annot, np.ascontiguousarray(tree.predict(self.dta))))

    def test_sklearn_deep(self):
        """Test plain tree prediction speed."""
        import sklearn.tree as skt
        import sklearn.metrics as skm

        def fit():
            """Closure for timing."""
            tree = skt.DecisionTreeClassifier(
                max_depth=20, min_samples_split=2, min_samples_leaf=1)
            tree.fit(self.dta, self.annot)
            return tree

        print("sklearn fitting time (deep): " + str(
            timeit.Timer(fit).timeit(int(1E1)) / 1E1))
        tree = fit()
        print("sklearn depth: ", tree.tree_.max_depth)
        print("sklearn accuracy: ", skm.accuracy_score(self.annot,
                                                       tree.predict(self.dta)))

    def test_sklearn_regr(self):
        """Test regr tree prediction quality."""
        import sklearn.tree as skt
        import sklearn.metrics as skm

        def fit():
            """Closure for timing."""
            tree = skt.DecisionTreeRegressor(
                criterion='mse',
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1)
            tree.fit(self.dta, self.annot_r)
            return tree

        print("sklearn fitting time (regr): " + str(
            timeit.Timer(fit).timeit(int(1E1)) / 1E1))
        tree = fit()
        print("sklearn depth: ", tree.tree_.max_depth)
        print("sklearn MSE: ", skm.mean_squared_error(self.annot,
                                                      tree.predict(self.dta)))


if __name__ == '__main__':
    unittest.main()
