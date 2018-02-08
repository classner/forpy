#!/usr/bin/env python
"""Benchmark prediction speeds."""
# pylint: disable=no-member, no-self-use
from __future__ import print_function

import os.path as path
import sys
import timeit
import unittest

import numpy as np

sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))


class PredictionTestCase(unittest.TestCase):
    """Test prediction performance."""

    def setUp(self):
        """Create the fixture."""
        self.dta = np.array(range(20), dtype=np.float32).reshape((5, 4))
        self.dta_t = np.ascontiguousarray(self.dta.T)
        self.annot = np.array(range(5), dtype=np.uint32).reshape((5, 1))
        self.annot_r = np.random.normal(size=(5, 1)).astype('float32')

    def test_plain_shallow(self):
        """Test plain tree prediction speed."""
        import forpy
        tree = forpy.ClassificationTree(max_depth=20)
        tree.fit(self.dta_t, self.annot)
        tree.enable_fast_prediction()
        results = tree.predict(self.dta)
        self.assertTrue(np.all(results.flat == range(5)))
        print("forpy prediction time: " + str(
            timeit.Timer(lambda: tree.predict(self.dta)).timeit(int(1E5)) / 1E5
        ))

    def test_sklearn_shallow(self):
        """Compare with sklearn."""
        import sklearn.tree as skt
        tree = skt.DecisionTreeClassifier()
        tree.fit(self.dta, self.annot)
        results = tree.predict(self.dta)
        self.assertTrue(np.all(results == range(5)))
        self.assertEqual(tree.tree_.max_depth, 4)
        print("sklearn prediction time: " + str(
            timeit.Timer(lambda: tree.predict(self.dta)).timeit(int(1E5)) / 1E5
        ))

    def test_plain_shallow_single(self):
        """Test plain tree prediction speed."""
        import forpy
        tree = forpy.ClassificationTree(max_depth=20)
        tree.fit(self.dta_t, self.annot)
        tree.enable_fast_prediction()
        results = tree.predict(self.dta)
        self.assertTrue(np.all(results.flat == range(5)))
        print("forpy prediction time (single): " + str(
            timeit.Timer(lambda: tree.predict(self.dta))
            .timeit(int(1E5)) / 1E5))

    def test_sklearn_shallow_single(self):
        """Compare with sklearn."""
        import sklearn.tree as skt
        tree = skt.DecisionTreeClassifier()
        tree.fit(self.dta, self.annot)
        results = tree.predict(self.dta)
        self.assertTrue(np.all(results == range(5)))
        self.assertEqual(tree.tree_.max_depth, 4)
        print("sklearn prediction time (single): " + str(
            timeit.Timer(lambda: tree.predict(self.dta))
            .timeit(int(1E5)) / 1E5))

    def test_plain_deep(self):
        """Test plain tree prediction speed (deep)."""
        import forpy
        tree = forpy.ClassificationTree(max_depth=20)
        tree.fit(self.dta_t, self.annot)
        tree.enable_fast_prediction()
        print("tree depth: ", tree.depth)
        print("forpy prediction time (deep): " + str(
            timeit.Timer(lambda: tree.predict(self.dta)).timeit(int(1E2)) / 1E2
        ))

    def test_reg_deep(self):
        """Test regression tree prediction speed (deep)."""
        import forpy
        tree = forpy.RegressionTree(
            max_depth=20, min_samples_at_leaf=6, min_samples_at_node=12)
        tree.fit(self.dta_t, self.annot_r)
        tree.enable_fast_prediction()
        print("tree depth: ", tree.depth)
        print("forpy prediction time (deep, reg): " + str(
            timeit.Timer(lambda: tree.predict(self.dta)).timeit(int(1E2)) / 1E2
        ))

    def test_sklearn_deep(self):
        """Compare with sklearn (deep tree)."""
        import sklearn.tree as skt
        tree = skt.DecisionTreeClassifier(max_depth=20)
        tree.fit(self.dta, self.annot)
        print('tree depth sklearn', tree.tree_.max_depth)
        print("sklearn prediction time (deep): " + str(
            timeit.Timer(lambda: tree.predict(self.dta)).timeit(int(1E2)) / 1E2
        ))

    def test_sklearn_reg_deep(self):
        """Compare with sklearn (deep reg tree)."""
        import sklearn.tree as skt
        np.random.seed(1)
        tree = skt.DecisionTreeRegressor(max_depth=20)
        tree.fit(self.dta, self.annot_r)
        print('tree depth sklearn', tree.tree_.max_depth)
        print("sklearn prediction time (deep, reg): " + str(
            timeit.Timer(lambda: tree.predict(self.dta)).timeit(int(1E2)) / 1E2
        ))

    def test_plain_deep_single(self):
        """Test plain tree prediction speed (deep, single sample)."""
        import forpy
        tree = forpy.ClassificationTree(max_depth=20)
        tree.fit(self.dta_t, self.annot)
        tree.enable_fast_prediction()
        print("tree depth: ", tree.depth)
        print("forpy prediction time (deep, single): " + str(
            timeit.Timer(lambda: tree.predict(self.dta))
            .timeit(int(1E5)) / 1E5))

    def test_sklearn_deep_single(self):
        """Compare with sklearn (deep tree, single sample)."""
        import sklearn.tree as skt
        tree = skt.DecisionTreeClassifier(max_depth=20)
        tree.fit(self.dta, self.annot)
        print('tree depth sklearn', tree.tree_.max_depth)
        print("sklearn prediction time (deep, single): " + str(
            timeit.Timer(lambda: tree.predict(self.dta))
            .timeit(int(1E3)) / 1E3))

    def test_forest_deep(self):
        """Test forest prediction speed deep."""
        import forpy
        tree = forpy.ClassificationForest(max_depth=20)
        tree.fit(self.dta_t, self.annot)
        tree.enable_fast_prediction()
        print("tree depths: ", tree.depths)
        print("forpy prediction time (deep, forest): " + str(
            timeit.Timer(lambda: tree.predict(self.dta))
            .timeit(int(1E5)) / 1E5))

    def test_sklearn_forest_deep(self):
        """Compare with sklearn (deep tree, single sample)."""
        import sklearn.ensemble as ske
        tree = ske.RandomForestClassifier(max_depth=20)
        tree.fit(self.dta, self.annot)
        print("sklearn prediction time (deep, forest): " + str(
            timeit.Timer(lambda: tree.predict(self.dta))
            .timeit(int(1E2)) / 1E2))


if __name__ == "__main__":
    unittest.main()
