#!/usr/bin/env python
"""Test tree."""
# pylint: disable=no-member, too-many-instance-attributes
import os.path as path
import sys
import unittest

import numpy as np

sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))


class TestTree(unittest.TestCase):
    """Test tree."""

    def setUp(self):
        """Set up fixture."""
        import forpy
        np.random.seed(1)
        self.dta = np.array(range(20), dtype=np.float32).reshape((5, 4))
        self.annot = np.array(range(5), dtype=np.uint32).reshape((5, 1))
        self.annot_r = np.array(range(10), dtype=np.float32).reshape((5, 2))
        self.dta_t = np.ascontiguousarray(self.dta.T)
        self.dprov = forpy.FastDProv(self.dta_t, self.annot)
        self.wdta = np.random.normal(size=(1, 32)).astype(np.float32)
        self.wannot = np.random.randint(0, 10, size=(32, 1)).astype(np.uint32)
        self.weights = np.random.normal(size=(32, ))
        self.weights += np.abs(self.weights.min())
        self.wprov = forpy.FastDProv(self.wdta, self.wannot, self.weights)
        self.dprov_r = forpy.FastDProv(self.dta_t, self.annot_r)

    def test_plain(self):
        """Test plain tree."""
        import forpy
        tree = forpy.Tree()
        tree.fit(self.dta_t, self.annot)
        results = tree.predict(self.dta)
        self.assertTrue(np.all(results.flat == range(5)))
        self.assertTrue(
            np.all(
                np.argmax(tree.predict_proba(self.dta), axis=1) == range(5)))
        self.assertTrue(tree.initialized)
        self.assertEqual(tree.weight, 1.)
        self.assertEqual(tree.n_nodes, 9)
        self.assertEqual(tree.get_input_data_dimensions(), 4)
        self.assertEqual(tree.n_samples_stored, 5)
        self.assertRaises(RuntimeError,
                          lambda: tree.fit(self.dta_t, self.annot))
        self.assertRaises(RuntimeError, lambda: tree.fit_dprov(self.dprov))
        tree = forpy.Tree()
        tree.fit_dprov(self.dprov)
        results = tree.predict(self.dta)
        self.assertTrue(np.all(results.flat == range(5)))

    def test_classification(self):
        """Test classification tree."""
        import forpy
        tree = forpy.ClassificationTree()
        tree.fit(self.dta_t, self.annot)
        results = tree.predict(self.dta)
        self.assertTrue(np.all(results.flat == range(5)))
        self.assertTrue(
            np.all(
                np.argmax(tree.predict_proba(self.dta), axis=1) == range(5)))
        self.assertTrue(tree.initialized)
        self.assertEqual(tree.weight, 1.)
        self.assertEqual(tree.n_nodes, 9)
        self.assertEqual(tree.get_input_data_dimensions(), 4)
        self.assertEqual(tree.n_samples_stored, 5)
        self.assertRaises(RuntimeError,
                          lambda: tree.fit(self.dta_t, self.annot))
        self.assertRaises(RuntimeError, lambda: tree.fit_dprov(self.dprov))

    def test_weights(self):
        """Test classification tree weights."""
        import forpy
        from sklearn.tree import DecisionTreeClassifier
        tree = forpy.ClassificationTree(max_depth=2)
        tree.fit_dprov(self.wprov)
        results = tree.predict(np.ascontiguousarray(self.wdta.T))
        dt = DecisionTreeClassifier(max_depth=2)  # pylint: disable=invalid-name
        dt.fit(self.wdta.T, self.wannot, self.weights)
        ressk = dt.predict(self.wdta.T)
        self.assertTrue(np.all(ressk == results.T))

    def test_regression(self):
        """Test regression tree."""
        import forpy
        tree = forpy.RegressionTree()
        tree.fit(self.dta_t, self.annot_r)
        results = tree.predict(self.dta)
        self.assertTrue(np.all(results == self.annot_r))
        self.assertTrue(tree.initialized)
        self.assertEqual(tree.weight, 1.)
        self.assertEqual(tree.n_nodes, 9)
        self.assertEqual(tree.get_input_data_dimensions(), 4)
        self.assertEqual(tree.n_samples_stored, 5)
        self.assertRaises(RuntimeError,
                          lambda: tree.fit(self.dta_t, self.annot))
        self.assertRaises(RuntimeError, lambda: tree.fit_dprov(self.dprov))


if __name__ == '__main__':
    unittest.main()
