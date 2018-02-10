#!/usr/bin/env python
"""Test tree."""
# pylint: disable=no-member
import os.path as path
import sys
import unittest

import numpy as np

sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))


class TestForest(unittest.TestCase):
    """Test the forest."""

    def setUp(self):
        """Set up fixture."""
        import forpy
        self.dta = np.array(range(20), dtype=np.float32).reshape((5, 4))
        self.annot = np.array(range(5), dtype=np.uint32).reshape((5, 1))
        self.annot_r = np.array(range(10), dtype=np.float32).reshape((5, 2))
        self.dta_t = np.ascontiguousarray(self.dta.T)
        self.dprov = forpy.FastDProv(self.dta_t, self.annot)
        self.dprov_r = forpy.FastDProv(self.dta_t, self.annot_r)

    def test_plain(self):
        """Test plain forest."""
        import forpy
        forest = forpy.Forest()
        forest.fit(self.dta_t, self.annot)
        results = forest.predict(self.dta)
        self.assertTrue(np.all(results.flat == range(5)))
        self.assertTrue(
            np.all(
                np.argmax(forest.predict_proba(self.dta), axis=1) == range(5)))
        self.assertTrue(np.allclose(forest.tree_weights, 1.))
        self.assertEqual(len(forest.trees), 10)
        self.assertEqual(forest.get_input_data_dimensions(), 4)
        self.assertRaises(RuntimeError,
                          lambda: forest.fit(self.dta_t, self.annot))
        self.assertRaises(RuntimeError, lambda: forest.fit_dprov(self.dprov))
        forest = forpy.Forest()
        forest.fit_dprov(self.dprov)
        results = forest.predict(self.dta)
        self.assertTrue(np.all(results.flat == range(5)))

    def test_classification(self):
        """Test classification forest."""
        import forpy
        forest = forpy.ClassificationForest()
        forest.fit(self.dta_t, self.annot)
        results = forest.predict(self.dta)
        self.assertTrue(np.all(results.flat == range(5)))
        self.assertTrue(
            np.all(
                np.argmax(forest.predict_proba(self.dta), axis=1) == range(5)))
        self.assertTrue(np.allclose(forest.tree_weights, 1.))
        self.assertEqual(len(forest.trees), 10)
        self.assertEqual(forest.get_input_data_dimensions(), 4)
        self.assertRaises(RuntimeError,
                          lambda: forest.fit(self.dta_t, self.annot))
        self.assertRaises(RuntimeError, lambda: forest.fit_dprov(self.dprov))
        forest = forpy.Forest()
        forest.fit_dprov(self.dprov)
        results = forest.predict(self.dta)
        self.assertTrue(np.all(results.flat == range(5)))

    def test_bootstrap(self):
        """Test classification forest bootstrapping."""
        import forpy
        forest = forpy.ClassificationForest()
        forest.fit(self.dta_t, self.annot)
        results = forest.predict(self.dta)
        self.assertTrue(np.all(results.flat == range(5)))
        self.assertTrue(
            np.all(
                np.argmax(forest.predict_proba(self.dta), axis=1) == range(5)))
        self.assertTrue(np.allclose(forest.tree_weights, 1.))
        self.assertEqual(len(forest.trees), 10)
        self.assertEqual(forest.get_input_data_dimensions(), 4)
        self.assertRaises(RuntimeError,
                          lambda: forest.fit(self.dta_t, self.annot))
        self.assertRaises(RuntimeError, lambda: forest.fit_dprov(self.dprov))
        forest = forpy.Forest()
        forest.fit_dprov(self.dprov)
        results = forest.predict(self.dta)
        self.assertTrue(np.all(results.flat == range(5)))

    def test_regression(self):
        """Test regression forest."""
        import forpy
        forest = forpy.RegressionForest()
        forest.fit(self.dta_t, self.annot_r, bootstrap=False)
        results = forest.predict(self.dta)
        self.assertTrue(np.all(results == self.annot_r))
        self.assertRaises(RuntimeError,
                          lambda: forest.fit(self.dta_t, self.annot))
        self.assertRaises(RuntimeError, lambda: forest.fit_dprov(self.dprov))


if __name__ == '__main__':
    unittest.main()
