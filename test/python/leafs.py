#!/usr/bin/env python2
"""Test leaf storage."""
import os.path as path
import sys
import unittest

# pylint: disable=no-member, invalid-name, no-self-use
import numpy as np

sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))


class TestLeafs(unittest.TestCase):
    """Leafs working."""

    def test_classification(self):
        """Test classification leaf."""
        import forpy
        pdp = forpy.FastDProv(np.ones((5, 10)), np.array(range(10), np.uint32))
        cto = forpy.ClassificationOpt()
        cto.check_annotations(pdp)
        cl = forpy.ClassificationLeaf()
        cl.__repr__()

    def test_regression(self):
        """Test regression leaf."""
        import forpy
        tree_1 = forpy.RegressionTree(summarize=False, store_variance=True)
        tree_1.fit(
            np.zeros((2, 10), dtype=np.float32),
            np.ones((10, 2), dtype=np.float32))
        tree_0 = forpy.RegressionTree(summarize=False, store_variance=True)
        tree_0.fit(
            np.zeros((2, 10), dtype=np.float32),
            np.zeros((10, 2), dtype=np.float32))
        fr = forpy.Forest([tree_0, tree_1])
        self.assertTrue(
            np.all(
                np.array([[0., 0., 0., 0., 1., 0., 1., 0.], [
                    0., 0., 0., 0., 1., 0., 1., 0.
                ]]) == fr.predict_proba(np.zeros((2, 2), dtype=np.float32))))
        tree_1 = forpy.RegressionTree(summarize=True, store_variance=True)
        tree_1.fit(
            np.zeros((2, 10), dtype=np.float32),
            np.ones((10, 2), dtype=np.float32))
        tree_0 = forpy.RegressionTree(summarize=True, store_variance=True)
        tree_0.fit(
            np.zeros((2, 10), dtype=np.float32),
            np.zeros((10, 2), dtype=np.float32))
        fr = forpy.Forest([tree_0, tree_1])
        self.assertTrue(
            np.all(np.array([[0.5, 0.25, 0.5, 0.25], [0.5, 0.25, 0.5, 0.25]])))
        cl = forpy.RegressionLeaf()
        cl.__repr__()


if __name__ == '__main__':
    unittest.main()
