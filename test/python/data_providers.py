#!/usr/bin/env python2
"""Getting data into the framework properly."""
# pylint: disable=no-member
import os.path as path
import sys
import unittest

import numpy as np

sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))


class TestDataProviders(unittest.TestCase):
    """Data providers working."""

    def test_fast(self):
        """Test fast data provider."""
        import forpy
        pdp = forpy.FastDProv(np.ones((5, 10)), np.zeros((10, 3)))
        with self.assertRaises(RuntimeError):
            pdp = forpy.FastDProv(np.ones((8, 5)), np.zeros((10, 3)))
        self.assertEqual(pdp.get_initial_sample_list(), list(range(10)))
        self.assertEqual(pdp.feat_vec_dim, 5)
        self.assertEqual(pdp.annot_vec_dim, 3)
        tps = pdp.create_tree_providers([(range(10), []), (range(5), [])])
        self.assertEqual(tps[0], pdp)
        self.assertEqual(len(tps[1].get_initial_sample_list()), 5)
        res = pdp.get_annotations()
        self.assertEqual(res.shape, (10, 3))
        for feat_idx in range(5):
            res = pdp.get_feature(feat_idx)
            self.assertEqual(res.sum(), 10)
            self.assertEqual(res.shape[0], 10)
        _ = pdp.__repr__()


if __name__ == '__main__':
    unittest.main()
