#!/usr/bin/env python2
import numpy as np
import unittest
import sys
import os.path as path
sys.path.insert(0,
                path.join(path.dirname(__file__),
                          '..', '..'))
import forpy

class TestDataProviders(unittest.TestCase):

    def test_plain(self):
        """Test plain data provider."""
        pdp = forpy.PlainDataProvider(np.ones((10, 5)),
                                      np.zeros((10, 3)))
        with self.assertRaises(RuntimeError):
            pdp = forpy.PlainDataProvider(np.ones((8, 5)),
                                          np.zeros((10, 3)))
        self.assertEqual(pdp.get_initial_sample_list(),
                         range(10))
        samples = pdp.get_samples()
        for sample in samples:
            self.assertTrue(np.all(sample.data == np.ones((1, 1))))
            self.assertTrue(np.all(sample.annotation == np.zeros((1, 1))))
            self.assertTrue(sample.weight == 1.)
            self.assertRaises(RuntimeError, lambda: sample.parent_dt)
            self.assertRaises(RuntimeError, lambda: sample.parent_at)
        self.assertEqual(pdp.feat_vec_dim, 5)
        self.assertEqual(pdp.annot_vec_dim, 3)
        tps = pdp.create_tree_providers([range(10), range(5)])
        self.assertEqual(tps[0], pdp)
        self.assertEqual(len(tps[1].get_initial_sample_list()), 5)
        _ = pdp.__repr__()

    def test_sample(self):
        """Test the sample object."""
        s1 = forpy.Sample_d_d(np.array(range(10), dtype=np.float64).reshape((1, 10)),
                              np.ones((1, 4), dtype=np.float64),
                              1.)
        self.assertTrue(np.all(s1.data == range(10)))
        self.assertTrue(np.all(s1.annotation == np.ones((1, 4))))
        self.assertEqual(s1.weight, 1.)
        s2 = forpy.Sample_d_d(np.array(range(10), dtype=np.float64).reshape((1, 10)),
                              np.ones((1, 4)),
                              1.)
        self.assertEqual(s1, s2)
        s3 = forpy.Sample_d_ui(np.array(range(10), dtype=np.float64).reshape((1, 10)),
                               np.ones((1, 4), dtype=np.uint32),
                               1.)
        self.assertRaises(TypeError, lambda: s2 == s3)
        self.assertRaises(RuntimeError, lambda: s3.parent_dt)
        self.assertRaises(RuntimeError, lambda: s3.parent_at)
        _ = s3.__repr__()


if __name__ == '__main__':
    unittest.main()
