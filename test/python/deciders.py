#!/usr/bin/env python2
import numpy as np
import unittest
import sys
import os.path as path
sys.path.insert(0,
                path.join(path.dirname(__file__),
                          '..', '..'))
import forpy

class TestDeciders(unittest.TestCase):

    def test_threshold(self):
        """Test threshold decider."""
        td = forpy.ThresholdDecider(forpy.ClassificationThresholdOptimizer(),
                                    2)
        self.assertTrue(td.supports_weights())
        td = forpy.ThresholdDecider(forpy.RegressionThresholdOptimizer(),
                                    2)
        self.assertTrue(not td.supports_weights())
        td = forpy.ThresholdDecider(forpy.ClassificationThresholdOptimizer(),
                                    2)
        self.assertEqual(td.get_data_dim(), 1)
        td.__repr__()
        self.assertRaises(RuntimeError, lambda: td.decide(0, np.ones((3, 3))))
        dta = np.array(range(20), dtype=np.float32).reshape((5, 4))
        annotations = np.array(range(5), dtype=np.uint32).reshape((5, 1))
        dp = forpy.PlainDataProvider(dta, annotations)
        td.make_node(0, 0, 1, range(5), dp)
        results = []
        for s_idx in range(5):
            results.append(td.decide(0, dta[s_idx:s_idx+1, :]))
        self.assertEqual(np.sum(results), 2)
        td.make_node(1, 1, 1, range(5), dp)


if __name__ == '__main__':
    unittest.main()
