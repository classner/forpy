#!/usr/bin/env python2
"""Check the deciders."""
# pylint: disable=no-member, invalid-name
import os.path as path
import sys
import unittest

import numpy as np

sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))


class TestDeciders(unittest.TestCase):
    """Deciders work."""

    def test_threshold(self):
        """Test fast decider."""
        import forpy
        td = forpy.FastDecider()
        self.assertFalse(td.supports_weights())
        td = forpy.FastDecider(forpy.RegressionOpt())
        self.assertTrue(not td.supports_weights())
        td = forpy.FastDecider(forpy.ClassificationOpt(), 2)
        td.set_data_dim(4)
        self.assertEqual(td.get_data_dim(), 4)
        td.__repr__()
        self.assertRaises(RuntimeError, lambda: td.decide(0, np.ones((3, 3))))


if __name__ == '__main__':
    unittest.main()
