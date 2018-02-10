#!/usr/bin/env python
"""Test threshold optimizers."""
# pylint: disable=no-member, too-many-instance-attributes
from __future__ import print_function

import math
import os.path as path
import sys
import timeit
import unittest

import numpy as np

sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))


class TestThreshopt(unittest.TestCase):
    """Threshold optimizers working."""

    def setUp(self):
        """Set up fixture."""
        import forpy
        np.random.seed(1)
        self.dtamat = np.array(range(150), dtype=np.float32).reshape((30, 5))
        self.anmat = np.array(range(60), dtype=np.uint32).reshape((30, 2))
        self.anmat_reg = np.array(range(60), dtype=np.float32).reshape((30, 2))
        timedta = np.random.normal(size=(4, 10000)).astype(np.float64)
        timeant = np.random.randint(0, 20, size=(10000, 1)).astype(np.uint32)
        self.anmat[:15, :] = 0
        self.anmat_reg[:15, :] = 0.
        self.weights = [0.] * 13 + np.random.randint(
            0, 10, size=(17, )).tolist()
        self.pdp = forpy.FastDProv(
            np.ascontiguousarray(self.dtamat.T),
            np.ascontiguousarray(self.anmat[:, 0:1]))
        self.pdp_w = forpy.FastDProv(
            np.ascontiguousarray(self.dtamat.T),
            np.ascontiguousarray(self.anmat[:, 0:1]), self.weights)
        self.pdp_reg = forpy.FastDProv(
            np.ascontiguousarray(self.dtamat.T),
            np.ascontiguousarray(self.anmat_reg))
        self.pdp_reg_w = forpy.FastDProv(
            np.ascontiguousarray(self.dtamat.T),
            np.ascontiguousarray(self.anmat_reg), self.weights)
        self.pdp_tim = forpy.FastDProv(timedta, timeant)

    def test_regression(self):
        """Test regression threshold optimization."""
        import forpy
        rto = forpy.RegressionOpt()
        self.assertEqual(rto.supports_weights(), False)
        rto.check_annotations(self.pdp_reg)
        full_entropy = rto.full_entropy(self.pdp_reg)
        self.assertEqual(full_entropy, 1064.9166259765625)
        res = rto.optimize(self.pdp_reg, 0)
        self.assertEqual(res.split_idx, 15)
        self.assertEqual(res.thresh, 72.5)
        self.assertTrue(res.gain > 0.)
        self.assertTrue(res.valid)
        # Random selection.
        rto = forpy.RegressionOpt(5)
        full_entropy = rto.full_entropy(self.pdp_reg)
        self.assertEqual(full_entropy, 1064.9166259765625)
        res = rto.optimize(self.pdp_reg, 0)
        self.assertTrue(res.split_idx > 0)
        self.assertTrue(res.thresh > 0.)
        self.assertTrue(res.gain > 0.)
        self.assertTrue(res.valid)

    def test_reg_weights(self):
        """Test FastClassOpt with weights."""
        import forpy
        rto = forpy.RegressionOpt()
        rto.check_annotations(self.pdp_reg_w)
        rto.full_entropy(self.pdp_reg_w)
        self.assertAlmostEqual(
            rto.full_entropy(self.pdp_reg_w), 637.6682739257812)
        res = rto.optimize(self.pdp_reg_w, 0)
        self.assertEqual(res.split_idx, 15)
        self.assertEqual(res.thresh, 72.5)
        self.assertTrue(np.abs(res.gain - 37492.359375) < 0.1)
        self.assertEqual(res.valid, True)

    def test_classification(self):
        """Test classification threshold optimization."""
        import forpy
        # Best split.
        rto = forpy.ClassificationOpt()
        self.assertEqual(rto.supports_weights(), False)
        rto.check_annotations(self.pdp)
        self.assertEqual(rto.n_classes, 16)
        self.assertEqual(rto.true_max_class, 58)
        self.assertTrue(
            np.all(
                np.sort(np.unique(rto.class_translation)) ==
                [0] + list(range(30, 60, 2))))
        self.assertEqual(rto.full_entropy(self.pdp), 0.7333333492279053)
        res = rto.optimize(self.pdp, 0)
        self.assertEqual(res.split_idx, 15)
        self.assertEqual(res.thresh, 72.5)
        self.assertAlmostEqual(res.gain, 0.2666666805744171)
        self.assertEqual(res.valid, True)
        # Entropies.
        for entr in [
                forpy.ShannonEntropy(),
                forpy.InducedEntropy(2.),
                forpy.InducedEntropy(3.),
                forpy.TsallisEntropy(2.),
                forpy.TsallisEntropy(3.),
                forpy.RenyiEntropy(1.01)
        ]:
            rto = forpy.ClassificationOpt(entropy_function=entr)
            rto.check_annotations(self.pdp)
            res = rto.optimize(self.pdp, 0)
            print(entr, res.split_idx, res.thresh)
            self.assertEqual(res.split_idx, 15)
            self.assertEqual(res.thresh, 72.5)
            self.assertEqual(res.valid, True)
            self.assertTrue(res.gain > 0. and not math.isnan(res.gain)
                            and res.gain != float("Inf"))

        rto = forpy.ClassificationOpt(
            entropy_function=forpy.ClassificationError())
        rto.check_annotations(self.pdp)
        res = rto.optimize(self.pdp, 0)
        self.assertIn(res.split_idx, [15, 27])
        self.assertEqual(res.valid, True)
        self.assertTrue(res.gain > 0. and not math.isnan(res.gain)
                        and res.gain != float("Inf"))

    def test_class_weights(self):
        """Test ClassificationOpt with weights."""
        import forpy
        rto = forpy.ClassificationOpt()
        rto.check_annotations(self.pdp_w)
        self.assertEqual(rto.n_classes, 16)
        self.assertEqual(rto.true_max_class, 58)
        self.assertTrue(
            np.all(
                np.sort(np.unique(rto.class_translation)) ==
                [0] + list(range(30, 60, 2))))
        rto.full_entropy(self.pdp_w)
        self.assertAlmostEqual(
            rto.full_entropy(self.pdp_w), 0.9028487801551819)
        res = rto.optimize(self.pdp_w, 0)
        self.assertEqual(res.split_idx, 15)
        self.assertEqual(res.thresh, 72.5)
        self.assertAlmostEqual(res.gain, 0.14957934617996216, places=6)
        self.assertEqual(res.valid, True)

    def test_fast_class_opt(self):
        """Test FastClassOpt."""
        import forpy
        rto = forpy.FastClassOpt()
        rto.check_annotations(self.pdp)
        self.assertEqual(rto.n_classes, 16)
        self.assertEqual(rto.true_max_class, 58)
        self.assertTrue(
            np.all(
                np.sort(np.unique(rto.class_translation)) ==
                [0] + list(range(30, 60, 2))))
        self.assertEqual(rto.full_entropy(self.pdp), 0.7333333492279053)
        res = rto.optimize(self.pdp, 0)
        self.assertEqual(res.split_idx, 15)
        self.assertEqual(res.thresh, 72.5)
        self.assertEqual(res.gain, 0.2666666805744171)
        self.assertEqual(res.valid, True)

    def test_fco_weights(self):
        """Test FastClassOpt with weights."""
        import forpy
        rto = forpy.FastClassOpt()
        rto.check_annotations(self.pdp_w)
        self.assertEqual(rto.n_classes, 16)
        self.assertEqual(rto.true_max_class, 58)
        self.assertTrue(
            np.all(
                np.sort(np.unique(rto.class_translation)) ==
                [0] + list(range(30, 60, 2))))
        rto.full_entropy(self.pdp_w)
        self.assertEqual(rto.full_entropy(self.pdp_w), 0.9028487801551819)
        res = rto.optimize(self.pdp_w, 0)
        self.assertEqual(res.split_idx, 15)
        self.assertEqual(res.thresh, 72.5)
        self.assertEqual(res.gain, 0.14957934617996216)
        self.assertEqual(res.valid, True)

    def test_timings(self):
        """Test speed."""
        import forpy
        treg = forpy.ClassificationOpt(
            entropy_function=forpy.InducedEntropy(3))
        treg.check_annotations(self.pdp_tim)
        res = timeit.timeit(lambda: treg.optimize(self.pdp_tim, 0), number=100)
        print(
            'classification threshold optimizer benchmark (induced) timing: %fs'
            % (res / 100.))
        treg = forpy.ClassificationOpt(entropy_function=forpy.ShannonEntropy())
        treg.check_annotations(self.pdp_tim)
        res = timeit.timeit(lambda: treg.optimize(self.pdp_tim, 0), number=100)
        print(
            'classification threshold optimizer benchmark (shannon) timing: %fs'
            % (res / 100.))
        treg = forpy.FastClassOpt()
        treg.check_annotations(self.pdp_tim)
        res = timeit.timeit(lambda: treg.optimize(self.pdp_tim, 0), number=100)
        print(
            'classification threshold optimizer benchmark (fast) timing: %fs' %
            (res / 100.))


if __name__ == '__main__':
    unittest.main()
