#!/usr/bin/env python2
import numpy as np
import timeit
import math
import unittest
import sys
import os.path as path
sys.path.insert(0,
                path.join(path.dirname(__file__),
                          '..', '..'))
import forpy

class TestThreshopt(unittest.TestCase):

    def test_regression(self):
        """Test regression threshold optimization."""
        # Best split.
        rto = forpy.RegressionThresholdOptimizer()
        self.assertEqual(rto.supports_weights(), False)
        dtamat = np.array(range(150), dtype=np.float32).reshape((30, 5))
        ftmat = np.array(range(30), dtype=np.float32).reshape((30, 1))
        anmat = np.array(range(60), dtype=np.float32).reshape((30, 2))
        anmat[:15, :] = 0.
        pdp = forpy.PlainDataProvider(dtamat, anmat)
        rto.check_annotations(pdp)
        self.assertEqual(rto.check_for_early_stop(anmat, 0), False)
        self.assertRaises(RuntimeError,
                          lambda: rto.prepare_for_optimizing(0, 0))
        self.assertRaises(RuntimeError,
                          lambda: rto.prepare_for_optimizing(0, -1))
        rto.prepare_for_optimizing(0, 1)
        res = rto.optimize(np.asfortranarray(dtamat), ftmat, anmat)
        self.assertEqual(res[0][0], 14.5)
        self.assertEqual(res[1], forpy.EThresholdSelection.LessOnly)
        self.assertTrue(res[5])
        # Random selection.
        rto = forpy.RegressionThresholdOptimizer(5)
        rto.prepare_for_optimizing(0, 1)
        res = rto.optimize(np.asfortranarray(dtamat), ftmat, anmat)
        self.assertTrue(res[4] > 0.)
        self.assertTrue(res[5])
        # Entropies and regressors.
        for entr in [forpy.ClassificationError(),
                     forpy.ShannonEntropy(),
                     forpy.InducedEntropy(2.),
                     forpy.InducedEntropy(3.),
                     forpy.TsallisEntropy(2.),
                     forpy.TsallisEntropy(3.),
                     forpy.RenyiEntropy(2.)]:
            for regr in [forpy.ConstantRegressor(),
                         forpy.LinearRegressor()]:
                rto = forpy.RegressionThresholdOptimizer(
                    entropy_function=entr,
                    regressor_template=regr)
                rto.prepare_for_optimizing(0, 1)
                res = rto.optimize(np.asfortranarray(dtamat), ftmat, anmat)
                self.assertEqual(res[0][0], 14.5)
                self.assertEqual(res[1],
                                 forpy.EThresholdSelection.LessOnly)
                self.assertTrue(res[4] > 0. and
                                not math.isnan(res[4]) and
                                res[4] != float("Inf"))
                self.assertTrue(res[5])
        timedta = np.random.normal(size=(10000, 4))
        timeant = np.random.normal(size=(10000, 1))
        treg = forpy.RegressionThresholdOptimizer()
        treg.prepare_for_optimizing(0, 1)
        def tfun():
            treg.optimize(np.asfortranarray(timedta), timedta[:, 0:1].copy(), timedta)
        res = timeit.timeit(tfun, number=100)
        print 'fast regression threshold optimizer benchmark timing: %fs' % (res / 100.)
        treg = forpy.RegressionThresholdOptimizer(
            10,
            regressor_template=forpy.LinearRegressor())
        treg.prepare_for_optimizing(0, 1)
        def tfun():
            treg.optimize(np.asfortranarray(timedta), timedta[:, 0:1].copy(), timedta)
        res = timeit.timeit(tfun, number=10)
        print 'linear regression threshold optimizer benchmark timing: %fs' % (res / 10.)

    def test_classification(self):
        """Test classification threshold optimization."""
        # Best split.
        rto = forpy.ClassificationThresholdOptimizer()
        self.assertEqual(rto.supports_weights(), True)
        dtamat = np.array(range(150), dtype=np.float32).reshape((30, 5))
        ftmat = np.array(range(30), dtype=np.float32).reshape((30, 1))
        anmat = np.array(range(60), dtype=np.uint32).reshape((30, 2))
        anmat[:15, :] = 0.
        pdp = forpy.PlainDataProvider(dtamat, anmat[:, 0:1])
        rto.check_annotations(pdp)
        self.assertEqual(rto.n_classes, 59)
        self.assertEqual(rto.check_for_early_stop(anmat, 0), False)
        rto.prepare_for_optimizing(0, 1)
        res = rto.optimize(np.asfortranarray(dtamat), ftmat, anmat)
        self.assertEqual(res[0][0], 14.5)
        self.assertEqual(res[1], forpy.EThresholdSelection.LessOnly)
        self.assertTrue(res[5])
        # Entropies.
        for entr in [forpy.ClassificationError(),
                     forpy.ShannonEntropy(),
                     forpy.InducedEntropy(2.),
                     forpy.InducedEntropy(3.),
                     forpy.TsallisEntropy(2.),
                     forpy.TsallisEntropy(3.),
                     forpy.RenyiEntropy(1.01)]:
            rto = forpy.ClassificationThresholdOptimizer(
                gain_calculator=forpy.EntropyGain(entr))
            rto.check_annotations(pdp)
            rto.prepare_for_optimizing(0, 1)
            res = rto.optimize(np.asfortranarray(dtamat), ftmat, anmat[:, :])
            self.assertEqual(res[0][0], 14.5)
            self.assertEqual(res[1],
                             forpy.EThresholdSelection.LessOnly)
            self.assertTrue(res[4] > 0. and
                            not math.isnan(res[4]) and
                            res[4] != float("Inf"))
            self.assertTrue(res[5])
        timedta = np.random.normal(size=(10000, 4)).astype("float64")
        timeant = np.random.randint(0, 20, size=(10000, 1)).astype("uint32")
        treg = forpy.ClassificationThresholdOptimizer(
            gain_calculator=forpy.EntropyGain(forpy.InducedEntropy(2.)))
        treg.prepare_for_optimizing(0, 1)
        pdp = forpy.PlainDataProvider(timedta, timeant)
        treg.check_annotations(pdp)
        def tfun():
            treg.optimize(np.asfortranarray(timedta), timedta[:, 0:1].copy(), timeant)
        res = timeit.timeit(tfun, number=100)
        print 'classification threshold optimizer benchmark (induced) timing: %fs' % (res / 100.)
        treg = forpy.ClassificationThresholdOptimizer(
            gain_calculator=forpy.EntropyGain(forpy.ShannonEntropy()))
        treg.prepare_for_optimizing(0, 1)
        treg.check_annotations(pdp)
        def tfun():
            treg.optimize(np.asfortranarray(timedta), timedta[:, 0:1].copy(), timeant)
        res = timeit.timeit(tfun, number=100)
        print 'classification threshold optimizer benchmark (shannon) timing: %fs' % (res / 100.)

if __name__ == '__main__':
    unittest.main()
