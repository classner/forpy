#!/usr/bin/env python2
import numpy as np
import unittest
import sys
import os.path as path
sys.path.insert(0,
                path.join(path.dirname(__file__),
                          '..', '..'))
import forpy

class TestLeafs(unittest.TestCase):

    def test_classification(self):
        """Test classification leaf."""
        pdp = forpy.PlainDataProvider(np.ones((10, 5)),
                                      np.zeros((10, 1)))
        cto = forpy.ClassificationThresholdOptimizer()
        cto.check_annotations(pdp)
        cl = forpy.ClassificationLeaf()
        self.assertFalse(cl.needs_data())
        # No n_classes set, check failing methods.
        self.assertRaises(RuntimeError, lambda: cl.make_leaf(0, range(10), pdp))
        self.assertRaises(RuntimeError, lambda: cl.get_result(0))
        self.assertRaises(RuntimeError, lambda: cl.get_result(0))
        # Set classes.
        self.assertTrue(cl.is_compatible_with(pdp))
        self.assertTrue(cl.is_compatible_with(cto))
        self.assertTrue(cl.get_result_columns(), 10)
        self.assertRaises(RuntimeError, lambda: cl.make_leaf(0, [], pdp))
        cl.make_leaf(0, range(10), pdp)
        res = cl.get_result(0)
        self.assertEqual(res[0, 0], 1.)
        self.assertEqual(res.shape, (1, 1))
        self.assertRaises(RuntimeError, lambda: cl.get_result(1))
        # Some data.
        pdp = forpy.PlainDataProvider(np.array(range(10), np.float64),
                                      np.array(range(10), np.uint32))
        cto = forpy.ClassificationThresholdOptimizer()
        cto.check_annotations(pdp)
        cl = forpy.ClassificationLeaf()
        cl.is_compatible_with(pdp)
        cl.is_compatible_with(cto)
        cl.make_leaf(1, range(5), pdp)
        res = cl.get_result(1)
        self.assertTrue(np.all(res[0, :5] == 0.2))
        self.assertTrue(np.all(res[0, 5:] == 0.))
        self.assertRaises(RuntimeError, lambda: cl.get_result([res, res * 2],
                                                              [0.5]))
        res = cl.get_result([res, res * 2], [3., 1.])
        self.assertTrue(np.all(np.isclose(res[0, :5], 0.25)))
        self.assertTrue(np.all(res[0, 5:] == 0.))
        cl.__repr__()

    def test_regression(self):
        """Test regression leaf."""
        pdp = forpy.PlainDataProvider(np.ones((10, 5)),
                                      np.zeros((10, 1)))
        cto = forpy.ClassificationThresholdOptimizer()
        cto.check_annotations(pdp)
        cl = forpy.RegressionLeaf()
        self.assertTrue(cl.needs_data())
        # No data provider checked yet, check failing methods.
        self.assertRaises(RuntimeError, lambda: cl.make_leaf(0, range(10), pdp))
        self.assertRaises(RuntimeError, lambda: cl.get_result(0))
        self.assertRaises(RuntimeError, lambda: cl.get_result(0))
        # Set classes.
        self.assertTrue(cl.is_compatible_with(pdp))
        self.assertTrue(cl.is_compatible_with(cto))
        self.assertTrue(cl.get_result_columns(), 2)
        self.assertRaises(RuntimeError, lambda: cl.make_leaf(0, [], pdp))
        cl.make_leaf(0, range(10), pdp)
        res = cl.get_result(0, np.ones((1, 5)))
        self.assertTrue(np.allclose(res[0, 0], 0.))
        self.assertTrue(np.allclose(res[0, 1], 0.))
        self.assertEqual(res.shape, (1, 2))
        self.assertRaises(RuntimeError, lambda: cl.get_result(1))
        # Some data.
        pdp = forpy.PlainDataProvider(np.array(range(10), np.float64).reshape((5, 2)),
                                      np.array(range(10), np.uint32).reshape((5, 2)))
        cto = forpy.RegressionThresholdOptimizer()
        cto.check_annotations(pdp)
        cl = forpy.RegressionLeaf()
        cl.is_compatible_with(pdp)
        cl.is_compatible_with(cto)
        cl.make_leaf(1, range(5), pdp)
        res = cl.get_result(1, [[0., 1.]])
        self.assertTrue(np.allclose(res, [0., 1., 0., 0.]))
        res = cl.get_result(1, [[4., 5.]])
        self.assertTrue(np.allclose(res, [4., 5., 0., 0.]))
        self.assertRaises(RuntimeError, lambda: cl.get_result([res, res * 2],
                                                              [0.5]))
        res = cl.get_result([res, res * 2], [3., 1.])
        # Checked in Python.
        self.assertTrue(np.all(np.isclose(res, [5., 6.25, 3., 4.6875])))
        # Summary mode 1.
        pdp = forpy.PlainDataProvider(np.array(range(10), np.float64).reshape((5, 2)),
                                      np.array(range(10), np.uint32).reshape((5, 2)),)
        cto = forpy.RegressionThresholdOptimizer()
        cto.check_annotations(pdp)
        cl = forpy.RegressionLeaf(summary_mode=1)
        cl.is_compatible_with(pdp)
        cl.is_compatible_with(cto)
        cl.make_leaf(1, range(5), pdp)
        res = cl.get_result(1, [[0., 1.]])
        self.assertTrue(np.allclose(res, [0., 1., 0., 0.]))
        res = cl.get_result(1, [[4., 5.]])
        self.assertTrue(np.allclose(res, [4., 5., 0., 0.]))
        self.assertRaises(RuntimeError, lambda: cl.get_result([res, res * 2],
                                                              [0.5]))
        res = cl.get_result([res, res * 2], [3., 1.])
        # Checked in Python.
        self.assertTrue(np.all(np.isclose(res, [5., 6.25, 0., 0.])))
        # Summary mode 2.
        pdp = forpy.PlainDataProvider(np.array(range(10), np.float64).reshape((5, 2)),
                                      np.array(range(10), np.uint32).reshape((5, 2)),)
        cto = forpy.RegressionThresholdOptimizer()
        cto.check_annotations(pdp)
        cl = forpy.RegressionLeaf(summary_mode=2)
        cl.is_compatible_with(pdp)
        cl.is_compatible_with(cto)
        cl.make_leaf(1, range(5), pdp)
        res = cl.get_result(1, [[0., 1.]])
        self.assertTrue(np.allclose(res, [0., 1., 0., 0.]))
        res = cl.get_result(1, [[4., 5.]])
        self.assertTrue(np.allclose(res, [4., 5., 0., 0.]))
        self.assertRaises(RuntimeError, lambda: cl.get_result([res, res * 2],
                                                              [0.5]))
        res = cl.get_result([res, res * 2], [3., 1.])
        # Checked in Python.
        self.assertTrue(np.all(np.isclose(res,
                                          [4., 5., 0., 0., 8., 10., 0., 0.])))
        # Feature selection.
        dta = np.array(range(10), np.float64).reshape((5, 2))
        dta[:, 1] = 0
        pdp = forpy.PlainDataProvider(dta,
                                      np.array(range(10), np.uint32).reshape((5, 2)))
        cto = forpy.RegressionThresholdOptimizer()
        cto.check_annotations(pdp)
        cl = forpy.RegressionLeaf(regression_input_dim=1,
                                  selections_to_try=2)
        cl.is_compatible_with(pdp)
        cl.is_compatible_with(cto)
        cl.make_leaf(1, range(5), pdp)
        res = cl.get_result(1, [[0., 1.]])
        self.assertTrue(np.allclose(res, [0., 1., 0., 0.]))
        res = cl.get_result(1, [[4., 5.]])
        self.assertTrue(np.allclose(res, [4., 5., 0., 0.]))
        cl.__repr__()

    def test_regression_float(self):
        """Test regression leaf for floats."""
        pdp = forpy.PlainDataProvider(np.ones((10, 5), dtype=np.float32),
                                      np.zeros((10, 1), dtype=np.float32))
        cto = forpy.ClassificationThresholdOptimizer()
        cto.check_annotations(pdp)
        cl = forpy.RegressionLeaf()
        self.assertTrue(cl.needs_data())
        # No data provider checked yet, check failing methods.
        self.assertRaises(RuntimeError, lambda: cl.make_leaf(0, range(10), pdp))
        self.assertRaises(RuntimeError, lambda: cl.get_result(0))
        self.assertRaises(RuntimeError, lambda: cl.get_result(0))
        # Set classes.
        self.assertTrue(cl.is_compatible_with(pdp))
        self.assertTrue(cl.is_compatible_with(cto))
        self.assertTrue(cl.get_result_columns(), 2)
        self.assertRaises(RuntimeError, lambda: cl.make_leaf(0, [], pdp))
        cl.make_leaf(0, range(10), pdp)
        res = cl.get_result(0, np.ones((1, 5)))
        self.assertTrue(np.allclose(res[0, 0], 0.))
        self.assertTrue(np.allclose(res[0, 1], 0.))
        self.assertEqual(res.shape, (1, 2))
        self.assertRaises(RuntimeError, lambda: cl.get_result(1))
        # Some data.
        pdp = forpy.PlainDataProvider(np.array(range(10), np.float32).reshape((5, 2)),
                                      np.array(range(10), np.float32).reshape((5, 2)))
        cto = forpy.RegressionThresholdOptimizer()
        cto.check_annotations(pdp)
        cl = forpy.RegressionLeaf()
        cl.is_compatible_with(pdp)
        cl.is_compatible_with(cto)
        cl.make_leaf(1, range(5), pdp)
        res = cl.get_result(1, [[0., 1.]])
        self.assertTrue(np.allclose(res, [0., 1., 0., 0.], atol=1e-5))
        res = cl.get_result(1, [[4., 5.]])
        self.assertTrue(np.allclose(res, [4., 5., 0., 0.]))
        self.assertRaises(RuntimeError, lambda: cl.get_result([res, res * 2],
                                                              [0.5]))
        res = cl.get_result([res, res * 2], [3., 1.])
        # Checked in Python.
        self.assertTrue(np.all(np.isclose(res, [5., 6.25, 3., 4.6875])))
        # Summary mode 1.
        pdp = forpy.PlainDataProvider(np.array(range(10), np.float32).reshape((5, 2)),
                                      np.array(range(10), np.float32).reshape((5, 2)),)
        cto = forpy.RegressionThresholdOptimizer()
        cto.check_annotations(pdp)
        cl = forpy.RegressionLeaf(summary_mode=1)
        cl.is_compatible_with(pdp)
        cl.is_compatible_with(cto)
        cl.make_leaf(1, range(5), pdp)
        res = cl.get_result(1, [[0., 1.]])
        self.assertTrue(np.allclose(res, [0., 1., 0., 0.], atol=1e-5))
        res = cl.get_result(1, [[4., 5.]])
        self.assertTrue(np.allclose(res, [4., 5., 0., 0.]))
        self.assertRaises(RuntimeError, lambda: cl.get_result([res, res * 2],
                                                              [0.5]))
        res = cl.get_result([res, res * 2], [3., 1.])
        # Checked in Python.
        self.assertTrue(np.all(np.isclose(res, [5., 6.25, 0., 0.])))
        # Summary mode 2.
        pdp = forpy.PlainDataProvider(np.array(range(10), np.float32).reshape((5, 2)),
                                      np.array(range(10), np.float32).reshape((5, 2)),)
        cto = forpy.RegressionThresholdOptimizer()
        cto.check_annotations(pdp)
        cl = forpy.RegressionLeaf(summary_mode=2)
        cl.is_compatible_with(pdp)
        cl.is_compatible_with(cto)
        cl.make_leaf(1, range(5), pdp)
        res = cl.get_result(1, [[0., 1.]])
        self.assertTrue(np.allclose(res, [0., 1., 0., 0.], atol=1e-5))
        res = cl.get_result(1, [[4., 5.]])
        self.assertTrue(np.allclose(res, [4., 5., 0., 0.]))
        self.assertRaises(RuntimeError, lambda: cl.get_result([res, res * 2],
                                                              [0.5]))
        res = cl.get_result([res, res * 2], [3., 1.])
        # Checked in Python.
        self.assertTrue(np.all(np.isclose(res,
                                          [4., 5., 0., 0., 8., 10., 0., 0.])))
        # Feature selection.
        dta = np.array(range(10), np.float32).reshape((5, 2))
        dta[:, 1] = 0
        pdp = forpy.PlainDataProvider(dta,
                                      np.array(range(10), np.float32).reshape((5, 2)))
        cto = forpy.RegressionThresholdOptimizer()
        cto.check_annotations(pdp)
        cl = forpy.RegressionLeaf(regression_input_dim=1,
                                  selections_to_try=2)
        cl.is_compatible_with(pdp)
        cl.is_compatible_with(cto)
        cl.make_leaf(1, range(5), pdp)
        res = cl.get_result(1, [[0., 1.]])
        self.assertTrue(np.allclose(res, [0., 1., 0., 0.], atol=1e-5))
        res = cl.get_result(1, [[4., 5.]])
        self.assertTrue(np.allclose(res, [4., 5., 0., 0.]))
        cl.__repr__()


    def test_regression_constant(self):
        """Test regression leaf for constant regressor."""
        pdp = forpy.PlainDataProvider(np.ones((10, 5), dtype=np.float32),
                                      np.zeros((10, 1), dtype=np.float32))
        cto = forpy.ClassificationThresholdOptimizer()
        cto.check_annotations(pdp)
        cl = forpy.RegressionLeaf(regressor_template=forpy.ConstantRegressor())
        self.assertTrue(cl.needs_data())
        # No data provider checked yet, check failing methods.
        self.assertRaises(RuntimeError, lambda: cl.make_leaf(0, range(10), pdp))
        self.assertRaises(RuntimeError, lambda: cl.get_result(0))
        self.assertRaises(RuntimeError, lambda: cl.get_result(0))
        # Set classes.
        self.assertTrue(cl.is_compatible_with(pdp))
        self.assertTrue(cl.is_compatible_with(cto))
        self.assertTrue(cl.get_result_columns(), 2)
        self.assertRaises(RuntimeError, lambda: cl.make_leaf(0, [], pdp))
        cl.make_leaf(0, range(10), pdp)
        res = cl.get_result(0, np.ones((1, 5)))
        self.assertTrue(np.allclose(res[0, 0], 0.))
        self.assertTrue(np.allclose(res[0, 1], 0.))
        self.assertEqual(res.shape, (1, 2))
        self.assertRaises(RuntimeError, lambda: cl.get_result(1))
        # Some data.
        pdp = forpy.PlainDataProvider(np.array(range(10), np.float32).reshape((5, 2)),
                                      np.array(range(10), np.float32).reshape((5, 2)))
        cto = forpy.RegressionThresholdOptimizer()
        cto.check_annotations(pdp)
        cl = forpy.RegressionLeaf(regressor_template=forpy.ConstantRegressor())
        cl.is_compatible_with(pdp)
        cl.is_compatible_with(cto)
        cl.make_leaf(1, range(5), pdp)
        res = cl.get_result(1, [[0., 1.]])
        self.assertTrue(np.allclose(res, [4., 5., 8., 8.], atol=1e-5))
        res = cl.get_result(1, [[4., 5.]])
        self.assertTrue(np.allclose(res, [4., 5., 8., 8.], atol=1e-5))
        res = cl.get_result(1)
        self.assertTrue(np.allclose(res, [4., 5., 8., 8.], atol=1e-5))
        cl.__repr__()

if __name__ == '__main__':
    unittest.main()
