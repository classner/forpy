#!/usr/bin/env python2
import unittest
import timeit
import cPickle as pickle
import numpy as np
import sys
import os.path as path
sys.path.insert(0, path.join(path.dirname(__file__),
                             '..', '..'))
import forpy

class TestRegression(unittest.TestCase):
          
  def test_constant(self):
      """Test constant regressor."""
      regr = forpy.ConstantRegressor()
      self.assertTrue(not regr.needs_input_data())
      self.assertTrue(regr.has_constant_prediction_covariance())
      self.assertTrue(not regr.has_solution())
      self.assertEqual(regr.index_interval, (-1L, -1L))
      with self.assertRaises(RuntimeError):
          regr.index_interval = [0, 1]
      self.assertRaises(RuntimeError, lambda: regr.get_kernel_dimension())
      self.assertTrue(not regr.frozen)
      self.assertEqual(regr.n_samples, 0)
      regr.__repr__()
      for meth in [regr.predict,
                   regr.predict_covar,
                   regr.get_constant_prediction_covariance,
                   regr.get_input_dimension,
                   regr.get_annotation_dimension]:
          with self.assertRaises(RuntimeError):
              meth()
      self.assertRaises(RuntimeError, lambda: pickle.dumps(regr, 2))
      # Initialize.
      annots = np.array(range(30)).reshape((10, 3))
      regr.initialize(np.array(range(50)).reshape((10, 5)),
                      np.array(range(30)).reshape((10, 3)))
      self.assertTrue(regr.has_solution)
      self.assertEqual(regr.index_interval, (0L, 10L))
      regr.index_interval = [0, 1]
      self.assertEqual(regr.index_interval, (0L, 1L))
      regr.index_interval = [0, 10]
      self.assertEqual(regr.index_interval, (0L, 10L))
      with self.assertRaises(RuntimeError):
          regr.index_interval = [-1, 3]
      with self.assertRaises(RuntimeError):
          regr.index_interval = [0, 100]
      self.assertEqual(regr.get_kernel_dimension(), 0)
      self.assertTrue(not regr.frozen)
      self.assertEqual(regr.n_samples, 10)
      regr.__repr__()
      self.assertTrue(np.all(np.isclose(regr.predict(),
                                        np.mean(annots, axis=0))))
      self.assertTrue(np.all(regr.predict_covar()[0] == regr.predict()))
      self.assertEqual(regr.predict_covar()[1][0, 0],
                       np.mean((annots - regr.predict()) ** 2))
      self.assertTrue(np.all(regr.predict_covar()[1] ==
                             regr.get_constant_prediction_covariance()))
      self.assertEqual(regr.get_input_dimension(), 5)
      self.assertEqual(regr.get_annotation_dimension(), 3)
      self.assertRaises(RuntimeError, lambda: pickle.dumps(regr, 2))
      # Freeze.
      regr.freeze()
      self.assertTrue(regr.has_solution)
      self.assertEqual(regr.index_interval, (-1L, -1L))
      with self.assertRaises(RuntimeError):
          regr.index_interval = [0, 1]
      self.assertEqual(regr.get_kernel_dimension(), 0)
      self.assertTrue(regr.frozen)
      self.assertEqual(regr.n_samples, 10)
      regr.__repr__()
      self.assertTrue(np.all(np.isclose(regr.predict(),
                                        np.mean(annots, axis=0))))
      self.assertTrue(np.all(regr.predict_covar()[0] == regr.predict()))
      self.assertEqual(regr.predict_covar()[1][0, 0],
                       np.mean((annots - regr.predict()) ** 2))
      self.assertTrue(np.all(regr.predict_covar()[1] ==
                             regr.get_constant_prediction_covariance()))
      self.assertEqual(regr.get_input_dimension(), 5)
      self.assertEqual(regr.get_annotation_dimension(), 3)
      # Pickle.
      recr = pickle.loads(pickle.dumps(regr, 2))
      self.assertTrue(recr.has_solution())
      self.assertEqual(recr.index_interval, (-1L, -1L))
      with self.assertRaises(RuntimeError):
          recr.index_interval = [0, 1]
      self.assertEqual(recr.get_kernel_dimension(), 0)
      self.assertTrue(recr.frozen)
      self.assertEqual(recr.n_samples, 10)
      recr.__repr__()
      self.assertTrue(np.all(np.isclose(recr.predict(),
                                        np.mean(annots, axis=0))))
      self.assertTrue(np.all(recr.predict_covar()[0] == recr.predict()))
      self.assertEqual(recr.predict_covar()[1][0, 0],
                       np.mean((annots - regr.predict()) ** 2))
      self.assertTrue(np.all(recr.predict_covar()[1] ==
                             recr.get_constant_prediction_covariance()))
      self.assertEqual(recr.get_input_dimension(), 5)
      self.assertEqual(recr.get_annotation_dimension(), 3)
      # Equality.
      self.assertEqual(recr, regr)
      # Timing.
      timedta = np.random.normal(size=(10000, 7))
      timeann = np.random.normal(size=(10000, 9))
      treg = forpy.ConstantRegressor()
      treg.initialize(timedta, timeann)
      def tfun():
          for i in range(10000):
              treg.index_interval = [0, i]
      res = timeit.timeit(tfun, number=100)
      print 'constant regressor benchmark timing: %fs' % (res / 100.)

  def test_linear(self):
      """Test the linear regressor."""
      regr = forpy.LinearRegressor()
      self.assertTrue(regr.needs_input_data())
      self.assertTrue(not regr.has_constant_prediction_covariance())
      self.assertTrue(not regr.has_solution())
      self.assertEqual(regr.index_interval, (-1L, -1L))
      with self.assertRaises(RuntimeError):
          regr.index_interval = [0, 1]
      self.assertRaises(RuntimeError, lambda: regr.get_kernel_dimension())
      self.assertTrue(not regr.frozen)
      self.assertEqual(regr.n_samples, 0)
      regr.__repr__()
      for meth in [regr.predict,
                   regr.predict_covar,
                   regr.get_constant_prediction_covariance,
                   regr.get_input_dimension,
                   regr.get_annotation_dimension]:
          with self.assertRaises(RuntimeError):
              meth()
      self.assertRaises(RuntimeError, lambda: pickle.dumps(regr, 2))
      # Initialize.
      inputs = np.array(range(50)).reshape((10, 5))
      annots = np.array(range(30)).reshape((10, 3))
      regr.initialize(np.array(range(50)).reshape((10, 5)),
                      np.array(range(30)).reshape((10, 3)))
      self.assertTrue(regr.has_solution())
      self.assertEqual(regr.index_interval, (0L, 10L))
      regr.index_interval = [0, 1]
      self.assertEqual(regr.index_interval, (0L, 1L))
      regr.index_interval = [0, 10]
      self.assertEqual(regr.index_interval, (0L, 10L))
      with self.assertRaises(RuntimeError):
          regr.index_interval = [-1, 3]
      with self.assertRaises(RuntimeError):
          regr.index_interval = [0, 100]
      cr = forpy.LinearRegressor()
      cr.initialize(np.zeros((10, 3)), np.zeros((10, 3)))
      self.assertEqual(cr.get_kernel_dimension(), 0)
      self.assertTrue(not regr.frozen)
      self.assertEqual(regr.n_samples, 10)
      regr.__repr__()
      self.assertRaises(RuntimeError, lambda: regr.predict())
      self.assertTrue(np.all(np.isclose(regr.predict(np.mean(inputs, axis=0)),
                                        np.mean(annots, axis=0))))
      self.assertTrue(np.all(regr.predict_covar(np.mean(inputs, axis=0))[0] ==
                             regr.predict(np.mean(inputs, axis=0))))
      self.assertRaises(RuntimeError,
                        lambda: regr.get_constant_prediction_covariance())
      self.assertEqual(regr.get_input_dimension(), 5)
      self.assertEqual(regr.get_annotation_dimension(), 3)
      self.assertRaises(RuntimeError, lambda: pickle.dumps(regr, 2))
      # Freeze.
      regr.freeze()
      self.assertTrue(regr.has_solution)
      self.assertEqual(regr.index_interval, (-1L, -1L))
      with self.assertRaises(RuntimeError):
          regr.index_interval = [0, 1]
      self.assertEqual(regr.get_kernel_dimension(), 1)
      self.assertTrue(regr.frozen)
      self.assertEqual(regr.n_samples, 10)
      regr.__repr__()
      self.assertTrue(np.all(np.isclose(regr.predict(np.mean(inputs, axis=0)),
                                        np.mean(annots, axis=0))))
      self.assertTrue(np.all(regr.predict_covar(np.mean(inputs, axis=0))[0] ==
                             regr.predict(np.mean(inputs, axis=0))))
      self.assertRaises(RuntimeError,
                        lambda: regr.get_constant_prediction_covariance())
      self.assertEqual(regr.get_input_dimension(), 5)
      self.assertEqual(regr.get_annotation_dimension(), 3)
      # Pickle.
      recr = pickle.loads(pickle.dumps(regr, 2))
      self.assertTrue(recr.has_solution())
      self.assertEqual(recr.index_interval, (-1L, -1L))
      with self.assertRaises(RuntimeError):
          recr.index_interval = [0, 1]
      self.assertEqual(recr.get_kernel_dimension(), 1)
      self.assertTrue(recr.frozen)
      self.assertEqual(recr.n_samples, 10)
      recr.__repr__()
      self.assertTrue(np.all(np.isclose(regr.predict(np.mean(inputs, axis=0)),
                                        np.mean(annots, axis=0))))
      self.assertTrue(np.all(regr.predict_covar(np.mean(inputs, axis=0))[0] ==
                             regr.predict(np.mean(inputs, axis=0))))
      self.assertRaises(RuntimeError,
                        lambda: regr.get_constant_prediction_covariance())
      self.assertEqual(recr.get_input_dimension(), 5)
      self.assertEqual(recr.get_annotation_dimension(), 3)
      # Equality.
      self.assertEqual(recr, regr)
      # Timing.
      timedta = np.random.normal(size=(1000, 7))
      timeann = np.random.normal(size=(1000, 9))
      treg = forpy.LinearRegressor()
      treg.initialize(timedta, timeann)
      def tfun():
          for i in range(1000):
              treg.index_interval = [0, i]
      res = timeit.timeit(tfun, number=10)
      print 'linear regressor benchmark timing: %fs' % (res / 10.)

if __name__ == '__main__':
    unittest.main()
