#!/usr/bin/env python2
import numpy as np
import unittest
import sys
import os.path as path
sys.path.insert(0,
                path.join(path.dirname(__file__),
                          '..', '..'))
import forpy

class TestTree(unittest.TestCase):

    def test_plain(self):
        tree = forpy.Tree(20, 1, 2,
                          forpy.ThresholdDecider(forpy.ClassificationThresholdOptimizer(), 2),
                          forpy.ClassificationLeaf())
        dta = np.array(range(20), dtype=np.float32).reshape((5, 4))
        annot = np.array(range(5), dtype=np.float32).reshape((5, 1))
        tree.fit(dta, annot)
        results = np.argmax(tree.predict(dta), axis=1)
        self.assertTrue(np.all(results == range(5)))
        self.assertEqual(tree.depth, 3)
        tpath = path.join(path.dirname(__file__),
                          'test.ft')
        tree.save(tpath)
        t2 = forpy.Tree(tpath)
        self.assertTrue(t2 == tree)

        tree = forpy.Tree(20, 4, 8,
                          forpy.ThresholdDecider(forpy.RegressionThresholdOptimizer(), 2),
                          forpy.RegressionLeaf(regressor_template=forpy.ConstantRegressor()))
        dta = np.array(range(20), dtype=np.float32).reshape((10, 2))
        annot = np.array(range(10), dtype=np.float32).reshape((10, 1))
        tree.fit(dta, annot)
        results = tree.predict(dta)
        #print dta, annot
        #print 'res', results
        #print tree.depth
        tree.save(tpath)
        t2 = forpy.Tree(tpath)
        self.assertTrue(t2 == tree)
        results = t2.predict(dta)
        #print dta, annot
        #print 'res', results
        #print tree.depth
        tree = forpy.Tree(20, 4, 8,
                          forpy.ThresholdDecider(forpy.RegressionThresholdOptimizer(), 2),
                          forpy.RegressionLeaf(regressor_template=forpy.LinearRegressor()))
        dta = np.array(range(20), dtype=np.float32).reshape((10, 2))
        annot = np.array(range(10), dtype=np.float32).reshape((10, 1))
        tree.fit(dta, annot)
        results = tree.predict(dta)
        tree.save(tpath)
        t2 = forpy.Tree(tpath)
        self.assertTrue(t2 == tree)
        results = t2.predict(dta)
        #print dta, annot
        #print 'res', results
        #print tree.depth

if __name__ == '__main__':
    unittest.main()
