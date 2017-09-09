#!/usr/bin/env python2
# pylint: disable=no-member, no-self-use
import unittest
import timeit
import sys
import os.path as path
import numpy as np
sys.path.insert(0,
                path.join(path.dirname(__file__),
                          '..', '..'))
import forpy  # pylint: disable=wrong-import-position


class TestTree(unittest.TestCase):

    """Tests for tree performance."""

    def test_plain_shallow(self):
        """Test plain tree prediction speed."""
        tree = forpy.Tree(20, 1, 2,
                          forpy.ThresholdDecider(
                              forpy.ClassificationThresholdOptimizer(), 2),
                          forpy.ClassificationLeaf())
        dta = np.array(range(20), dtype=np.float32).reshape((5, 4))
        annot = np.array(range(5), dtype=np.float32).reshape((5, 1))
        tree.fit(dta, annot)
        tree.enable_fast_prediction()
        results = np.argmax(tree.predict(dta), axis=1)
        self.assertTrue(np.all(results == range(5)))
        self.assertEqual(tree.depth, 3)
        print ("forpy prediction time: " +
               str(timeit.Timer(lambda: tree.predict(dta))
                   .timeit(int(1E5)) / 1E5))

    def test_sklearn_shallow(self):
        """Compare with sklearn."""
        import sklearn.tree as skt
        dta = np.array(range(20), dtype=np.float32).reshape((5, 4))
        annot = np.array(range(5), dtype=np.float32).reshape((5, 1))
        tree = skt.DecisionTreeClassifier()
        tree.fit(dta, annot)
        results = tree.predict(dta)
        self.assertTrue(np.all(results == range(5)))
        self.assertEqual(tree.tree_.max_depth, 4)
        print ("sklearn prediction time: " +
               str(timeit.Timer(lambda: tree.predict(dta))
                   .timeit(int(1E5)) / 1E5))

    def test_plain_shallow_single(self):
        """Test plain tree prediction speed."""
        tree = forpy.Tree(20, 1, 2,
                          forpy.ThresholdDecider(
                              forpy.ClassificationThresholdOptimizer(), 2),
                          forpy.ClassificationLeaf())
        dta = np.array(range(20), dtype=np.float32).reshape((5, 4))
        annot = np.array(range(5), dtype=np.float32).reshape((5, 1))
        tree.fit(dta, annot)
        tree.enable_fast_prediction()
        results = np.argmax(tree.predict(dta), axis=1)
        self.assertTrue(np.all(results == range(5)))
        self.assertEqual(tree.depth, 3)
        print ("forpy prediction time (single): " +
               str(timeit.Timer(lambda: tree.predict(dta[0:1, :]))
                   .timeit(int(1E5)) / 1E5))

    def test_sklearn_shallow_single(self):
        """Compare with sklearn."""
        import sklearn.tree as skt
        dta = np.array(range(20), dtype=np.float32).reshape((5, 4))
        annot = np.array(range(5), dtype=np.float32).reshape((5, 1))
        tree = skt.DecisionTreeClassifier()
        tree.fit(dta, annot)
        results = tree.predict(dta)
        self.assertTrue(np.all(results == range(5)))
        self.assertEqual(tree.tree_.max_depth, 4)
        print ("sklearn prediction time (single): " +
               str(timeit.Timer(lambda: tree.predict(dta[0:1, :]))
                   .timeit(int(1E5)) / 1E5))

    def test_plain_deep(self):
        """Test plain tree prediction speed (deep)."""
        tree = forpy.Tree(20, 1, 2,
                          forpy.ThresholdDecider(
                              forpy.ClassificationThresholdOptimizer(), 2),
                          forpy.ClassificationLeaf())
        np.random.seed(1)
        dta = np.random.normal(size=(1000, 4)).astype("float32")
        annot = np.random.randint(low=0, high=2, size=(1000, 1)).astype('float32')
        tree.fit(dta, annot)
        tree.enable_fast_prediction()
        print "tree depth: ", tree.depth
        #self.assertEqual(tree.depth, 3)
        dta = np.random.normal(size=(1000, 4)).astype("float32")
        print ("forpy prediction time (deep): " +
               str(timeit.Timer(lambda: tree.predict(dta))
                   .timeit(int(1E2)) / 1E2))

    def test_reg_deep(self):
        """Test regression tree prediction speed (deep)."""
        tree = forpy.Tree(20, 6, 12,
                          forpy.ThresholdDecider(
                              forpy.RegressionThresholdOptimizer(), 2),
                          forpy.RegressionLeaf())
        np.random.seed(1)
        dta = np.random.normal(size=(1000, 4)).astype("float32")
        annot = np.random.normal(size=(1000, 1)).astype('float32')
        tree.fit(dta, annot)
        tree.enable_fast_prediction()
        print "tree depth: ", tree.depth
        #self.assertEqual(tree.depth, 3)
        dta = np.random.normal(size=(1000, 4)).astype("float32")
        print ("forpy prediction time (deep, reg): " +
               str(timeit.Timer(lambda: tree.predict(dta))
                   .timeit(int(1E2)) / 1E2))

    def test_sklearn_deep(self):
        """Compare with sklearn (deep tree)."""
        import sklearn.tree as skt
        np.random.seed(1)
        dta = np.random.normal(size=(1000, 4)).astype("float32")
        annot = np.random.randint(low=0, high=2, size=(1000, 1)).astype("float32")
        tree = skt.DecisionTreeClassifier(max_depth=21)
        tree.fit(dta, annot)
        print 'tree depth sklearn', tree.tree_.max_depth
        #self.assertEqual(tree.tree_.max_depth, 4)
        dta = np.random.normal(size=(1000, 4)).astype("float32")
        print ("sklearn prediction time (deep): " +
               str(timeit.Timer(lambda: tree.predict(dta))
                   .timeit(int(1E2)) / 1E2))

    def test_sklearn_reg_deep(self):
        """Compare with sklearn (deep reg tree)."""
        import sklearn.tree as skt
        np.random.seed(1)
        dta = np.random.normal(size=(1000, 4)).astype("float32")
        annot = np.random.normal(size=(1000, 1)).astype("float32")
        tree = skt.DecisionTreeRegressor(max_depth=21)
        tree.fit(dta, annot)
        print 'tree depth sklearn', tree.tree_.max_depth
        #self.assertEqual(tree.tree_.max_depth, 4)
        dta = np.random.normal(size=(1000, 4)).astype("float32")
        print ("sklearn prediction time (deep, reg): " +
               str(timeit.Timer(lambda: tree.predict(dta))
                   .timeit(int(1E2)) / 1E2))

    def test_plain_deep_single(self):
        """Test plain tree prediction speed (deep, single sample)."""
        tree = forpy.Tree(20, 1, 2,
                          forpy.ThresholdDecider(
                              forpy.ClassificationThresholdOptimizer(), 2),
                          forpy.ClassificationLeaf())
        np.random.seed(1)
        dta = np.random.normal(size=(1000, 4)).astype("float32")
        annot = np.random.randint(low=0, high=2, size=(1000, 1)).astype('float32')
        tree.fit(dta, annot)
        tree.enable_fast_prediction()
        print "tree depth: ", tree.depth
        #self.assertEqual(tree.depth, 3)
        print ("forpy prediction time (deep, single): " +
               str(timeit.Timer(lambda: tree.predict(dta[0:1, :]))
                   .timeit(int(1E5)) / 1E5))

    def test_sklearn_deep_single(self):
        """Compare with sklearn (deep tree, single sample)."""
        import sklearn.tree as skt
        np.random.seed(1)
        dta = np.random.normal(size=(1000, 4)).astype("float32")
        annot = np.random.randint(low=0, high=2, size=(1000, 1)).astype("float32")
        tree = skt.DecisionTreeClassifier(max_depth=21)
        tree.fit(dta, annot)
        print 'tree depth sklearn', tree.tree_.max_depth
        #self.assertEqual(tree.tree_.max_depth, 4)
        print ("sklearn prediction time (deep, single): " +
               str(timeit.Timer(lambda: tree.predict(dta[0:1, :]))
                   .timeit(int(1E5)) / 1E5))


if __name__ == "__main__":
    unittest.main()
