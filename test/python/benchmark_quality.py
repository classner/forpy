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

    def test_plain_deep(self):
        """Test plain tree prediction speed."""
        import sklearn.metrics as skm
        np.random.seed(1)
        dta = np.random.normal(size=(10000, 4)).astype("float32")
        annot = np.random.randint(low=0, high=2, size=(10000, 1)).astype('float32')
        def fit():
            """Closure for timing."""
            tree = forpy.Tree(20, 1, 2,
                              forpy.ThresholdDecider(
                                  forpy.ClassificationThresholdOptimizer(),
                                  4),  # n valid features to use.
                              forpy.ClassificationLeaf())
            tree.fit(dta, annot)
            return tree
        print ("forpy fitting time (deep): " +
               str(timeit.Timer(fit)
                   .timeit(int(1E1)) / 1E1))
        tree = fit()
        print ("forpy depth: ", tree.depth)
        print ("forpy accuracy: ", skm.accuracy_score(annot,
                                                      np.argmax(tree.predict(dta), axis=1)))

    def test_sklearn_deep(self):
        """Test plain tree prediction speed."""
        import sklearn.tree as skt
        import sklearn.metrics as skm
        np.random.seed(1)
        dta = np.random.normal(size=(10000, 4)).astype("float32")
        annot = np.random.randint(low=0, high=2, size=(10000, 1)).astype('float32')
        def fit():
            """Closure for timing."""
            tree = skt.DecisionTreeClassifier(
                criterion='entropy', max_depth=20)
            tree.fit(dta, annot)
            return tree
        print ("sklearn fitting time (deep): " +
               str(timeit.Timer(fit)
                   .timeit(int(1E1)) / 1E1))
        tree = fit()
        print ("sklearn depth: ", tree.tree_.max_depth)
        print ("sklearn accuracy: ", skm.accuracy_score(annot,
                                                        tree.predict(dta)))


if __name__ == '__main__':
    unittest.main()
