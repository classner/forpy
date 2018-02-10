#!/usr/bin/env python
"""Test leaf storage."""
import os.path as path
import sys
import unittest
import warnings

# pylint: disable=no-member, invalid-name, no-self-use
import numpy as np

sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))


class TestSKLearnMixin(object):  # pylint: disable=too-few-public-methods
    """Test Scikit Learn drop-in interface."""

    def test_gridsearch(self):
        """Test the param interface."""
        from sklearn.model_selection import GridSearchCV
        parameters = [{'max_depth': [2, 5]}]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = GridSearchCV(
                self.FOREST_CLASS(),
                parameters,
                cv=2,
                scoring='explained_variance')
            try:
                clf.fit(
                    np.zeros((10, 2), dtype=np.float32),
                    np.array(range(10), dtype=np.uint32))
            except RuntimeError:
                clf.fit(
                    np.zeros((10, 2), dtype=np.float32),
                    np.array(range(10), dtype=np.float32))


class ClassificationTreeTest(TestSKLearnMixin, unittest.TestCase):
    """Test Classification Tree skinterface."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        import forpy
        self.FOREST_CLASS = forpy.ClassificationTree
        super(ClassificationTreeTest, self).__init__(*args, **kwargs)


class RegressionTreeTest(TestSKLearnMixin, unittest.TestCase):
    """Test Regression Tree skinterface."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        import forpy
        self.FOREST_CLASS = forpy.RegressionTree
        super(RegressionTreeTest, self).__init__(*args, **kwargs)


class ClassificationForestTest(TestSKLearnMixin, unittest.TestCase):
    """Test Classification Forest skinterface."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        import forpy
        self.FOREST_CLASS = forpy.ClassificationForest
        super(ClassificationForestTest, self).__init__(*args, **kwargs)


class RegressionForestTest(TestSKLearnMixin, unittest.TestCase):
    """Test Regression Forest skinterface."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        import forpy
        self.FOREST_CLASS = forpy.RegressionForest
        super(RegressionForestTest, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    unittest.main()
