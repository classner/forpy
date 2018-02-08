#!/usr/bin/env python
"""Test for serialization."""
# pylint: disable=no-member, no-self-use, invalid-name
try:
    import cPickle as pickle
except:  # pylint: disable=bare-except
    import pickle
import os
import os.path as path
import sys
import unittest

import numpy as np

sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))


class SerializationTestMixin(object):
    """Test serialization abilities."""

    def setUp(self):
        """Create the fixture."""
        np.random.seed(1)
        self.dta = np.random.normal(size=(100, 1)).astype(np.float32)
        self.dta_t = np.ascontiguousarray(self.dta.T)
        self.annot = np.random.randint(
            low=0, high=2, size=(100, 1)).astype(np.uint32)
        self.annot_r = np.random.randint(
            low=0, high=10, size=(100, 1)).astype(np.float32)
        self.forest = self.FOREST_CLASS()
        try:
            self.forest.fit(self.dta_t, self.annot)
        except RuntimeError:
            self.forest.fit(self.dta_t, self.annot_r)
        if hasattr(self.forest, 'tree_weights'):
            self.forest.tree_weights = range(10)

    def tearDown(self):
        """Remove artifacts."""
        if path.exists("pickle_test.pkl"):
            os.remove("pickle_test.pkl")
        if path.exists("bin_test.fpf"):
            os.remove("bin_test.fpf")
        if path.exists("bin_test.fpt"):
            os.remove("bin_test.fpt")
        if path.exists("json_test.json"):
            os.remove("json_test.json")

    #@unittest.skip("")
    def test_pickle(self):
        """Test the pickle protocol support."""
        with open("pickle_test.pkl", "wb") as outf:
            pickle.dump(self.forest, outf, -1)
        with open("pickle_test.pkl", "rb") as inf:
            forest2 = pickle.load(inf)
        self.assertEqual(self.forest, forest2)
        self.assertTrue(
            np.all(self.forest.predict(self.dta) == forest2.predict(self.dta)))

    def test_bin(self):
        """Test the binary protocol support."""
        self.forest.save("bin_test" + self.FPNAME_END)
        forest2 = self.FOREST_CLASS("bin_test" + self.FPNAME_END)
        self.assertEqual(self.forest, forest2)

    def test_json(self):
        """Test the json protocol support."""
        self.forest.save("json_test.json")
        forest2 = self.FOREST_CLASS("json_test.json")
        self.assertEqual(self.forest, forest2)


class TreeTest(SerializationTestMixin, unittest.TestCase):
    """Test the Tree serialization."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        import forpy
        self.FOREST_CLASS = forpy.Tree
        self.FPNAME_END = ".fpt"
        super(TreeTest, self).__init__(*args, **kwargs)


class ClassificationTreeTest(SerializationTestMixin, unittest.TestCase):
    """Test the Classification Tree serialization."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        import forpy
        self.FOREST_CLASS = forpy.ClassificationTree
        self.FPNAME_END = ".fpt"
        super(ClassificationTreeTest, self).__init__(*args, **kwargs)


class RegressionTreeTest(SerializationTestMixin, unittest.TestCase):
    """Test the Regression Tree serialization."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        import forpy
        self.FOREST_CLASS = forpy.RegressionTree
        self.FPNAME_END = ".fpt"
        super(RegressionTreeTest, self).__init__(*args, **kwargs)


class ForestTest(SerializationTestMixin, unittest.TestCase):
    """Test the Forest serialization."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        import forpy
        self.FOREST_CLASS = forpy.Forest
        self.FPNAME_END = ".fpf"
        super(ForestTest, self).__init__(*args, **kwargs)


class ClassificationForestTest(SerializationTestMixin, unittest.TestCase):
    """Test the Classification Forest serialization."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        import forpy
        self.FOREST_CLASS = forpy.ClassificationForest
        self.FPNAME_END = ".fpf"
        super(ClassificationForestTest, self).__init__(*args, **kwargs)


class RegressionForestTest(SerializationTestMixin, unittest.TestCase):
    """Test the Regression Forest serialization."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        import forpy
        self.FOREST_CLASS = forpy.RegressionForest
        self.FPNAME_END = ".fpf"
        super(RegressionForestTest, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    unittest.main()
