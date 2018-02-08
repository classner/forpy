#!/usr/bin/env python
"""Tests creating speed plots."""
# pylint: disable=no-member, no-self-use, cell-var-from-loop, invalid-name, bare-except
from __future__ import print_function

import os.path as path
import sys
import timeit
import unittest

import numpy as np

sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))


class SpeedTest(unittest.TestCase):
    """Test training speed."""

    def test_asymptotic_dset(self):  # pylint: disable=too-many-locals
        """Test asymptotic dataset size speed."""
        import sklearn.tree as skt
        import matplotlib
        import forpy
        maxdepth = 4294967295
        nfeats = 10
        nannots = 20
        n_classes = 20
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xvals = []
        y_vals = {}
        np.random.seed(1)
        for dset_size in [2**exp for exp in range(7, 15)]:
            for p_name, pred in [('sklearn_cl', skt.DecisionTreeClassifier),
                                 ('sklearn_reg', skt.DecisionTreeRegressor),
                                 ('forpy_cl', forpy.ClassificationTree),
                                 ('forpy_reg', forpy.RegressionTree)]:
                print('dset size: ', dset_size)
                if dset_size not in xvals:
                    xvals.append(dset_size)
                dta = np.random.normal(size=(dset_size, nfeats)).astype(
                    np.float32)
                dta_t = np.ascontiguousarray(dta[:, :].T)
                annot = np.random.normal(size=(dset_size, nannots)).astype(
                    np.float32)
                annot_c = np.random.randint(
                    low=0, high=n_classes, size=(dset_size, 1)).astype(
                        np.uint32)

                def fit_forpy():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta_t, annot, n_threads=1)
                    return tree

                def fit_forpy_c():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta_t, annot_c, n_threads=1)
                    return tree

                def fit_sklearn():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta, annot)
                    return tree

                def fit_sklearn_c():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta, annot_c)
                    return tree

                if '_cl' in p_name:
                    try:
                        time = timeit.Timer(fit_forpy_c).timeit(int(1E1)) / 1E1
                    except:
                        time = timeit.Timer(fit_sklearn_c).timeit(
                            int(1E1)) / 1E1
                else:
                    try:
                        time = timeit.Timer(fit_forpy).timeit(int(1E1)) / 1E1
                    except:
                        time = timeit.Timer(fit_sklearn).timeit(int(1E1)) / 1E1
                if not p_name in y_vals:
                    y_vals[p_name] = []
                y_vals[p_name].append(time)
        for p_name in y_vals:
            plt.plot(xvals, y_vals[p_name], label=p_name)
            plt.scatter(xvals, y_vals[p_name])
        # Analyze asymptotes.
        import sklearn.linear_model as skl
        for p_name, ys in y_vals.items():
            lrsk = skl.LinearRegression().fit(
                np.log(np.array(xvals)).reshape(-1, 1), np.log(ys))
            print(p_name + ' factor: ', lrsk.coef_)
            print(p_name + ' intercept: ', lrsk.intercept_)
        y_sk = y_vals['sklearn_cl']
        y_fp = y_vals['forpy_cl']
        print("Classification:")
        clavg = np.mean(np.array(y_sk) / np.array(y_fp))
        clmin = np.min(np.array(y_sk) / np.array(y_fp))
        clmax = np.max(np.array(y_sk) / np.array(y_fp))
        print('avg speedup:', clavg)
        print('min speedup:', clmin)
        print('max speedup:', clmax)
        y_sk = y_vals['sklearn_reg']
        y_fp = y_vals['forpy_reg']
        print("Regression:")
        regavg = np.mean(np.array(y_sk) / np.array(y_fp))
        regmin = np.min(np.array(y_sk) / np.array(y_fp))
        regmax = np.max(np.array(y_sk) / np.array(y_fp))
        print('avg speedup:', regavg)
        print('min speedup:', regmin)
        print('max speedup:', regmax)
        # Finalize plot.
        plt.loglog()
        plt.suptitle("Runtime vs. dset_size (logarithmic)", fontsize=14)
        plt.title(
            ("Cl. fac. (min/av./max): {:.2f}, {:.2f}, {:.2f}; Regr. "
             "fac.: {:.2f}, {:.2f}, {:.2f}.").format(clmin, clavg, clmax,
                                                     regmin, regavg, regmax),
            fontsize=10)
        plt.xlabel("Dataset size")
        plt.ylabel("Runtime")
        plt.legend()
        plt.savefig("timings_dset.png")
        plt.close()

    def test_asymptotic_feats(self):  # pylint: disable=too-many-locals
        """Test asymptotic number of features speed."""
        import sklearn.tree as skt
        import matplotlib
        import forpy
        maxdepth = 4294967295
        dset_size = 512
        nannots = 20
        n_classes = 20
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xvals = []
        y_vals = {}
        np.random.seed(1)
        for n_feats in [2**exp for exp in range(5, 10)]:
            for p_name, pred in [('sklearn_cl', skt.DecisionTreeClassifier),
                                 ('sklearn_reg', skt.DecisionTreeRegressor),
                                 ('forpy_cl', forpy.ClassificationTree),
                                 ('forpy_reg', forpy.RegressionTree)]:
                print('n_feats: ', n_feats)
                if n_feats not in xvals:
                    xvals.append(n_feats)
                dta = np.random.normal(size=(dset_size, n_feats)).astype(
                    np.float32)
                dta_t = np.ascontiguousarray(dta[:, :].T)
                annot = np.random.normal(size=(dset_size, nannots)).astype(
                    np.float32)
                annot_c = np.random.randint(
                    low=0, high=n_classes, size=(dset_size, 1)).astype(
                        np.uint32)

                def fit_forpy():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta_t, annot, n_threads=1)
                    return tree

                def fit_forpy_c():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta_t, annot_c, n_threads=1)
                    return tree

                def fit_sklearn():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta, annot)
                    return tree

                def fit_sklearn_c():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta, annot_c)
                    return tree

                if '_cl' in p_name:
                    try:
                        time = timeit.Timer(fit_forpy_c).timeit(int(1E1)) / 1E1
                    except:
                        time = timeit.Timer(fit_sklearn_c).timeit(
                            int(1E1)) / 1E1
                else:
                    try:
                        time = timeit.Timer(fit_forpy).timeit(int(1E1)) / 1E1
                    except:
                        time = timeit.Timer(fit_sklearn).timeit(int(1E1)) / 1E1
                if not p_name in y_vals:
                    y_vals[p_name] = []
                y_vals[p_name].append(time)
        for p_name in y_vals:
            plt.plot(xvals, y_vals[p_name], label=p_name)
            plt.scatter(xvals, y_vals[p_name])
        # Analyze asymptotes.
        import sklearn.linear_model as skl
        for p_name, ys in y_vals.items():
            lrsk = skl.LinearRegression().fit(
                np.log(np.array(xvals)).reshape(-1, 1), np.log(ys))
            print(p_name + ' factor: ', lrsk.coef_)
            print(p_name + ' intercept: ', lrsk.intercept_)
        y_sk = y_vals['sklearn_cl']
        y_fp = y_vals['forpy_cl']
        print("Classification:")
        clavg = np.mean(np.array(y_sk) / np.array(y_fp))
        clmin = np.min(np.array(y_sk) / np.array(y_fp))
        clmax = np.max(np.array(y_sk) / np.array(y_fp))
        print('avg speedup:', clavg)
        print('min speedup:', clmin)
        print('max speedup:', clmax)
        y_sk = y_vals['sklearn_reg']
        y_fp = y_vals['forpy_reg']
        print("Regression:")
        regavg = np.mean(np.array(y_sk) / np.array(y_fp))
        regmin = np.min(np.array(y_sk) / np.array(y_fp))
        regmax = np.max(np.array(y_sk) / np.array(y_fp))
        print('avg speedup:', regavg)
        print('min speedup:', regmin)
        print('max speedup:', regmax)
        # Finalize plot.
        plt.loglog()
        plt.suptitle("Runtime vs. n_features (logarithmic)", fontsize=14)
        plt.title(
            ("Cl. fac. (min/av./max): {:.2f}, {:.2f}, {:.2f}; Regr. "
             "fac.: {:.2f}, {:.2f}, {:.2f}.").format(clmin, clavg, clmax,
                                                     regmin, regavg, regmax),
            fontsize=10)
        plt.xlabel("Number of features")
        plt.ylabel("Runtime")
        plt.legend()
        plt.savefig("timings_n_features.png")
        plt.close()

    def test_asymptotic_annots(self):  # pylint: disable=too-many-locals
        """Test asymptotic number of annotations speed."""
        import sklearn.tree as skt
        import matplotlib
        import forpy
        maxdepth = 4294967295
        dset_size = 1024
        n_feats = 20
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xvals = []
        y_vals = {}
        np.random.seed(1)
        for n_annots in [2**exp for exp in range(5, 10)]:
            for p_name, pred in [('sklearn_cl', skt.DecisionTreeClassifier),
                                 ('sklearn_reg', skt.DecisionTreeRegressor),
                                 ('forpy_cl', forpy.ClassificationTree),
                                 ('forpy_reg', forpy.RegressionTree)]:
                print('n_annots: ', n_annots)
                if n_annots not in xvals:
                    xvals.append(n_annots)
                dta = np.random.normal(size=(dset_size, n_feats)).astype(
                    np.float32)
                dta_t = np.ascontiguousarray(dta[:, :].T)
                annot = np.random.normal(size=(dset_size, n_annots)).astype(
                    np.float32)
                annot_c = np.random.randint(
                    low=0, high=n_annots, size=(dset_size, 1)).astype(
                        np.uint32)

                def fit_forpy():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta_t, annot, n_threads=1)
                    return tree

                def fit_forpy_c():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta_t, annot_c, n_threads=1)
                    return tree

                def fit_sklearn():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta, annot)
                    return tree

                def fit_sklearn_c():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta, annot_c)
                    return tree

                if '_cl' in p_name:
                    try:
                        time = timeit.Timer(fit_forpy_c).timeit(int(1E1)) / 1E1
                    except:
                        time = timeit.Timer(fit_sklearn_c).timeit(
                            int(1E1)) / 1E1
                else:
                    try:
                        time = timeit.Timer(fit_forpy).timeit(int(1E1)) / 1E1
                    except:
                        time = timeit.Timer(fit_sklearn).timeit(int(1E1)) / 1E1
                if not p_name in y_vals:
                    y_vals[p_name] = []
                y_vals[p_name].append(time)
        for p_name in y_vals:
            plt.plot(xvals, y_vals[p_name], label=p_name)
            plt.scatter(xvals, y_vals[p_name])
        # Analyze asymptotes.
        import sklearn.linear_model as skl
        for p_name, ys in y_vals.items():
            lrsk = skl.LinearRegression().fit(
                np.log(np.array(xvals)).reshape(-1, 1), np.log(ys))
            print(p_name + ' factor: ', lrsk.coef_)
            print(p_name + ' intercept: ', lrsk.intercept_)
        y_sk = y_vals['sklearn_cl']
        y_fp = y_vals['forpy_cl']
        print("Classification:")
        clavg = np.mean(np.array(y_sk) / np.array(y_fp))
        clmin = np.min(np.array(y_sk) / np.array(y_fp))
        clmax = np.max(np.array(y_sk) / np.array(y_fp))
        print('avg speedup:', clavg)
        print('min speedup:', clmin)
        print('max speedup:', clmax)
        y_sk = y_vals['sklearn_reg']
        y_fp = y_vals['forpy_reg']
        print("Regression:")
        regavg = np.mean(np.array(y_sk) / np.array(y_fp))
        regmin = np.min(np.array(y_sk) / np.array(y_fp))
        regmax = np.max(np.array(y_sk) / np.array(y_fp))
        print('avg speedup:', regavg)
        print('min speedup:', regmin)
        print('max speedup:', regmax)
        # Finalize plot.
        plt.loglog()
        plt.suptitle("Runtime vs. n_annots (logarithmic)", fontsize=14)
        plt.title(
            ("Cl. fac. (min/av./max): {:.2f}, {:.2f}, {:.2f}; Regr. "
             "fac.: {:.2f}, {:.2f}, {:.2f}.").format(clmin, clavg, clmax,
                                                     regmin, regavg, regmax),
            fontsize=10)
        plt.xlabel("Number of annotations")
        plt.ylabel("Runtime")
        plt.legend()
        plt.savefig("timings_n_annots.png")
        plt.close()

    def test_threads(self):  # pylint: disable=too-many-locals, too-many-statements
        """Test number of threads speed."""
        import sklearn.tree as skt
        import matplotlib
        import forpy
        maxdepth = 4294967295
        dset_size = 4096
        n_feats = 20
        n_annots = 30
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xvals = []
        y_vals = {}
        np.random.seed(1)
        for n_threads in range(1, 5):
            for p_name, pred in [('sklearn_cl', skt.DecisionTreeClassifier),
                                 ('sklearn_reg', skt.DecisionTreeRegressor),
                                 ('forpy_cl', forpy.ClassificationTree),
                                 ('forpy_reg', forpy.RegressionTree)]:
                print('n_threads: ', n_threads)
                if n_threads not in xvals:
                    xvals.append(n_threads)
                dta = np.random.normal(size=(dset_size, n_feats)).astype(
                    np.float32)
                dta_t = np.ascontiguousarray(dta[:, :].T)
                annot = np.random.normal(size=(dset_size, n_annots)).astype(
                    np.float32)
                annot_c = np.random.randint(
                    low=0, high=n_annots, size=(dset_size, 1)).astype(
                        np.uint32)

                def fit_forpy():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta_t, annot, n_threads=n_threads)
                    return tree

                def fit_forpy_c():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta_t, annot_c, n_threads=n_threads)
                    return tree

                def fit_sklearn():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta, annot)
                    return tree

                def fit_sklearn_c():
                    """Closure for timing."""
                    tree = pred(max_depth=maxdepth)
                    tree.fit(dta, annot_c)
                    return tree

                if '_cl' in p_name:
                    try:
                        time = timeit.Timer(fit_forpy_c).timeit(int(1E1)) / 1E1
                    except:
                        time = timeit.Timer(fit_sklearn_c).timeit(
                            int(1E1)) / 1E1
                else:
                    try:
                        time = timeit.Timer(fit_forpy).timeit(int(1E1)) / 1E1
                    except:
                        time = timeit.Timer(fit_sklearn).timeit(int(1E1)) / 1E1
                if not p_name in y_vals:
                    y_vals[p_name] = []
                y_vals[p_name].append(time)
        colors = {
            'sklearn_reg': 'blue',
            'sklearn_cl': 'green',
            'forpy_reg': 'red',
            'forpy_cl': 'orange'
        }
        for p_name in y_vals:
            if p_name.startswith("sklearn"):
                plt.axhline(
                    y=y_vals[p_name][0], label=p_name, color=colors[p_name])
            else:
                plt.plot(
                    xvals, y_vals[p_name], label=p_name, color=colors[p_name])
                plt.scatter(xvals, y_vals[p_name], color=colors[p_name])
        # Analyze asymptotes.
        import sklearn.linear_model as skl
        for p_name, ys in y_vals.items():
            lrsk = skl.LinearRegression().fit(
                np.log(np.array(xvals)).reshape(-1, 1), np.log(ys))
            print(p_name + ' factor: ', lrsk.coef_)
            print(p_name + ' intercept: ', lrsk.intercept_)
        y_sk = y_vals['sklearn_cl']
        y_fp = y_vals['forpy_cl']
        print("Classification:")
        clavg = np.mean(np.array(y_sk) / np.array(y_fp))
        clmin = np.min(np.array(y_sk) / np.array(y_fp))
        clmax = np.max(np.array(y_sk) / np.array(y_fp))
        print('avg speedup:', clavg)
        print('min speedup:', clmin)
        print('max speedup:', clmax)
        y_sk = y_vals['sklearn_reg']
        y_fp = y_vals['forpy_reg']
        print("Regression:")
        regavg = np.mean(np.array(y_sk) / np.array(y_fp))
        regmin = np.min(np.array(y_sk) / np.array(y_fp))
        regmax = np.max(np.array(y_sk) / np.array(y_fp))
        print('avg speedup:', regavg)
        print('min speedup:', regmin)
        print('max speedup:', regmax)
        # Finalize plot.
        plt.loglog()
        plt.suptitle("Runtime vs. n_threads (logarithmic)", fontsize=14)
        plt.title(
            ("Cl. fac. (min/av./max): {:.2f}, {:.2f}, {:.2f}; Regr. "
             "fac.: {:.2f}, {:.2f}, {:.2f}.").format(clmin, clavg, clmax,
                                                     regmin, regavg, regmax),
            fontsize=10)
        plt.xlabel("Threads")
        plt.ylabel("Runtime")
        plt.legend()
        plt.savefig("timings_n_threads.png")
        plt.close()


if __name__ == '__main__':
    unittest.main()
