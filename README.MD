# forpy

[![PyPI version](https://badge.fury.io/py/forpy.svg)](https://pypi.python.org/pypi/forpy) [![Build Status](https://travis-ci.org/classner/forpy.svg?branch=master)](https://travis-ci.org/classner/forpy)[![Documentation](https://readthedocs.org/projects/forpy/badge/?version=latest)](http://forpy.readthedocs.io/en/latest/)

Elegant Decision Forests in Python. Fast, clean and clear C++14 implementation
with no dependencies and with an easy to use interface in Python and C++.

## Principles

1. Easy to hack  
   The library provides an object oriented implementation of parts of the
   theoretical framework
   of
   [Criminisi et al.](https://www.microsoft.com/en-us/research/publication/decision-forests-for-classification-regression-density-estimation-manifold-learning-and-semi-supervised-learning/).
   This makes the library easy to understand and extend while maintaining full
   generality. Objects are easy to reuse and recombine.
2. Easy to compile  
   There are no dependencies. The only build requirement is a C++14 compatible
   compiler (tested with GCC and clang). We use the CMake build system with
   pre-compiled headers for fast builds and compatibility to many platforms.
3. Easy to use  
   The library exposes a Python and C++ interface that accepts numpy/Eigen
   arrays of many types. The interface follows the widely used Scikit-learn
   interface with `fit` and `predict` methods.
4. Fast  
   The implementation of algorithms, while caring for readability and
   maintainability, is highly optimized for speed (benchmarks see below). We
   outperform scikit learn in all settings by notable margins at fit and predict
   time. For fitting, we enable fully deterministic parallelization even during
   node optimization, allowing to fully leverage modern CPUs with many cores.
5. Efficient  
   We use the [cereal](https://uscilab.github.io/cereal/) C++ serialization
   library enabling binary persistence (C++, pickle) or JSON export. The binary
   models need only a fraction of space of the corresponding scikit learn
   models.

## Approach

We use modern C++14 to create highly flexible, highly efficient data structures
and algorithm implementations. Core building blocks are:

* A highly
  efficient [`variant` implementation](https://github.com/mapbox/variant)
  for high efficiency w.r.t. datatype dependent storage and processing while
  maintaining an uncluttered interface and
  implementation. [`glog`](https://github.com/google/glog)
  and [`gperftools`](https://github.com/gperftools/gperftools) integration is
  used for easy debugging and optimization. All dependency libraries don't need
  to be installed but are part of the repository and completely integrated into
  the build.

* CMake is used as a build system so that many platforms and compilers can be
  targeted easily. [`cotire`](https://github.com/sakra/cotire) is used for
  automatic fast pre-compiled header builds. The whole package is automatically
  `pip` installable even without an installed CMake thanks
  to [`scikit-build`](https://github.com/scikit-build/scikit-build).

* We use standard [`Eigen`](http://eigen.tuxfamily.org) datastructures wrapped
  in `variant` to provide an easy-to-use C++ interface. We create a small and
  sleek Python interface thanks
  to [pybind11](https://github.com/pybind/pybind11).
  
* Threading is implemented
  with [cpptask](https://github.com/Kolkir/cpptask) to be efficient and
  cross-platform.

## Documentation

API documentation can be found [here](http://forpy.readthedocs.io/en/latest/).
Check out the tests for usage examples. The interface largely precisely matches
the one of scikit-learn (`forpy.ClassificationTree`, `forpy.RegressionTree`,
`forpy.ClassificationForest` and `forpy.RegressionForest` are even fully
compatible to the scikit learn predictor API and can be used within the
hyperparameter optimization functions out of the box).

## Compilation & Installation

If you want to use it from python, a simple `python setup.py install` should do.
If you want to use it from C++, you can rely on CMake and do `mkdir build; cd
build; cmake ..; cmake --build . -- -j` for an out-of-source build.

## Benchmarks

For the following plot, artificial random data is used to build maximally deep
trees. The benchmarks have been recorded on an Intel i7@2.8Ghz with four
physical cores.

![](https://github.com/classner/forpy/blob/master/build_support/benchmarks/timings_dset.png | width=250)
![](https://github.com/classner/forpy/blob/master/build_support/benchmarks/timings_n_annots.png | width=250)
![](https://github.com/classner/forpy/blob/master/build_support/benchmarks/timings_n_features.png | width=250)
![](https://github.com/classner/forpy/blob/master/build_support/benchmarks/timings_n_threads.png | width=250)

## Development & Contributing

### Formatting and conventions

The code is formatted
with [clang-format](https://clang.llvm.org/docs/ClangFormat.html) according to
the google C++ style guidelines (a `.clang-format` file is provided on project
level). We use abbreviated CamCase class names. For library internal
functions, assertions must be done using the `FASSERT` macro to selectively enable it,
but should be disabled for performance reasons for release builds.

Error messages *must* be meaningful and provide additional information on what
caused the error.

Design decisions are made according to the following priorities:: correctness &
numerical stability >> speed >> in-memory efficiency >> storage efficiency.

#### Rough naming conventions

Raw pointer variable names end with `_p`, indicating (1) high performance
element access and (2) special care. In performance relevant loops, only `_p`
variables should be used. Variant variable names end with `_v`.

### Programming concepts

**Lock-free parallelism:**

Memory is allocated for the tree and leaf storage structures in before entering
parallel regions in a pessimistic way. Since it is linear in the number of
samples, this is not too much of an overhead. The pointer to where next
nodes/leafs can be created is an `std::atomic<size_t>`. After training, the
datastructures are resized to their proper size.

**The [Desk](@ref forpydeskGroup) classes:** this is a helper concept to
simplify safe parallelism. A 'desk' contains all thread-local variables and
pointers to the shared storage. Each thread has its own [desk](@ref forpy::Desk)
with sub-objects containing thread-local storage for the corresponding
sub-functions. It is constructed with pointers to the memory for storing the
results of the training.

## License

The library itself is available under the 2-clause BSD license. All libraries
used are also available under open source licenses, for details see
`build_support/external`.
