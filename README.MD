# forpy

Inofficial, incomplete successor to
the [fertilized forests](http://www.fertilized-forests.org) project. It is
incomplete work in progress.

## Aims and approach

The parts that are there are strongly reworked and restructured while reflecting
the basic concepts of the original library.

* There are no dependencies and the library can be built directly from pip.
* The build process is fast and the resulting binaries small thanks to the use
of the excellent pybind11.
* All type inference is done within the library thanks to
the [mapbox variant implementation](https://github.com/mapbox/variant), so it is
easy and convenient to use.

## Compilation & Installation

If you want to use it from python, a simple `python setup.py install` should do.
If you want to use it from C++, you can rely on CMake and do `mkdir build; cd
build; cmake ..; cmake --build . -- -j` for an out-of-source build.

## License

The library itself is available under the 2-clause BSD license. All libraries
used are also available under open source licenses, for details see
`build_support/external`.
