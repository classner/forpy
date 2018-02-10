#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script for the forpy project."""
from __future__ import print_function

import subprocess
import sys
import unittest

try:
    from skbuild import setup
except:  # pylint: disable=bare-except
    print(
        "This package requires `scikit-build` for setup. Trying to install it..."
    )
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'scikit-build'])
    except:  # pylint: disable=bare-except
        print(
            "Automatic installation failed. Please install `scikit-build` and "
        )
        print("rerun the package setup!")
from skbuild import setup  # isort:skip

# Parse the version string.
with open("CMakeLists.txt", 'r') as inf:
    for line in inf:
        if 'forpy_VERSION_MAJOR' in line:
            ver_major = line[line.find("forpy_VERSION_MAJOR") + 20:line.find(
                ")")]
        if 'forpy_VERSION_MINOR' in line:
            ver_minor = line[line.find("forpy_VERSION_MINOR") + 20:line.find(
                ")")]
        if 'forpy_VERSION_PATCH' in line:
            ver_patch = line[line.find("forpy_VERSION_PATCH") + 20:line.find(
                ")")]
            break
VERSION = ver_major + '.' + ver_minor + '.' + ver_patch

try:
    subprocess.check_call(["cmake", "--version"])
except:  # pylint: disable=bare-except
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'cmake'])
    except:  # pylint: disable=bare-except
        print("Automatic installation failed. Please install `cmake` and ")
        print("rerun the package setup!")


def python_test_suite():
    """Discover all python tests."""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('test/python', pattern='*.py')
    return test_suite


setup(
    name='forpy',
    author='Christoph Lassner',
    author_email='mail@christophlassner.de',
    url='https://github.com/classner/forpy',
    download_url='https://github.com/classner/forpy/tarball/v{}'.format(
        VERSION),
    keywords='random forests decision machine learning',
    classifiers=[],
    py_modules=['forpy'],
    cmake_source_dir='.',
    cmake_install_dir='',
    cmake_args=[
        '-DWITH_PYTHON=On',
        '-DPYTHON_EXECUTABLE=' + sys.executable,
    ],
    test_suite='setup.python_test_suite',
    setup_requires=['numpy'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'sklearn'],
    version=VERSION,
    license='BSD 2-clause',
)
