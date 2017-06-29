#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
The setup script for the forpy project.
@author: Christoph Lassner
"""
import sys
import subprocess
import unittest
try:
    from skbuild import setup
except:  # pylint: disable=bare-except
    print "This package requires `scikit-build` for setup. Trying to install it..."
    try:
        subprocess.check_call([sys.executable,
                               '-m', 'pip', 'install', 'scikit-build'])
    except:  # pylint: disable=bare-except
        print "Automatic installation failed. Please install `scikit-build` and "
        print "rerun the package setup!"
from skbuild import setup
from pip.req import parse_requirements

# Parse the version string.
with open("CMakeLists.txt", 'r') as inf:
    for line in inf:
        if 'forpy_VERSION_MAJOR' in line:
            ver_major = line[line.find("forpy_VERSION_MAJOR")+20:line.find(")")]
        if 'forpy_VERSION_MINOR' in line:
            ver_minor = line[line.find("forpy_VERSION_MINOR")+20:line.find(")")]
        if 'forpy_VERSION_PATCH' in line:
            ver_patch = line[line.find("forpy_VERSION_PATCH")+20:line.find(")")]
            break
VERSION = ver_major + '.' + ver_minor + '.' + ver_patch
REQS = [str(ir.req) for ir in parse_requirements('requirements.txt',
                                                 session='tmp')]
try:
    subprocess.check_call(["cmake", "--version"])
except:  # pylint: disable=bare-except
    REQS.append("cmake")

def python_test_suite():
    """Discover all python tests."""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test/python/*.py')
    return test_suite

setup(
    name='forpy',
    author='Christoph Lassner',
    author_email='mail@christophlassner.de',
    py_modules=['forpy'],
    cmake_install_dir='',
    cmake_args=[
        '-DWITH_PYTHON=ON',
        '-DPYTHON_EXECUTABLE=' + sys.executable,],
    test_suite='setup.python_test_suite',
    install_requires=REQS,
    version=VERSION,
    license='BSD 2-clause',
)
