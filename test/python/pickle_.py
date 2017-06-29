#!/usr/bin/env python2
import numpy as np
import cPickle as pickle
import os.path as path
import sys
sys.path.insert(0,
                path.join(path.dirname(__file__),
                          '..', '..'))
import forpy

cr = forpy.ConstantRegressor()
cr.initialize(range(10), range(10))
cr.freeze()
assert cr == pickle.loads(pickle.dumps(cr, 2))

lr = forpy.LinearRegressor()
lr.initialize(range(10), range(10))
lr.freeze()
assert lr == pickle.loads(pickle.dumps(lr, 2))
