#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# @author: Moritz Einfalt, Christoph Lassner.
import os.path as path
import sys

if len(sys.argv) < 2:
  import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0,
                path.join(path.dirname(__file__), '..', '..'))
import forpy as fp


def to_im_coords(points, im_spec):
  retpoints = []
  for point in points:
    if point < im_spec[0] or point > im_spec[1]:
      print('Warning: detected point outside im region.')
      continue
    retpoints.append((point - im_spec[0]) / (im_spec[1] - im_spec[0]) * im_spec[2])
  return np.array(retpoints)

# The probability density function for the univariate normal distribution.
# Since it is a continuous distribution, it does not return probabilites,
# but rather likelihoods.
def normal_pdf(x, mean, var):
  return (1/np.sqrt(2*np.pi*var))*np.exp(-np.power((x-mean),2) / (2 * var))

# Helper function to transform a likelihood value into a red saturated color,
# relative to a maximum value.
def color_for_likelihood (likelihood, max_likelihood):
  try:
    color_strength = min(255. * (1. - np.power(likelihood / max_likelihood, 0.2)), 255)
    return np.array([255,color_strength,color_strength], dtype='uint8')
  except:
    #print likelihood, max_likelihood
    return np.array([255, 255, 255], dtype='uint8')


# Function to plot the probabilistic result of a 1D - 1D regressor.
# The interface is identical to the point_prob_plot-function for the
# classification example.
def probabilistic_plot(regressor, X_train, Y_train, plotX, plotY):
  # Define the x values for prediction input.
  X_test = np.linspace(plotX[0], plotX[1], num=plotX[2])
  X_test.shape = (X_test.shape[0],1)
  # Predict the y values based on X_test
  prediction = []
  for sample_idx in range(X_test.shape[0]):
      pred = regressor.predict_covar(X_test[sample_idx])
      prediction.append([pred[0], pred[1][0, 0]])
  prediction = np.array(prediction, dtype=np.float32)

  # Adjust the plotted area, if the predicitons exceed the initial y-bounds
  minY = prediction[:,0].min()
  maxY = prediction[:,0].max()
  plotY = [min(plotY[0], minY), max(plotY[1], maxY), plotY[2]]
  # Define the possible predictions in the plotted area
  Y_test = np.linspace(plotY[0], plotY[1], num=plotY[2])
  # Shape adjustment
  X_test.shape=(X_test.shape[0],)
  # Define the image for the red saturation representing the likelihood of
  # the various prediction values
  image = np.zeros((Y_test.shape[0], X_test.shape[0], 3), dtype='uint8')
  # Get the maximum likelihood of all predicitons
  # Only the mean-predititons are taken into consideration,
  # since their likelihood is always the highest.
  print np.max(prediction[:, 1]), np.min(prediction[:, 1])
  max_likelihood = np.max(normal_pdf(prediction[:,0],
                                     prediction[:,0],
                                     prediction[:,1]))
  print 'max_likelihood:', max_likelihood
  # Get the likelihood for all x-y combinations
  # and transform it into a red saturated color
  for y in range(image.shape[0]):
    for x in range(image.shape[1]):
      image[y,x] = color_for_likelihood(normal_pdf(Y_test[y],
                                                   prediction[x,0],
                                                   prediction[x,1]),
                                        max_likelihood)
  plt.imshow(image, origin='lower')
  # Transform all plotted values into the image coordinate system
  Xtraintransformed = to_im_coords(X_train[:, 0], plotX)
  Ytraintransformed = to_im_coords(Y_train[:, 0], plotY)
  Xtesttransformed = to_im_coords(X_test, plotX)
  Ytesttransformed = to_im_coords(prediction[:,0], plotY)
  # Plot the training samples and the mean predicitons
  plt.plot(Xtesttransformed, Ytesttransformed, c='g', linewidth=1, label='Regression model')
  plt.plot(Xtraintransformed, Ytraintransformed ,linestyle='None', marker='o',
           markeredgecolor='blue', markeredgewidth=1,
           markerfacecolor='None', markersize=6,
           label='Observations')
  # Plot settings
  frame = plt.gca()
  frame.set_xlim(to_im_coords(plotX[0:2],plotX))
  frame.set_ylim(to_im_coords(plotY[0:2],plotY))
  frame.axes.get_xaxis().tick_bottom()
  frame.axes.get_yaxis().tick_left()
  frame.axes.set_aspect('equal')
  plt.legend(loc='upper left')
  plt.xlabel(r"$X$", fontsize=18)
  plt.ylabel(r"$Y$", fontsize=18)
  plt.rcParams['xtick.direction'] = 'out'
  plt.rcParams['ytick.direction'] = 'out'
  plt.grid(True)
  return

np.random.seed(0)
n_samples = 30

annot_dim = 1
input_dim = 1
samplesX = np.random.uniform(low=-6, high=6, size=(n_samples, input_dim))
#samplesX = np.array([2, 3, 4, 5]).reshape((4, 1))
#samplesX[:5] = 0
#samplesY = (samplesX ** 2) - 0.2 * samplesX - 1
samplesY = samplesX.copy()
#samplesY -= samplesY.mean()

noise_var = 2.3
random_noise = np.random.randn(n_samples, annot_dim) * noise_var
samplesY += random_noise

# Define the plotted space
minX = samplesX.min()
maxX = samplesX.max()
deltaX = (maxX - minX)
minY = samplesY.min()
maxY = samplesY.max()
deltaY = (maxY - minY)
plotX = [minX -0.2*deltaX, maxX +0.2*deltaX, 100]
plotY = [minY -0.2*deltaY, maxY +0.2*deltaY, 100]

regressor = fp.ConstantRegressor()
regressor.initialize(samplesX, samplesY)
prediction = []
for sample_idx in range(samplesX.shape[0]):
  pred = regressor.predict_covar(samplesX[sample_idx])
  prediction.append([pred[0], pred[1][0, 0]])
prediction = np.array(prediction, dtype=np.float32)
sqe = np.sum(np.square(prediction[:, 0] - samplesX[:, 0])) / samplesX.shape[0]
assert sqe < 12.
print "Constant regressor sqe: ", sqe

regressor = fp.LinearRegressor(force_numerical_stability=False)
regressor.initialize(samplesX, samplesY)
prediction = []
for sample_idx in range(samplesX.shape[0]):
  pred = regressor.predict_covar(samplesX[sample_idx])
  prediction.append([pred[0], pred[1][0, 0]])
prediction = np.array(prediction, dtype=np.float32)
sqe = np.sum(np.square(prediction[:, 0] - samplesX[:, 0])) / samplesX.shape[0]
print "Linear regressor sqe: ", sqe
assert sqe < 0.1

# Plot the result
if len(sys.argv) < 2:
  plt.figure()
  probabilistic_plot(regressor, samplesX, samplesY, plotX, plotY)
  plt.show()

# Speed.
# cr = forpy.ConstantRegressor()
#%timeit cr.initialize(range(100000), range(100000)); cr.index_interval = [0, 1]; cr.index_interval = [0, 100000]
