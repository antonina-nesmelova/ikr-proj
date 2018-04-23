#!/usr/bin/env python3

import matplotlib.pyplot as plt
import scipy.linalg
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt # import
from sklearn.decomposition import PCA
import matplotlib.cm as cm 
from glob import glob
from numpy.random import randint
from ikrlib import rand_gauss, plot2dfun, gellipse, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm, logistic_sigmoid

from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

from skimage import filter

TARGET_DEV = 'data/target_dev/'
NONTARGET_DEV = 'data/non_target_dev'
TARGET_TRAIN = 'data/target_train'
NONTARGET_TRAIN = 'data/non_target_train'

def weightedAverage(pixel):
    """ Truns pixel to greyscale. """
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

def getTrain_TargetFeatures():
    result_array = np.array([])
    for f in glob(TARGET_TRAIN + '/*.png'):
        image = misc.imread(f)
        grey = np.zeros((image.shape[0], image.shape[1]))
        for rownum in range(len(image)):
            for colnum in range(len(image[rownum])):
                grey[rownum][colnum] = weightedAverage(image[rownum][colnum])
        result_array = np.append(result_array, grey)
        edges = filter.sobel(grey)
    result_array = result_array.reshape(20,6400)
    return result_array

def getTrain_NonTargetFeatures():
    result_array = np.array([])
    for f in glob(NONTARGET_TRAIN + '/*.png'):
        image = misc.imread(f)
        grey = np.zeros((image.shape[0], image.shape[1]))
        for rownum in range(len(image)):
            for colnum in range(len(image[rownum])):
                grey[rownum][colnum] = weightedAverage(image[rownum][colnum])
        result_array = np.append(result_array, grey)
        edges = filter.sobel(grey)
    result_array = result_array.reshape(131,6400)
    return result_array

def getTest_TargetFeatures():
    result_array = np.array([])
    for f in glob(TARGET_DEV + '/*.png'):
        image = misc.imread(f)
        grey = np.zeros((image.shape[0], image.shape[1]))
        for rownum in range(len(image)):
            for colnum in range(len(image[rownum])):
                grey[rownum][colnum] = weightedAverage(image[rownum][colnum])
        result_array = np.append(result_array, grey)
        edges = filter.sobel(grey)
    result_array = result_array.reshape(10,6400)
    return result_array

def getTest_NonTargetFeatures():
    result_array = np.array([])
    for f in glob(NONTARGET_DEV + '/*.png'):
        image = misc.imread(f)
        grey = np.zeros((image.shape[0], image.shape[1]))
        for rownum in range(len(image)):
            for colnum in range(len(image[rownum])):
                grey[rownum][colnum] = weightedAverage(image[rownum][colnum])
        result_array = np.append(result_array, grey)
        edges = filter.sobel(grey)
    result_array = result_array.reshape(60,6400)
    return result_array
    
if __name__ == "__main__":
    raise NotImplementedError('This module is not executable!')
