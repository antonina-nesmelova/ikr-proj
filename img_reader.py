#!/usr/bin/env python3

import matplotlib.pyplot as plt
import sklearn
from ikrlib import png2fea, raw8khz2mfcc, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import scipy.linalg
import numpy as np
from numpy.random import randint
from scipy import misc
import matplotlib.pyplot as plt # import
from sklearn.decomposition import PCA
import matplotlib.cm as cm 
from glob import glob

from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

from skimage import filters

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
        edges = filters.sobel(grey)
    result_array = result_array.reshape(10,6400)
    A = result_array
    print(A.shape)
    # calculate the mean of each column
    M = mean(A.T, axis=1)
    print(M)
    # center columns by subtracting column means
    C = A - M
    print(C)
    # calculate covariance matrix of centered matrix
    V = cov(C.T)
    print(V)
    # eigendecomposition of covariance matrix
    values, vectors = eig(V)
    return values, vectors 

def getTrain_NonTargetFeatures():
    result_array = np.array([])
    for f in glob(NONTARGET_TRAIN + '/*.png'):
        image = misc.imread(f)
        grey = np.zeros((image.shape[0], image.shape[1]))
        for rownum in range(len(image)):
            for colnum in range(len(image[rownum])):
                grey[rownum][colnum] = weightedAverage(image[rownum][colnum])
        result_array = np.append(result_array, grey)
        edges = filters.sobel(grey)
    result_array = result_array.reshape(10,6400)
    A = result_array
    print(A.shape)
    # calculate the mean of each column
    M = mean(A.T, axis=1)
    print(M)
    # center columns by subtracting column means
    C = A - M
    print(C)
    # calculate covariance matrix of centered matrix
    V = cov(C.T)
    print(V)
    # eigendecomposition of covariance matrix
    values, vectors = eig(V)
    return values, vectors 

def getTest_TargetFeatures():
    result_array = np.array([])
    for f in glob(TARGET_DEV + '/*.png'):
        image = misc.imread(f)
        grey = np.zeros((image.shape[0], image.shape[1]))
        for rownum in range(len(image)):
            for colnum in range(len(image[rownum])):
                grey[rownum][colnum] = weightedAverage(image[rownum][colnum])
        result_array = np.append(result_array, grey)
        edges = filters.sobel(grey)
    result_array = result_array.reshape(10,6400)
    A = result_array
    print(A.shape)
    # calculate the mean of each column
    M = mean(A.T, axis=1)
    print(M)
    # center columns by subtracting column means
    C = A - M
    print(C)
    # calculate covariance matrix of centered matrix
    V = cov(C.T)
    print(V)
    # eigendecomposition of covariance matrix
    values, vectors = eig(V)
    return values, vectors 

def getTest_NonTargetFeatures():
    result_array = np.array([])
    for f in glob(NONTARGET_DEV + '/*.png'):
        image = misc.imread(f)
        grey = np.zeros((image.shape[0], image.shape[1]))
        for rownum in range(len(image)):
            for colnum in range(len(image[rownum])):
                grey[rownum][colnum] = weightedAverage(image[rownum][colnum])
        result_array = np.append(result_array, grey)
        edges = filters.sobel(grey)
    result_array = result_array.reshape(10,6400)
    A = result_array
    print(A.shape)
    # calculate the mean of each column
    M = mean(A.T, axis=1)
    print(M)
    # center columns by subtracting column means
    C = A - M
    print(C)
    # calculate covariance matrix of centered matrix
    V = cov(C.T)
    print(V)
    # eigendecomposition of covariance matrix
    values, vectors = eig(V)
    return values, vectors 


if __name__ == "__main__":
    raise NotImplementedError('This module is not executable!')
