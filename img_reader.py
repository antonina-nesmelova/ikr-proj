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

import os

from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig


def weightedAverage(pixel):
    """ Truns pixel to greyscale. """
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]


def getFeatures(target_name):
    """ Reads features from the given directory name. """
    result_array = np.array([])
    names = []
    f_count = 0
    for f in glob(target_name + '/*.png'):
        image = misc.imread(f)
        grey = np.zeros((image.shape[0], image.shape[1]))
        for rownum in range(len(image)):
            for colnum in range(len(image[rownum])):
                grey[rownum][colnum] = weightedAverage(image[rownum][colnum])
        #edges = filter.sobel(grey)
        name = '.'.join(f.split('.')[:-1])
        name = name.split(os.sep)[-1]
        names.append(name)
        result_array = np.append(result_array, grey)
        f_count += 1

    result_array = result_array.reshape(f_count,6400)
    return result_array, names

if __name__ == "__main__":
    raise NotImplementedError('This module is not executable!')
