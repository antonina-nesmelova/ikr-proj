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

<<<<<<< HEAD
=======
from skimage import filters
>>>>>>> master

def weightedAverage(pixel):
    """ Truns pixel to greyscale. """
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]


def getFeatures(target_name):
    result_array = np.array([])
<<<<<<< HEAD
    names = np.array([])
=======
    name_array = []

    # File counter for result array reshaping
>>>>>>> master
    f_count = 0
    for f in glob(target_name + '/*.png'):
        image = misc.imread(f)
        grey = np.zeros((image.shape[0], image.shape[1]))
        for rownum in range(len(image)):
            for colnum in range(len(image[rownum])):
                grey[rownum][colnum] = weightedAverage(image[rownum][colnum])
<<<<<<< HEAD
        #edges = filter.sobel(grey)
        names = np.append(names, f)
        result_array = np.append(result_array, grey)
        f_count += 1

    result_array = result_array.reshape(f_count,6400)
    return result_array, names
=======
        filtered_gray = filters.sobel(grey)
        result_array = np.append(result_array, filtered_gray)

        # Creating name array for future dictionary
        name_array.append(f)
        f_count += 1

    result_array = result_array.reshape(f_count,6400)

    return name_array, result_array
>>>>>>> master

if __name__ == "__main__":
    raise NotImplementedError('This module is not executable!')
