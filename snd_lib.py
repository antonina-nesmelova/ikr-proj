
"""
This is library for whole sound classifier.
"""

import os
import ikrlib as ikr
import numpy as np
import matplotlib.pyplot as plt
import scipy

TARGET_TRAIN = 'data' + os.sep + 'target_train'
NONTARGET_TRAIN = 'data' + os.sep + 'non_target_train'
TARGET_DEV = 'data' + os.sep + 'target_dev'
NONTARGET_DEV = 'data' + os.sep + 'non_target_dev'

def getFeatures(directory):
    """
    Loads extracted features *.wav files at given directory.
    """
    # get mfcc features
    mfcc = ikr.wav16khz2mfcc(directory).values()

    # return in lists
    #features = []
    #for coef in mfcc:
    #    features.append(coef.tolist())

    return np.vstack(mfcc)

def plotFeatures(features, nonfeatures):
    # verticalize inputs
    features = np.vstack(features)
    nonfeatures = np.vstack(nonfeatures)
    # get size
    dim = features.shape[1]
    n = len(features)

    # covariant matrix of all
    cov = np.cov(np.vstack([features, nonfeatures]).T, bias=True)
    # take 2 largest eigenvalues and corresponding eigenvectors
    df, ef = scipy.linalg.eigh(cov, eigvals=(dim-2, dim-1))

    # count pca
    features_pca = features.dot(ef)
    nonfeatures_pca = nonfeatures.dot(ef)

    # show
    plt.plot(nonfeatures_pca[:,1], nonfeatures_pca[:,0], 'r.', ms=1)
    plt.plot(features_pca[:,1], features_pca[:,0], 'b.', ms=1)
    plt.show()
