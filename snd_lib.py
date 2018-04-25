
"""
This is library for whole sound classifier.
"""

import os
import ikrlib as ikr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy
import math

TARGET_TRAIN = 'data' + os.sep + 'target_train'
NONTARGET_TRAIN = 'data' + os.sep + 'non_target_train'
TARGET_DEV = 'data' + os.sep + 'target_dev'
NONTARGET_DEV = 'data' + os.sep + 'non_target_dev'
sigmoid = lambda x: logistic_sigmoid(ikr.logpdf_gauss(x, mu1, cov1) + np.log(p1) - logpdf_gauss(x, mu2, cov2) - np.log(p2))

target_gauss = []
nontarget_gauss = []

def getFeatures(directory):
    """
    Loads extracted features *.wav files at given directory.
    """
    # get mfcc features
    mfcc = ikr.wav16khz2mfcc(directory)
    result = []
    res_names = []
    # go around the files
    for n in mfcc.keys():
        result2 = []
        # go around the feature vectors in one file
        for k in mfcc[n]:
            result2.append( np.array([*k]) )
        result.append( np.array(result2) )
        res_names.append(n) 
    return np.array(result), res_names


def train(x1, x2):
    global target_gauss
    global nontarget_gauss
    mu1, cov1 = ikr.train_gauss(x1)
    mu2, cov2 = ikr.train_gauss(x2)
    p1 = p2 = 0.5
    m1 = 2
    # Initialize mean vectors to randomly selected data points from corresponding class
    mus1 = x1[np.random.randint(1, len(x1), m1)]
    # Initialize all covariance matrices to the same covariance matrices computed using
    # all the data from the given class
    covs1 = [cov1] * m1
    # Use uniform distribution as initial guess for the weights
    ws1 = np.ones(m1) / m1
##
    m2 = 2
    mus2 = x2[np.random.randint(1, len(x2), m2)]
    covs2 = [cov2] * m2
    ws2 = np.ones(m2) / m2
##
    for i in range(30):
        ws1, mus1, covs1, ttl1 = ikr.train_gmm(x1, ws1, mus1, covs1)
        ws2, mus2, covs2, ttl2 = ikr.train_gmm(x2, ws2, mus2, covs2)
        #print('Total log-likelihood: %s for class X1; %s for class X2' % (ttl1, ttl2))
##
    target_gauss = (ws1, mus1, covs1)
    nontarget_gauss = (ws2, mus2, covs2)


def classify(record):
    score = 0
    for sample in record:
        mscore = 0
        for ws,mu,cov in zip(*target_gauss):
            mscore += ikr.logpdf_gauss(sample,mu,cov)*ws
        for ws,mu,cov in zip(*nontarget_gauss):
            mscore -= ikr.logpdf_gauss(sample,mu,cov)*ws
        score += mscore
    return score + 2000
