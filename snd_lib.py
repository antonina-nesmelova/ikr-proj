
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
    Loads *.wav files at given directory.
    Extracts MFCC features (13), returns 3D numpy array
    and python list of names of the files.
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
            result2.append( np.array(list(k)) )
        result.append( np.array(result2) )
        res_names.append(n) 
    return np.array(result), res_names


def train(x1, x2):
    """
    Trains sound classifier with given data.
    x1 has positive result, x2 has negative result.
    """
    assert( isinstance(x1,np.ndarray) )
    assert( isinstance(x2,np.ndarray) )
    global target_gauss
    global nontarget_gauss
    # initial gauss
    mu1, cov1 = ikr.train_gauss(x1)
    mu2, cov2 = ikr.train_gauss(x2)
    # apriori probability
    p1 = p2 = 0.5
    
    # initial count of class1 gausses
    m1 = 2
    # initialize mean vectors randomly
    mus1 = x1[np.random.randint(1, len(x1), m1)]
    # initialize the covariance matrices as the initial
    covs1 = [cov1] * m1
    # initialize uniform weights
    ws1 = np.ones(m1) / m1

    # initialization of class 2 property
    m2 = 2
    mus2 = x2[np.random.randint(1, len(x2), m2)]
    covs2 = [cov2] * m2
    ws2 = np.ones(m2) / m2

    # train cycle
    for i in range(35):
        ws1, mus1, covs1, ttl1 = ikr.train_gmm(x1, ws1, mus1, covs1)
        ws2, mus2, covs2, ttl2 = ikr.train_gmm(x2, ws2, mus2, covs2)
        #print('Total log-likelihood: %s for class X1; %s for class X2' % (ttl1, ttl2))

    # distribute trained gausses
    target_gauss = (ws1, mus1, covs1)
    nontarget_gauss = (ws2, mus2, covs2)

    # save train data
    tdir = 'train'+os.sep
    if not os.path.exists(tdir):
        os.makedirs(tdir)
    np.save(tdir+'ws1.npy',ws1)
    np.save(tdir+'mus1.npy',mus1)
    np.save(tdir+'covs1.npy',covs1)
    np.save(tdir+'ws2.npy',ws2)
    np.save(tdir+'mus2.npy',mus2)
    np.save(tdir+'covs2.npy',covs2)

def load_trained():
    """
    Loads train data from previous training from train/ directory.
    """
    global target_gauss
    global nontarget_gauss
    # check train dir
    tdir = 'train'+os.sep
    if not os.path.exists(tdir):
        raise IOError('no train directory')
    # load train data
    ws1 = np.load(tdir+'ws1.npy')
    mus1 = np.load(tdir+'mus1.npy')
    covs1 = np.load(tdir+'covs1.npy')
    ws2 = np.load(tdir+'ws2.npy')
    mus2 = np.load(tdir+'mus2.npy')
    covs2 = np.load(tdir+'covs2.npy')
    # connect to the classifier
    target_gauss = (ws1, mus1, covs1)
    nontarget_gauss = (ws2, mus2, covs2)



def classify(record):
    """
    Classifies the given record. Returns softmax score.
    """
    assert(isinstance(record,np.ndarray))
    # count score
    score = 0
    for sample in record:
        mscore = 0
        for ws,mu,cov in zip(*target_gauss):
            mscore += ikr.logpdf_gauss(sample,mu,cov)*ws
        for ws,mu,cov in zip(*nontarget_gauss):
            mscore -= ikr.logpdf_gauss(sample,mu,cov)*ws
        score += mscore
    return score + 2000
