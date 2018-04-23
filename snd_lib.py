
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
    return np.vstack(mfcc)

def processFeatures(features, nonfeatures):
    cov = np.cov(np.vstack([features, nonfeatures]).T, bias=True)
    w,v = np.linalg.eig(cov)
    f_pca = features.dot(v)
    nf_pca = nonfeatures.dot(v)
    return f_pca, nf_pca

def plotFeatures(features, nonfeatures, x=4,y=3):
    # verticalize inputs
    features = np.vstack(features)
    nonfeatures = np.vstack(nonfeatures)
    # get size
    dim = features.shape[1]
    n = len(features)
    # covariant matrix of all
    cov = np.cov(np.vstack([features, nonfeatures]).T, bias=True)
    # take 2 largest eigenvalues and corresponding eigenvectors
    df, ef = scipy.linalg.eigh(cov, eigvals=(dim-x, dim-y))
    # count pca
    features_pca = features.dot(ef)
    nonfeatures_pca = nonfeatures.dot(ef)
    # show
    plt.plot(nonfeatures_pca[:,1], nonfeatures_pca[:,0], 'r.', ms=1)
    plt.plot(features_pca[:,1], features_pca[:,0], 'b.', ms=1)
    plt.show()
    



#def train(features, nonfeatures):
#    features = np.array(features)
#    nonfeatures = np.array(nonfeatures)
#    print(features)
#    print(nonfeatures)
#    samples = np.r_[features, nonfeatures]
#    legend = np.r_[np.ones(len(features)),np.zeros(len(nonfeatures))]
#    plt.plot(features[:,0], features[:,1], 'r.', nonfeatures[:,0], nonfeatures[:,1], 'b.')
#    ax = plt.axis()
#    plt.show()
#
#    w, w0, data_cov = ikr.train_generative_linear_classifier(samples, legend)
#    x1, x2 = ax[:2]
#    y1 = (-w0 - (w[0] * x1)) / w[1]
#    y2 = (-w0 - (w[0] * x2)) / w[1]
#
#    plt.plot(samples[:,0], red_data[:,1], 'r.', blue_data[:,0], blue_data[:,1], 'b.')
#    plt.plot([x1, x2], [y1, y2], 'k', linewidth=2)
#    ikr.gellipse(np.mean(red_data, axis=0), data_cov, 100, 'r')
#    ikr.gellipse(np.mean(blue_data, axis=0), data_cov, 100, 'r')
#    plt.show()
#
#    for i in range(100):
#        ikr.plot2dfun(lambda x: ikr.logistic_sigmoid(x.dot(w) + w0), ax, 1000)
#        plt.plot(red_data[:,0], red_data[:,1], 'r.', blue_data[:,0], blue_data[:,1], 'b.')
#        plt.show()
#        w, w0 = ikr.train_linear_logistic_regression(x, t, w, w0)

POSTER = 0.5
POSTER_NON = 1 - POSTER
target_gauss = []
nontarget_gauss = []

def train(features, nonfeatures):
    target_gauss.append( ikr.train_gauss(features) )
    nontarget_gauss.append( ikr.train_gauss(nonfeatures) )

    ikr.gellipse(target_gauss[0][0][0:2], target_gauss[0][1][0:2,0:2])
    #print(mu_f.shape)
    #print(cov_f.shape)

def classify(sample):
    
    # mean
    t = []
    for i in target_gauss:
        t.append(*ikr.logpdf_gauss(sample, i[0], i[1]))
    n = []
    for i in nontarget_gauss:
        n.append(*ikr.logpdf_gauss(sample, i[0], i[1]))
    
    return max(max(t), max(n)) + 495

    # eigen numbers and vectors
    #n, v = scipy.linalg.eigh(data)
    
    #print(n)