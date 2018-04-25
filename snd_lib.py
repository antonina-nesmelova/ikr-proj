
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
import seaborn as sns
sns.set(color_codes=True)

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
    #return np.vstack(mfcc)
    #result = np.array( np.array([*k]) for n in mfcc for k in n )
    result = []
    for n in mfcc:
        for k in n:
            result.append( np.array([*k]) )
            #result = np.concatenate((result, [np.array([*k])]))
            #result.concatenate(np.array([*k]))
    return np.array(result)

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
    #plt.plot(nonfeatures_pca[:,1], nonfeatures_pca[:,0], 'r.', ms=1)
    #plt.plot(features_pca[:,1], features_pca[:,0], 'b.', ms=1)
    #plt.show()
    return features_pca, nonfeatures_pca
    



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

def train(x1, x2):
    global target_gauss
    global nontarget_gauss
    print(x1)
    print(x2)
    sigmoid  = lambda x: logistic_sigmoid(ikr.logpdf_gauss(x, mu1, cov1) + np.log(p1) - logpdf_gauss(x, mu2, cov2) - np.log(p2))
    #target_gauss.append( ikr.train_gauss(x1) )
    #nontarget_gauss.append( ikr.train_gauss(x2) )
##
    mu1, cov1 = ikr.train_gauss(x1)
    mu2, cov2 = ikr.train_gauss(x2)
    p1 = p2 = 0.5
##
    # Plot the data
    #plt.plot(x1[:,0], x1[:,1], 'r.', x2[:,0], x2[:,1], 'b.')
    #ikr.gellipse(mu1, cov1, 100, 'r')
    #ikr.gellipse(mu2, cov2, 100, 'b')
    #ax = plt.axis()
    #plt.show()
##
    m1 = 2
##
    # Initialize mean vectors to randomly selected data points from corresponding class
    mus1 = x1[np.random.randint(1, len(x1), m1)]
##
    # Initialize all covariance matrices to the same covariance matrices computed using
    # all the data from the given class
    covs1 = [cov1] * m1
##
    # Use uniform distribution as initial guess for the weights
    ws1 = np.ones(m1) / m1
##
    m2 = 2
    mus2 = x2[np.random.randint(1, len(x2), m2)]
    covs2 = [cov2] * m2
    ws2 = np.ones(m2) / m2
##
    for i in range(30):
        #plt.plot(x1[:,0], x1[:,1], 'r.', x2[:,0], x2[:,1], 'b.')
        #for w, m, c in zip(ws1, mus1, covs1): gellipse(m, c, 100, 'r', lw=round(w * 10))
        #for w, m, c in zip(ws2, mus2, covs2): gellipse(m, c, 100, 'b', lw=round(w * 10))
        ws1, mus1, covs1, ttl1 = ikr.train_gmm(x1, ws1, mus1, covs1)
        ws2, mus2, covs2, ttl2 = ikr.train_gmm(x2, ws2, mus2, covs2)
        print('Total log-likelihood: %s for class X1; %s for class X2' % (ttl1, ttl2))
        #plt.show()
##
    target_gauss = (ws1, mus1, covs1)
    nontarget_gauss = (ws2, mus2, covs2)


    

    #ikr.gellipse(target_gauss[0][0][0:2], target_gauss[0][1][0:2,0:2])
    #print(mu_f.shape)
    #print(cov_f.shape)

def classify(sample):
    score = 0
    for ws,mu,cov in zip(*target_gauss):
        score += ikr.logpdf_gauss(sample,mu,cov)*ws
    for ws,mu,cov in zip(*nontarget_gauss):
        score -= ikr.logpdf_gauss(sample,mu,cov)*ws
    return score[0] + 10
##
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

#regr_coefs = []
#def train(features, nonfeatures):
#    for dim in range(0,len(features[0])):
#        fdata = np.array( [i[dim] for i in features] )
#        ndata = np.array( [i[dim] for i in nonfeatures] )

#        fmu, fdev = np.mean(fdata,axis=0), np.std(fdata,axis=0)
#        nmu, ndev = np.mean(ndata,axis=0), np.std(ndata,axis=0)

#        mu = abs(fmu-nmu) / (nmu+fmu)
#        if fmu > nmu:
#            mu *= nmu
#            mu += nmu
#        else:
#            mu *= fmu
#            mu = fmu
        
#        dev = (fdev+ndev)/2.
#        if fmu < nmu:
#            dev *= -1

#        regr_coefs.append( (mu, dev) )

#def classify(sample):
#    score = 0
#    for i,c in enumerate(regr_coefs):
#        mu,dev = c
#        score += (sample[i] - mu) * dev * 100
#    return score