import numpy as np
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import scipy.linalg
from ikrlib import gellipse, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import matplotlib.pyplot as plt
from numpy.random import randint

RANGE = 30

def transformData(data, v, transform=True):
    D = data
    M = mean(D.T, axis=1)

    data = np.vstack((data - M).dot(v))
    if (transform):
        for i in range(data.shape[0]):
            data[i][0], data[i][1] = data[i][1], data[i][0]

    return data

def getVectors(target, nonetarget):
    
    result_array = np.vstack([target, nonetarget])
    A = result_array
    # calculate the mean of each column
    M = mean(A.T, axis=1)


    # center columns by subtracting column means
    C = A - M
    # calculate covariance matrix of centered matrix
    V = cov(C.T)
    # eigendecomposition of covariance matrix
    dim = target.shape[1]
    print('Getting vectors')
    values, vectors = scipy.linalg.eigh(V,  eigvals=(dim-2, dim-1))
    values1, vectors1 = scipy.linalg.eigh(V,  eigvals=(dim-3, dim-2))
    values2, vectors2 = scipy.linalg.eigh(V,  eigvals=(dim-4, dim-3))

    return vectors, vectors1, vectors2

def getGauss(target, nonetarget, vectors, vectors1, vectors2, test_target, test_nonetarget):
    # dostaneme vlastni vektory##################

    # calculate the mean of each column
    #D = nonetarget
    #M3 = mean(D.T, axis=1)

    #D = target
    #M4 = mean(D.T, axis=1)

    # gausovky pro prvni dva vlastni vektory ###################
    #tar = (target - M4).dot(vectors)
    #ntar = (nonetarget - M3).dot(vectors)
    #print(tar)

    #for i in range(tar.shape[0]):
    #    tar[i][0], tar[i][1] = tar[i][1], tar[i][0]
        #print(tar[i])

    #for i in range(ntar.shape[0]):
    #    ntar[i][0], ntar[i][1] = ntar[i][1], ntar[i][0]
        #print(ntar[i])

    tar = transformData(target,vectors)
    ntar = transformData(nonetarget,vectors)
    ttar = transformData(test_target,vectors)
    tntar = transformData(test_nonetarget,vectors)

    tar = np.vstack([tar, ttar])
    mu1, cov1 = train_gauss(tar)
    #mu2, cov2 = train_gauss(ntar)

    #plt.plot(tar[:,0], tar[:,1], 'r.', ntar[:,0], ntar[:,1], 'b.')
    #gellipse(mu1, cov1, 100, 'r')
    #gellipse(mu2, cov2, 100, 'b')
    #ax = plt.axis()
    #plt.show()

    p1 = p2 = 0.5
    # Train and test with GMM models with full covariance matrices
    #Decide for number of gaussian mixture components used for the model
    m1 = 2

    # Initialize mean vectors to randomly selected data points from corresponding class
    mus1 = tar[randint(1, len(tar), m1)]

    # Initialize all covariance matrices to the same covariance matrices computed using
    # all the data from the given class
    covs1 = [cov1] * m1

    # Use uniform distribution as initial guess for the weights
    ws1 = np.ones(m1) / m1

    #m2 = 3
    #mus2 = ntar[randint(1, len(ntar), m2)]
    #covs2 = [cov2] * m2
    #ws2 = np.ones(m2) / m2


    for i in range(RANGE):
        ws1, mus1, covs1, ttl1 = train_gmm(tar, ws1, mus1, covs1)
    #    ws2, mus2, covs2, ttl2 = train_gmm(ntar, ws2, mus2, covs2)
        print('Total log-likelihood: %s for class X1;' % (ttl1))

    tar = np.vstack([tar, ttar])
    ntar = np.vstack([ntar, tntar])

    plt.figure('First two vectors');
    plt.plot(tar[:,0], tar[:,1], 'r.', ntar[:,0], ntar[:,1], 'b.')
    for w, m, c in zip(ws1, mus1, covs1): gellipse(m, c, 3000, 'r', lw=round(w * 10))
    #for w, m, c in zip(ws2, mus2, covs2): gellipse(m, c, 3000, 'b', lw=round(w * 10))
    

    fw = ws1
    fm = mus1
    fc = covs1

    # gausovky pro 3 a 4 vv##############
    
    tar = transformData(target,vectors1, False)
    ntar = transformData(nonetarget,vectors1, False)
    ttar = transformData(test_target,vectors1, False)
    tntar = transformData(test_nonetarget,vectors1, False)
    tar = np.vstack([tar, ttar])

    mu1, cov1 = train_gauss(tar)

    p1 = p2 = 0.5
    # Train and test with GMM models with full covariance matrices
    #Decide for number of gaussian mixture components used for the model
    m1 = 2

    # Initialize mean vectors to randomly selected data points from corresponding class
    mus1 = tar[randint(1, len(tar), m1)]

    # Initialize all covariance matrices to the same covariance matrices computed using
    # all the data from the given class
    covs1 = [cov1] * m1

    # Use uniform distribution as initial guess for the weights
    ws1 = np.ones(m1) / m1

    for i in range(RANGE):
        ws1, mus1, covs1, ttl1 = train_gmm(tar, ws1, mus1, covs1)
    #    ws2, mus2, covs2, ttl2 = train_gmm(ntar, ws2, mus2, covs2)
        print('Total log-likelihood: %s for class X1;' % (ttl1))
    tar = np.vstack([tar, ttar])
    ntar = np.vstack([ntar, tntar])

    plt.figure('Second two vectors')
    plt.plot(tar[:,0], tar[:,1], 'r.', ntar[:,0], ntar[:,1], 'b.')
    for w, m, c in zip(ws1, mus1, covs1): gellipse(m, c, 3000, 'r', lw=round(w * 10))
    #for w, m, c in zip(ws2, mus2, covs2): gellipse(m, c, 3000, 'b', lw=round(w * 10))
    

    sw = ws1
    sm = mus1
    sc = covs1

        # gausovky pro 3 a 4 vv##############
    tar = transformData(target,vectors2)
    ntar = transformData(nonetarget,vectors2)
    ttar = transformData(test_target,vectors2)
    tntar = transformData(test_nonetarget,vectors2)
    tar = np.vstack([tar, ttar])

    mu1, cov1 = train_gauss(tar)
    #mu2, cov2 = train_gauss(ntar)

    #plt.plot(tar[:,0], tar[:,1], 'r.', ntar[:,0], ntar[:,1], 'b.')
    #gellipse(mu1, cov1, 100, 'r')
    #gellipse(mu2, cov2, 100, 'b')
    #ax = plt.axis()
    #plt.show()

    p1 = p2 = 0.5
    # Train and test with GMM models with full covariance matrices
    #Decide for number of gaussian mixture components used for the model
    m1 = 3

    # Initialize mean vectors to randomly selected data points from corresponding class
    mus1 = tar[randint(1, len(tar), m1)]

    # Initialize all covariance matrices to the same covariance matrices computed using
    # all the data from the given class
    covs1 = [cov1] * m1

    # Use uniform distribution as initial guess for the weights
    ws1 = np.ones(m1) / m1


    #m2 = 3
    #mus2 = ntar[randint(1, len(ntar), m2)]
    #covs2 = [cov2] * m2
    #ws2 = np.ones(m2) / m2


    for i in range(RANGE):
        ws1, mus1, covs1, ttl1 = train_gmm(tar, ws1, mus1, covs1)
    #    ws2, mus2, covs2, ttl2 = train_gmm(ntar, ws2, mus2, covs2)
        print('Total log-likelihood: %s for class X1;' % (ttl1))

    
    ntar = np.vstack([ntar, tntar])
    
    plt.figure('Third two vectors');
    plt.plot(tar[:,0], tar[:,1], 'r.', ntar[:,0], ntar[:,1], 'b.')
    for w, m, c in zip(ws1, mus1, covs1): gellipse(m, c, 3000, 'r', lw=round(w * 10))
    #for w, m, c in zip(ws2, mus2, covs2): gellipse(m, c, 3000, 'b', lw=round(w * 10))
    plt.show()

    tw = ws1
    tm = mus1
    tc = covs1

    return [fw, sw, tw], [fm, sm, tm], [fc, sc, tc]

def multByVectors(data, vectors):
    M = mean(data.T, axis=1)
    data = (data - M).dot(vectors)
    for i in range(data.shape[0]):
        data[i][0], data[i][1] = data[i][1], data[i][0]
        #print(data[i])
    return data

def getScore(test, ws, mus, covs, vector):
    print('In getScore')
    #for i in range(3):
    #    print(i)
    test = transformData(test, vector, False)

    t = 0.5
    nt = 1 - t   

    score=[]
    for tst in test:
        print('In for')
        ll1 = logpdf_gmm(tst, ws, mus, covs)
        #print(ll1)
        #ll2 = logpdf_gmm(tst, ws[1], mus[1], covs[1])
        #print(ll2)
        #ll3 = logpdf_gmm(tst, ws[2], mus[2], covs[2])
        #print(ll3)
        print('Score')
        #print(ll1)
        #print(ll2)
        #print(ll3)
        score.append(sum(ll1) + np.log(nt))
    print(score)
    return score




if __name__ == "__main__":
    raise NotImplementedError('This module is not executable!')