import numpy as np
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import scipy.linalg
from ikrlib import gellipse, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import matplotlib.pyplot as plt
from numpy.random import randint


def getGauss(target, nonetarget):
    # dostaneme vlastni vektory##################
    result_array = np.vstack([target, nonetarget])
    A = result_array
    # calculate the mean of each column
    M = mean(A.T, axis=1)
    print(M)

    #B = test_target
    #M1 = mean(B.T, axis=1)

    #D = test_nonetarget
    #M2 = mean(D.T, axis=1)

    D = nonetarget
    M3 = mean(D.T, axis=1)

    D = target
    # calculate the mean of each column
    M4 = mean(D.T, axis=1)

    # center columns by subtracting column means
    C = A - M
    # calculate covariance matrix of centered matrix
    V = cov(C.T)
    # eigendecomposition of covariance matrix
    dim = target.shape[1]
    print('Getting vectors')
    values, vectors = scipy.linalg.eigh(V,  eigvals=(dim-2, dim-1))
    values1, vectors1 = scipy.linalg.eigh(V,  eigvals=(dim-3, dim-2))

    # gausovky pro prvni dva vlastni vektory ###################
    tar = (target - M4).dot(vectors)
    ntar = (nonetarget - M3).dot(vectors)
    print(tar)

    for i in range(tar.shape[0]):
        tar[i][0], tar[i][1] = tar[i][1], tar[i][0]
        print(tar[i])

    for i in range(ntar.shape[0]):
        ntar[i][0], ntar[i][1] = ntar[i][1], ntar[i][0]
        print(ntar[i])

    mu1, cov1 = train_gauss(tar)
    mu2, cov2 = train_gauss(ntar)

    plt.plot(tar[:,0], tar[:,1], 'r.', ntar[:,0], ntar[:,1], 'b.')
    gellipse(mu1, cov1, 100, 'r')
    gellipse(mu2, cov2, 100, 'b')
    ax = plt.axis()
    plt.show()

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

    m2 = 3
    mus2 = ntar[randint(1, len(ntar), m2)]
    covs2 = [cov2] * m2
    ws2 = np.ones(m2) / m2


    for i in range(30):
        ws1, mus1, covs1, ttl1 = train_gmm(tar, ws1, mus1, covs1)
        ws2, mus2, covs2, ttl2 = train_gmm(ntar, ws2, mus2, covs2)
        print('Total log-likelihood: %s for class X1;' % (ttl1))

    plt.figure(1);
    plt.plot(tar[:,0], tar[:,1], 'r.', ntar[:,0], ntar[:,1], 'b.')
    for w, m, c in zip(ws1, mus1, covs1): gellipse(m, c, 3000, 'r', lw=round(w * 10))
    for w, m, c in zip(ws2, mus2, covs2): gellipse(m, c, 3000, 'b', lw=round(w * 10))
    plt.show()

    fw = ws1
    fm = mus1
    fc = covs1

    # gausovky pro 3 a 4 vv##############
    tar = np.vstack((target - M4).dot(vectors1))
    ntar = np.vstack((nonetarget - M3).dot(vectors1))

    for i in range(tar.shape[0]):
        tar[i][0], tar[i][1] = tar[i][1], tar[i][0]
        print(tar[i])

    for i in range(ntar.shape[0]):
        ntar[i][0], ntar[i][1] = ntar[i][1], ntar[i][0]
        print(ntar[i])

    mu1, cov1 = train_gauss(tar)
    mu2, cov2 = train_gauss(ntar)

    plt.plot(tar[:,0], tar[:,1], 'r.', ntar[:,0], ntar[:,1], 'b.')
    gellipse(mu1, cov1, 100, 'r')
    gellipse(mu2, cov2, 100, 'b')
    ax = plt.axis()
    plt.show()

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


    m2 = 3
    mus2 = ntar[randint(1, len(ntar), m2)]
    covs2 = [cov2] * m2
    ws2 = np.ones(m2) / m2


    for i in range(30):
        ws1, mus1, covs1, ttl1 = train_gmm(tar, ws1, mus1, covs1)
        ws2, mus2, covs2, ttl2 = train_gmm(ntar, ws2, mus2, covs2)
        print('Total log-likelihood: %s for class X1;' % (ttl1))
    
    plt.figure(2);
    plt.plot(tar[:,0], tar[:,1], 'r.', ntar[:,0], ntar[:,1], 'b.')
    for w, m, c in zip(ws1, mus1, covs1): gellipse(m, c, 3000, 'r', lw=round(w * 10))
    for w, m, c in zip(ws2, mus2, covs2): gellipse(m, c, 3000, 'b', lw=round(w * 10))
    plt.show()

    sw = ws1
    sm = mus1
    sc = covs1

    return [fw, sw], [fm, sm], [fc, sc], vectors, vectors1

def multByVectors(data, vectors):
    M = mean(data.T, axis=1)
    data = (data - M).dot(vectors)
    for i in range(data.shape[0]):
        data[i][0], data[i][1] = data[i][1], data[i][0]
        print(data[i])
    return data

def getScore(test, ws, mus, covs, vector):

    test = multByVectors(test, vector)

    t = 0.5
    nt = 1 - t   

    score=[]
    for tst in test:
        ll1 = logpdf_gmm(tst, ws, mus, covs)
        score.append(sum(ll1) + np.log(nt))
    print(score)




if __name__ == "__main__":
    raise NotImplementedError('This module is not executable!')