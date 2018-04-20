
import numpy as np
import scipy
import ikrlib as ikr
import matplotlib.pyplot as plt

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

def train(features, nonfeatures):
    ll_f = ikr.logpdf_gauss(features,np.mean(features,axis=0),np.var(features,axis=0))
    ll_n = ikr.logpdf_gauss(nonfeatures[0],np.mean(nonfeatures,axis=0),np.var(nonfeatures,axis=0))

    print(np.exp(ll_f))
    plt.figure()
    plt.plot(np.exp(ll_f), 'b')
    plt.plot(np.exp(ll_n), 'r')
    plt.show()

def classify(sample):
    data = np.array(sample)
    #print(data.shape)
    
    # mean
    mu = np.mean(np.mean(np.r_[data], axis=0), axis=0)
    print(mu)

    # eigen numbers and vectors
    #n, v = scipy.linalg.eigh(data)
    
    #print(n)
    