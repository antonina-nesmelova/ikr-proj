import matplotlib.pyplot as plt
import sklearn
from ikrlib import png2fea, raw8khz2mfcc, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import scipy.linalg
import numpy as np
from numpy.random import randint
from scipy import misc
import matplotlib.pyplot as plt # import
from sklearn.decomposition import PCA
import matplotlib.cm as cm 
from glob import glob

from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

from skimage import filter
#target_dev = png2fea('target_dev/').values()
#target_train = png2fea('target_train/').values()
#non_target_dev = png2fea('non_target_dev/').values()
#non_target_train = png2fea('non_target_train/').values()

#target = np.vstack((target_dev, target_train))
#non_target = np.vstack((non_target_dev, non_target_train))

#image = misc.imread('target_train/m431_01_p01_i0_0.png')
def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

dir_name = 'target_dev/'
result_array = np.array([])
for f in glob(dir_name + '/*.png'):
    image = misc.imread(f)
    grey = np.zeros((image.shape[0], image.shape[1]))
    for rownum in range(len(image)):
        for colnum in range(len(image[rownum])):
            grey[rownum][colnum] = weightedAverage(image[rownum][colnum])
    result_array = np.append(result_array, grey)
    edges = filter.sobel(grey)
    plt.imshow(edges, cmap = cm.Greys_r) #load
    plt.show()
print(result_array.shape)
result_array = result_array.reshape(10,6400)
print(result_array.shape)
print(result_array)
A = result_array
print(A.shape)
# calculate the mean of each column
M = mean(A.T, axis=1)
print(M)
# center columns by subtracting column means
C = A - M
print(C)
# calculate covariance matrix of centered matrix
V = cov(C.T)
print(V)
# eigendecomposition of covariance matrix
values, vectors = eig(V)
print(vectors)
print(values)

print(vectors.shape)
print(values.shape)
# project data
P = vectors.T.dot(C.T)
print(P.T)
