
import numpy as np

def train(features, nonfeatures):
    pass

def classify(sample):
    s = np.array(sample)
    print(s.shape)
    # eigen numbers and vectors
    n, v = np.linalg.eigh(s)
    print(n)
    