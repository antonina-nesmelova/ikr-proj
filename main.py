#!/usr/bin/env python3

import snd_reader as sr
import img_reader as img
import train_img as train
import numpy as np


def main():
    target = img.getTrain_TargetFeatures()
    nonetarget = img.getTrain_NonTargetFeatures()
    test_target = img.getTest_TargetFeatures()
    test_nonetarget = img.getTest_NonTargetFeatures()
    print('Get gauss')
    w, m, c, v1, v2 = train.getGauss(np.vstack([target, target]), np.vstack([nonetarget, nonetarget]))
    print(w)
    print(m)
    print(c)
    train.getScore(test_target, w[0], m[0], c[0], v1)
    train.getScore(test_target, w[1], m[1], c[1], v2)


if __name__ == '__main__':
    main()