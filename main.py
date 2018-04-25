#!/usr/bin/env python3

import snd_reader as sr
import img_reader as img
import train_img as train
import numpy as np
import matplotlib.pyplot as plt

from operator import add

TARGET_DEV = 'data/target_dev/'
NONTARGET_DEV = 'data/non_target_dev'
TARGET_TRAIN = 'data/target_train'
NONTARGET_TRAIN = 'data/non_target_train'

def main():
    target = img.getFeatures(TARGET_DEV)
    nonetarget = img.getFeatures(NONTARGET_DEV)
    test_target = img.getFeatures(TARGET_TRAIN)
    test_nonetarget = img.getFeatures(NONTARGET_TRAIN)
    print('Get gauss')
    w, m, c, v1, v2 = train.getGauss(target, nonetarget)
    print(w)
    print(m)
    print(c)
    skt1 = train.getScore(test_target, w[0], m[0], c[0], v1)
    skn1 = train.getScore(test_nonetarget, w[0], m[0], c[0], v1)
    skt2 = train.getScore(test_target, w[1], m[1], c[1], v2)
    skn2 = train.getScore(test_nonetarget, w[1], m[1], c[1], v2)

    skt3 = train.getScore(target, w[0], m[0], c[0], v1)
    skn3 = train.getScore(nonetarget, w[0], m[0], c[0], v1)
    skt4 = train.getScore(target, w[1], m[1], c[1], v2)
    skn4 = train.getScore(nonetarget, w[1], m[1], c[1], v2)

    score_test_target = map(add, skt1, skt2) 
    score_test_nonetarget = map(add, skn1, skn2)

    score_target = map(add, skt3, skt4)
    score_nonetarget = map(add, skn3, skn4)

    target_ok = 0
    nonetarget_ok = 0

    print(score_test_target)
    print(len(score_test_target))
    print(score_target)
    print(len(score_target))
    print(score_test_nonetarget)
    print(len(score_test_nonetarget))
    print(score_nonetarget)
    print(len(score_nonetarget))

    for t in score_test_target:
        if t > -40:
            target_ok = target_ok + 1

    for t in score_target:
        if t > -40:
            target_ok = target_ok + 1

    for t in score_test_nonetarget:
        if t <= -40:
            nonetarget_ok = nonetarget_ok + 1 

    for t in score_nonetarget:
        if t <= -40:
            nonetarget_ok = nonetarget_ok + 1

    print("TargetOk%: {}/{}={}".format(target_ok, len(score_test_target + score_target), float(target_ok)/len(score_test_target + score_target)))
    print("NontargetOk: {}/{}={}".format(nonetarget_ok, len(score_nonetarget + score_test_nonetarget), float(nonetarget_ok)/len(score_nonetarget + score_test_nonetarget)))

    plt.figure(1)
    plt.plot(score_test_target, 'r.', score_test_nonetarget, 'b.')

    plt.figure(2)
    plt.plot(score_target, 'r.', score_nonetarget, 'b.')

    plt.show()


if __name__ == '__main__':
    main()