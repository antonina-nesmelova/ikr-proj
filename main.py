#!/usr/bin/env python3

import img_reader as img
import train_img as train
import snd_lib as lib
import numpy as np
import sys

import snd_reader as sr
import matplotlib.pyplot as plt

from operator import add

TARGET_DEV = 'data/target_dev/'
NONTARGET_DEV = 'data/non_target_dev'
TARGET_TRAIN = 'data/target_train'
NONTARGET_TRAIN = 'data/non_target_train'

TRAIN = True

def mergeWithin(x):
    res = []
    for n in x:
        for i in n:
            res.append(i)
    return np.array(res)

def getSoundScore():

    # get data
    target,_ = lib.getFeatures( lib.TARGET_TRAIN )
    nontarget,_ = lib.getFeatures( lib.NONTARGET_TRAIN )
    
    # this works just wierd
    #target,nontarget = lib.processFeatures(target,nontarget)

    #lib.train( np.array(list(target.values())), np.array(list(nontarget.values())) )
    lib.train(mergeWithin(target), mergeWithin(nontarget))

    if TRAIN:
        # validate target
        target, target_name = lib.getFeatures( lib.TARGET_DEV )
        target_score = {}
        for i,record in enumerate(target):
            target_score[ target_name[i] ] = lib.classify( record )

        # validate nontarget
        nontarget, nontarget_name = lib.getFeatures( lib.NONTARGET_DEV )
        nontarget_score = {}
        for i,record in enumerate(nontarget):
            nontarget_score[ nontarget_name[i] ] = lib.classify( record )

        # evaluate score
        ts = 0
        for c in target_score.values():
            if c > 0:
                ts += 1
        ns = 0
        for c in nontarget_score.values():
            if c <= 0:
                ns += 1

        print("target score:", ts/len(target_score) *100 )
        print("nontarget score:", ns/len(nontarget_score) *100 )

        #print(target)
        #print(nontarget)
        


    else:
        pass
        # read real data

   

def getImageScore():
    
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



def fusion():
    soundSc = getSoundScore()
    imgSc = getImageScore()

    assert(len(soundSc) == len(imgSc))

    result = {k: [v1, imgSc[k]] for k, v1 in soundSc.values()}
    for file, results in result.values():
        pass
        #print(f"File: - {file}\nSound {result[0]}\tImage{result[1]}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: ./main [--image | --sound]', file=sys.stderr)
        exit() 
    if sys.argv[1] == '--image':
        getImageScore()
    elif sys.argv[1] == '--sound':
        getSoundScore()
    else:
        print('Usage: ./main [--image | --sound]', file=sys.stderr)
        exit()