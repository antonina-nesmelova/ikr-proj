#!/usr/bin/env python3

import img_reader as img
import train_img as train
import snd_lib as snd
import numpy as np
import sys
import os

import matplotlib.pyplot as plt

from operator import add

from operator import add

TARGET_DEV = 'data/target_dev/'
NONTARGET_DEV = 'data/non_target_dev'
TARGET_TRAIN = 'data/target_train'
NONTARGET_TRAIN = 'data/non_target_train'

REALDATA = False
train = False

def getSoundScore():
    """
    Trains and saves classifier, or loads coefficients to the classifier.
    Counts sound score of the data, returns result {'filename': softmax score}.
    """

    def mergeWithin(x):
        """
        Merge numpy list of lists into list.        
        """
        res = []
        for n in x:
            for i in n:
                res.append(i)
        return np.array(res)

    # train classifier
    if train:
        # get data
        target,_ = snd.getFeatures( snd.TARGET_TRAIN )
        nontarget,_ = snd.getFeatures( snd.NONTARGET_TRAIN )
        # train
        snd.train(mergeWithin(target), mergeWithin(nontarget))
    # load classifier
    else:
        snd.load_trained()

    # read real data
    if REALDATA:
        loc = 'data'+os.sep+'eval'
        data,dataname = snd.getFeatures(loc)
        score = {}
        for i,d in enumerate(data):
            score[ dataname[i] ] = snd.classify(d)
        for k in score.keys():
            print(str(k)+' : '+str(score[k]))
        for k in score.keys():
            if score[k] > 0:
                print(k)
        return score
    # cross validation
    else:
        # validate target
        target, target_name = snd.getFeatures( snd.TARGET_DEV )
        target_score = {}
        # validate nontarget
        nontarget, nontarget_name = snd.getFeatures( snd.NONTARGET_DEV )
        nontarget_score = {}
        max_score = (0,0)
        treshold = 0
        for n in range(1000,5000,100):
            snd.move = n
            for i,record in enumerate(target):
                target_score[ target_name[i] ] = snd.classify( record )
            
            for i,record in enumerate(nontarget):
                nontarget_score[ nontarget_name[i] ] = snd.classify( record )
            # evaluate score
            ts = 0
            for c in target_score.values():
                if c > 0:
                    ts += 1
            ns = 0
            for c in nontarget_score.values():
                if c <= 0:
                    ns += 1
            tscore = ts/len(target_score) *100
            nscore = ns/len(nontarget_score) *100
            print("target score:", tscore )
            print("nontarget score:", nscore )
            
            if tscore > max_score[0] and nscore > max_score[1]:
                max_score = (tscore,nscore)
                treshold = n
                print('found', treshold,':',tscore,nscore)
            

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
    """
    Fuses image score and sound score and makes hard decision.
    """
    soundSc = getSoundScore()
    imgSc = getImageScore()

    if len(soundSc) != len(imgSc):
        for s in soundSc.keys():
            if not s in imgSc.keys():
                print(str(s)+' only present in sound!')
        for s in imgSc.keys():
            if not s in soundSc.keys():
                print(str(s)+' only present in image!')

    assert len(soundSc) == len(imgSc), "Sound recognition number of files is different from image"

    result = {k: [v1, imgSc[k]] for k, v1 in soundSc.values()}
    for file, results in result.values():
        print("File: - {}\nSound {}\tImage{}".format(file, result[0], result[1]))


if __name__ == '__main__':
    # --train
    if len(sys.argv) == 2 and sys.argv[1] == '--train':
        train = True
        fusion()
    # no argument
    elif len(sys.argv) == 1:
        fusion()
    # bad arguments
    else:
        print('Usage: ./main [--train]', file=sys.stderr)
        exit()
