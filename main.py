#!/usr/bin/env python3

import snd_reader as sr
import img_reader as img
import train_img as train
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from ikrlib import gellipse

from operator import add

TARGET_DEV = 'data/target_dev/'
NONTARGET_DEV = 'data/non_target_dev'
TARGET_TRAIN = 'data/target_train'
NONTARGET_TRAIN = 'data/non_target_train'

REALDATA = True
t = True

def makeDictionary(keys, values):
    return dict(zip(keys, values))

def main():
    global t
    # --train
    if len(sys.argv) == 2 and sys.argv[1] == '--train':
        t = True
        fusion()
    # no argument
    elif len(sys.argv) == 1:
        t = False
        fusion()
    # bad arguments
    else:
        #print('Usage: ./main [--train]', file=sys.stderr)
        exit()

def fusion():
    global t
    test_target, test_target_names = img.getFeatures(TARGET_DEV)
    test_nonetarget, test_nonetarget_names = img.getFeatures(NONTARGET_DEV)
    target, target_names = img.getFeatures(TARGET_TRAIN)
    nonetarget, nonetarget_names = img.getFeatures(NONTARGET_TRAIN)
    if (t):
        
        #v1, v2, v3 = train.getVectors(target, nonetarget)

        # save train data
        tdir = 'train'+os.sep
        if not os.path.exists(tdir):
            os.makedirs(tdir)

        v1 = np.load(tdir+'v1.npy')
        v2 = np.load(tdir+'v2.npy')
        v3 = np.load(tdir+'v3.npy')

        #np.save(tdir+'v1.npy',v1)
        #np.save(tdir+'v2.npy',v2)
        #np.save(tdir+'v3.npy',v3)
        w, m, c = train.getGauss(target, nonetarget, v1, v2, v3, test_target, test_nonetarget)
        print('W')
        print(w)
        print('M')
        print(m)
        print('C')
        print(c)
        exit()
        #np.save(tdir+'w1.npy',w[0])
        #np.save(tdir+'m1.npy',m[0])
        #np.save(tdir+'c1.npy',c[0])

        #np.save(tdir+'w2.npy',w[1])
        #np.save(tdir+'m2.npy',m[1])
        #np.save(tdir+'c2.npy',c[1])

        #np.save(tdir+'w3.npy',w[2])
        #np.save(tdir+'m3.npy',m[2])
        #np.save(tdir+'c3.npy',c[2])

    else:
        tdir = 'train'+os.sep
        if not os.path.exists(tdir):
            raise IOError('no train directory')
        # load train data
        v1 = np.load(tdir+'v1.npy')
        v2 = np.load(tdir+'v2.npy')
        v3 = np.load(tdir+'v3.npy')


        w1 = np.load(tdir+'w1.npy')
        m1 = np.load(tdir+'m1.npy')
        c1 = np.load(tdir+'c1.npy')

        w2 = np.load(tdir+'w2.npy')
        m2 = np.load(tdir+'m2.npy')
        c2 = np.load(tdir+'c2.npy')

        w3 = np.load(tdir+'w3.npy')
        m3 = np.load(tdir+'m3.npy')
        c3 = np.load(tdir+'c3.npy')

        ww=[w1,w2,w3]
        mm=[m1,m2,m3]
        cc=[c1,c2,c3]

        print(ww)
        wf1=ww[0]
        mf1=mm[0]
        cf1=cc[0]


        tar = train.transformData(target,v1, False)
        ntar = train.transformData(nonetarget,v1, False)
        ttar = train.transformData(test_target,v1, False)
        tntar = train.transformData(test_nonetarget,v1, False)
        tar = np.vstack([tar, ttar])
        ntar = np.vstack([ntar, tntar])

        plt.figure('First two vectors');
        plt.plot(tar[:,0], tar[:,1], 'r.', ntar[:,0], ntar[:,1], 'b.')
        for w, m, c in zip(wf1, mf1, cf1): gellipse(m, c, 3000, 'r', lw=round(w * 10))

        tar = train.transformData(target,v2, False)
        ntar = train.transformData(nonetarget,v2, False)
        ttar = train.transformData(test_target,v2, False)
        tntar = train.transformData(test_nonetarget,v2, False)
        tar = np.vstack([tar, ttar])
        ntar = np.vstack([ntar, tntar])

        wf2=ww[1]
        mf2=mm[1]
        cf2=cc[1]

        plt.figure('Second two vectors');
        plt.plot(tar[:,0], tar[:,1], 'r.', ntar[:,0], ntar[:,1], 'b.')
        for w, m, c in zip(wf2, mf2, cf2): gellipse(m, c, 3000, 'r', lw=round(w * 10))

        tar = train.transformData(target,v3, False)
        ntar = train.transformData(nonetarget,v3, False)
        ttar = train.transformData(test_target,v3, False)
        tntar = train.transformData(test_nonetarget,v3, False)
        tar = np.vstack([tar, ttar])
        ntar = np.vstack([ntar, tntar])

        wf3=ww[2]
        mf3=mm[2]
        cf3=cc[2]

        plt.figure('Third two vectors');
        plt.plot(tar[:,0], tar[:,1], 'r.', ntar[:,0], ntar[:,1], 'b.')
        for w, m, c in zip(wf3, mf3, cf3): gellipse(m, c, 3000, 'r', lw=round(w * 10))
        plt.show()
    

    print('Get gauss')
    #print(v1)
    #print(v2)
    #print(v3)
    
    #print(w)
    #print(m)
    #print(c)
    v = []
    v = np.append(v, v1)
    v = np.append(v, v2)
    v = np.append(v, v3)
    #print(Vectors)
    #print(v)
    score1_test_target = train.getScore(test_target, wf1, mf1, cf1, v1)
    score1_test_nonetarget = train.getScore(test_nonetarget, wf1, mf1, cf1, v1)

    score2_test_target = train.getScore(test_target, wf2, mf2, cf2, v2)
    score2_test_nonetarget = train.getScore(test_nonetarget, wf2, mf2, cf2, v2)

    score3_test_target = train.getScore(test_target, wf3, mf3, cf3, v3)
    score3_test_nonetarget = train.getScore(test_nonetarget, wf3, mf3, cf3, v3)

    score1_target = train.getScore(target, wf1, mf1, cf1, v1)
    score1_nonetarget = train.getScore(nonetarget, wf1, mf1, cf1, v1)

    score2_target = train.getScore(target, wf2, mf2, cf2, v2)
    score2_nonetarget = train.getScore(nonetarget, wf2, mf2, cf2, v2)

    score3_target = train.getScore(target, wf3, mf3, cf3, v3)
    score3_nonetarget = train.getScore(nonetarget, wf3, mf3, cf3, v3)

    score_test_target = list(map(add, score1_test_target, score2_test_target))
    score_test_target = list(map(add, score_test_target, score3_test_target))
    score_test_nonetarget = list(map(add, score1_test_nonetarget, score2_test_nonetarget))
    score_test_nonetarget = list(map(add, score_test_nonetarget, score3_test_nonetarget))

    score_target = list(map(add, score1_target, score2_target))
    score_target = list(map(add, score_target, score3_target))
    score_nonetarget = list(map(add, score1_nonetarget, score2_nonetarget))
    score_nonetarget = list(map(add, score_nonetarget, score3_nonetarget))

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
        if t > -51:
            target_ok = target_ok + 1

    for t in score_target:
        if t > -51:
            target_ok = target_ok + 1

    for t in score_test_nonetarget:
        if t <= -51:
            nonetarget_ok = nonetarget_ok + 1 

    for t in score_nonetarget:
        if t <= -51:
            nonetarget_ok = nonetarget_ok + 1

    print("TargetOk: {}/{}={}".format(target_ok, len(score_test_target + score_target), float(target_ok)/len(score_test_target + score_target)))
    print("NontargetOk: {}/{}={}".format(nonetarget_ok, len(score_nonetarget + score_test_nonetarget), float(nonetarget_ok)/len(score_nonetarget + score_test_nonetarget)))

    plt.figure(1)
    plt.plot(score_test_target, 'r.', score_test_nonetarget, 'b.')

    plt.figure(2)
    plt.plot(score_target, 'r.', score_nonetarget, 'b.')

    plt.show()

    dic = makeDictionary(test_target_names, score_test_target)
    print(dic)


if __name__ == '__main__':
    main()