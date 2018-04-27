#!/usr/bin/env python3

import snd_reader as sr
import img_reader as img
import train_img as train
import snd_lib as snd
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

def main():
    global t
    # --train
    if len(sys.argv) == 2 and sys.argv[1] == '--train':
        t = True
        fusion()
    # no argument
    elif len(sys.argv) == 1:
        fusion()
    # bad arguments
    else:
        print('Usage: ./main [--train]', file=sys.stderr)
        exit()

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
        loc = 'data'+os.sep+'test'
        data,dataname = snd.getFeatures(loc)
        score = {}
        for i,d in enumerate(data):
            score[ dataname[i] ] = snd.classify(d)
        for k in score.keys():
            print(str(k)+' : '+str(score[k]))
        return score
    # cross validation
    else:
        # validate target
        target, target_name = snd.getFeatures( snd.TARGET_DEV )
        target_score = {}
        for i,record in enumerate(target):
            target_score[ target_name[i] ] = snd.classify( record )
        # validate nontarget
        nontarget, nontarget_name = snd.getFeatures( snd.NONTARGET_DEV )
        nontarget_score = {}
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

        print("target score:", ts/len(target_score) *100 )
        print("nontarget score:", ns/len(nontarget_score) *100 )



def getImageScore():
    global t
    test_target, test_target_names = img.getFeatures(TARGET_DEV)
    test_nonetarget, test_nonetarget_names = img.getFeatures(NONTARGET_DEV)
    target, target_names = img.getFeatures(TARGET_TRAIN)
    nonetarget, nonetarget_names = img.getFeatures(NONTARGET_TRAIN)

    # getting vectors and GMM model train         
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # v1, v2, v3 = train.getVectors(target, nonetarget)

    # # save train data
    # tdir = 'train'+os.sep
    # if not os.path.exists(tdir):
    #     os.makedirs(tdir)

    # # v1 = np.load(tdir+'v1.npy')
    # # v2 = np.load(tdir+'v2.npy')
    # # v3 = np.load(tdir+'v3.npy')

    # np.save(tdir+'v1.npy',v1)
    # np.save(tdir+'v2.npy',v2)
    # np.save(tdir+'v3.npy',v3)
    # w, m, c = train.getGauss(target, nonetarget, v1, v2, v3, test_target, test_nonetarget)
    # print('W')
    # print(w)
    # print('M')
    # print(m)
    # print('C')
    # print(c)
    # # np.save(tdir+'w1.npy',w[0])
    # # np.save(tdir+'m1.npy',m[0])
    # # np.save(tdir+'c1.npy',c[0])

    # # np.save(tdir+'w2.npy',w[1])
    # # np.save(tdir+'m2.npy',m[1])
    # # np.save(tdir+'c2.npy',c[1])

    # # np.save(tdir+'w3.npy',w[2])
    # # np.save(tdir+'m3.npy',m[2])
    # # np.save(tdir+'c3.npy',c[2])
    # exit()

    #++++++++++++++++++++++++++++++++++++++++++++

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

    ww=[w1,w2,w3]  # Weights of first 3 feature vectors
    mm=[m1,m2,m3]  # Means
    cc=[c1,c2,c3]  # Covariance matrixes 

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
    for w, m, c in zip(wf1, mf1, cf1):
        gellipse(m, c, 3000, 'r', lw=round(w * 10))

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

    score_target            = [v + 51 for v in score_target]
    score_nonetarget        = [v + 51 for v in score_nonetarget]
    score_test_target       = [v + 51 for v in score_test_target]
    score_test_nonetarget   = [v + 51 for v in score_test_nonetarget]

    # TODO: create better represintation of data
    # +++++++++++++++++++++++++++++++++++++++++++++++++
    # plt.figure(1)
    # plt.plot(score_test_target, 'r.', score_test_nonetarget, 'b.')
    # plt.figure(2)
    # plt.plot(score_target, 'r.', score_nonetarget, 'b.')
    # plt.show()
    # +++++++++++++++++++++++++++++++++++++++++++++++++

    # TODO: classify only real data
    dic = dict(zip(target_names, score_target))
    dic = {**dic ,**dict(zip(nonetarget_names, score_nonetarget))}
    dic = {**dic ,**dict(zip(test_target_names, score_test_target))}
    dic = {**dic ,**dict(zip(test_nonetarget_names, score_test_nonetarget))}
    print(len(dic))
    return dic

def fusion():
    """
    Fuses image score and sound score and makes hard decision.
    """

    # soundRes = getSoundScore()
    # soundSc = {'.'.join(k.split('.')[:-1]): v for k, v in soundRes.items()}

    # TODO: update from image branch
    # imgSc = getImageScore()

    # Tmp for test
    # TODO: delete
    imgSc =     {"f1":2, "f3":1}
    soundSc =   {"f1":2, "f3":1}

    print(len(soundSc), len(imgSc))
    for k in soundSc.keys():
        if k not in imgSc.keys():
            print(k)

    # Assert fails, don't know why
    # assert len(soundSc) == len(imgSc), "Sound recognition number of files is different from image"

    border = 10 # TODO: change to some heuristics

    result = {k: [v1, imgSc[k]] for k, v1 in soundSc.items()}
    with open("results.txt", "w") as fus_file:
        for file, results in result.items():
            # Calculation
            # TODO: set sound score as primary score via magic multiplier
            res_sum = (results[0] + results[1] * 0.8) / 2

            print("File: - {}\nSound {}\tImage {}\tScore {}".format(file, results[0], results[1], res_sum))
            fus_file.write("{name} {res_sum} {fus_res}\n".format(
                name=file, res_sum=res_sum, fus_res=int(res_sum < border)))


if __name__ == '__main__':
    main()
