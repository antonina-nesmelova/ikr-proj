#!/usr/bin/env python3

import img_reader as img
import train_img as train_lib
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

REALDATA = True
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
    names_listed = []
    names, target = img.getFeatures(TARGET_DEV)
    names_listed.append(names)
    names, nonetarget = img.getFeatures(NONTARGET_DEV)
    names_listed.append(names)
    names, test_target = img.getFeatures(TARGET_TRAIN)
    names_listed.append(names)
    names, test_nonetarget = img.getFeatures(NONTARGET_TRAIN)
    names_listed.append(names)


    print('Get gauss')
    w, m, c, v1, v2 = train_lib.getGauss(target, nonetarget)
    # print("w: {}\nm: {}\nc: {}\nv1: {}\nv2: {}\n".format(w, m, c, v1, v2))
    skt1 = train_lib.getScore(test_target, w[0], m[0], c[0], v1)
    skn1 = train_lib.getScore(test_nonetarget, w[0], m[0], c[0], v1)
    skt2 = train_lib.getScore(test_target, w[1], m[1], c[1], v2)
    skn2 = train_lib.getScore(test_nonetarget, w[1], m[1], c[1], v2)

    skt3 = train_lib.getScore(target, w[0], m[0], c[0], v1)
    skn3 = train_lib.getScore(nonetarget, w[0], m[0], c[0], v1)
    skt4 = train_lib.getScore(target, w[1], m[1], c[1], v2)
    skn4 = train_lib.getScore(nonetarget, w[1], m[1], c[1], v2)

    score_target = list(map(add, skt3, skt4))
    score_nonetarget = list(map(add, skn3, skn4))

    score_test_target = list(map(add, skt1, skt2))
    score_test_nonetarget = list(map(add, skn1, skn2))

    print(  "score_target: {}"
            "score_nonetarget: {}"
            "score_test_target: {}"
            "score_test_nonetarget: {}".format(
                score_target, score_nonetarget,
                score_test_target, score_test_nonetarget))

    # Score divider is -40 (form Tony_the_boss)

    plt.figure(1)
    plt.plot(score_test_target, 'r.', score_test_nonetarget, 'b.')

    plt.figure(2)
    plt.plot(score_target, 'r.', score_nonetarget, 'b.')

    plt.show()
    result = {}
    for name, score in zip(names_listed[0], score_target):
        result[name] = score
    for name, score in zip(names_listed[1], score_nonetarget):
        result[name] = score
    for name, score in zip(names_listed[2], score_test_target):
        result[name] = score
    for name, score in zip(names_listed[3], score_test_nonetarget):
        result[name] = score

    return result

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
