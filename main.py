#!/usr/bin/env python3

import img_reader as img
import img_trainGMM as trainLib
import snd_trainGMM as snd
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

# True  - real data from data/eval/ are used
# False - test data from data/[non_]target_dev are used
REALDATA = False

# train, or classify
train = False
# show the diagrams
show = False

def CheckEmpty(cond, loc):
    if cond:
        print('No input data in src/data/' + loc)
        exit(1)

def main():
    global train
    global show
    # --train
    if len(sys.argv) == 2 and sys.argv[1] == '--train':
        train = True
        getSoundScore()
        getImageScore()
    elif len(sys.argv) == 2 and sys.argv[1] == "--show":
        show = False
        getSoundScore()
        getImageScore()
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

    global REALDATA, train

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
        CheckEmpty(len(target) == 0, 'target_train')
        nontarget,_ = snd.getFeatures( snd.NONTARGET_TRAIN )
        CheckEmpty(len(target) == 0, 'non_target_train')
        # train
        snd.train(mergeWithin(target), mergeWithin(nontarget))
        exit(0)
    # load classifier
    else:
        snd.load_trained()

    # read real data
    if REALDATA:
        loc = 'data'+os.sep+'eval'
        data,dataname = snd.getFeatures(loc)
        CheckEmpty(len(data) == 0, 'eval')

        score = {}
        for i,d in enumerate(data):
            score[ dataname[i] ] = float(snd.classify(d))
        #for k in score.keys():
        #    print(str(k)+' : '+str(score[k]))
        return score
    # cross validation
    else:
        # load
        target, target_name = snd.getFeatures( snd.TARGET_DEV )
        CheckEmpty(len(target) == 0, 'target_dev')
        nontarget, nontarget_name = snd.getFeatures( snd.NONTARGET_DEV )
        CheckEmpty(len(nontarget) == 0, 'non_target_dev')
        # validate (rewritten to get it in one dict)
        score = {}
        for i,record in enumerate(target):
            score[ target_name[i] ] = snd.classify( record )
        for i,record in enumerate(nontarget):
            score[ nontarget_name[i] ] = snd.classify( record )
        # evaluate score
        #ts = 0
        #for c in target_score.values():
        #    if c > 0:
        #        ts += 1
        #ns = 0
        #for c in nontarget_score.values():
        #    if c <= 0:
        #        ns += 1
        #tscore = ts/len(target_score) *100
        #nscore = ns/len(nontarget_score) *100
        #print("target score:", tscore )
        #print("nontarget score:", nscore )
        return score 

  

def getImageScore():
    """ Counts image score. """
    global t, REALDATA, train, show

    def showData(x1,x2,title, wf, mf, cf):
        """ Data show. """
        if show:
            plt.figure(title);
            plt.plot(x1[:,0], x1[:,1], 'r.', x2[:,0], x2[:,1], 'b.')
            for w, m, c in zip(wf, mf, cf):
                gellipse(m, c, 3000, 'r', lw=round(w * 10))


    # training
    if train:
        target, target_names = img.getFeatures(TARGET_TRAIN)
        nonetarget, nonetarget_names = img.getFeatures(NONTARGET_TRAIN)

        v1, v2, v3 = train.getVectors(target, nonetarget)

        # # save train data
        tdir = 'train'+os.sep
        if not os.path.exists(tdir):
            os.makedirs(tdir)
        np.save(tdir+'v1.npy',v1)
        np.save(tdir+'v2.npy',v2)
        np.save(tdir+'v3.npy',v3)
        w, m, c = train.getGauss(target, nonetarget, v1, v2, v3, test_target, test_nonetarget)
        np.save(tdir+'w1.npy',w[0])
        np.save(tdir+'m1.npy',m[0])
        np.save(tdir+'c1.npy',c[0])
        np.save(tdir+'w2.npy',w[1])
        np.save(tdir+'m2.npy',m[1])
        np.save(tdir+'c2.npy',c[1])
        np.save(tdir+'w3.npy',w[2])
        np.save(tdir+'m3.npy',m[2])
        np.save(tdir+'c3.npy',c[2])
        exit()


    # classify
    tdir = 'train'+os.sep
    if not os.path.exists(tdir):
        raise IOError('no train directory')
    # load
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

    # real data
    if REALDATA:
        # load
        loc = 'data'+os.sep+'eval'
        test, test_names = img.getFeatures(loc)
        CheckEmpty(len(test) == 0, 'eval')
        # score
        score1_test = trainLib.getScore(test, ww[0], mm[0], cc[0], v1)
        score2_test = trainLib.getScore(test, ww[1], mm[1], cc[1], v2)
        score3_test = trainLib.getScore(test, ww[2], mm[2], cc[2], v3)
        score = {}
        for i,key in enumerate(test_names):
            score[key] = score1_test[i] + score2_test[i] + score3_test[i] + 58
        return score
    
    # cross validation
    else:
        test_target, test_target_names = img.getFeatures(TARGET_DEV)
        test_nonetarget, test_nonetarget_names = img.getFeatures(NONTARGET_DEV)

        # transform data
        ttar = trainLib.transformData(test_target,v1, False)
        tntar = trainLib.transformData(test_nonetarget,v1, False)
        showData(np.vstack([ttar]), np.vstack([tntar]), 'First two vectors', ww[0], mm[0], cc[0])
        ttar = trainLib.transformData(test_target,v2, False)
        tntar = trainLib.transformData(test_nonetarget,v2, False)
        showData(np.vstack([ttar]), np.vstack([tntar]), 'Second two vectors', ww[1], mm[1], cc[1])
        ttar = trainLib.transformData(test_target,v3, False)
        tntar = trainLib.transformData(test_nonetarget,v3, False)
        showData(np.vstack([ttar]), np.vstack([tntar]), 'Third two vectors', ww[1], mm[1], cc[1])

        # count score
        score1_test_target = trainLib.getScore(test_target, ww[0], mm[0], cc[0], v1)
        score1_test_nonetarget = trainLib.getScore(test_nonetarget, ww[0], mm[0], cc[0], v1)
        score2_test_target = trainLib.getScore(test_target, ww[1], mm[1], cc[1], v2)
        score2_test_nonetarget = trainLib.getScore(test_nonetarget, ww[1], mm[1], cc[1], v2)
        score3_test_target = trainLib.getScore(test_target, ww[2], mm[2], cc[2], v3)
        score3_test_nonetarget = trainLib.getScore(test_nonetarget, ww[2], mm[2], cc[2], v3)

        # merge score
        score_test_target = list(map(add, score1_test_target, score2_test_target))
        score_test_target = list(map(add, score_test_target, score3_test_target))
        score_test_nonetarget = list(map(add, score1_test_nonetarget, score2_test_nonetarget))
        score_test_nonetarget = list(map(add, score_test_nonetarget, score3_test_nonetarget))
        score_test_target       = [v + 58 for v in score_test_target]
        score_test_nonetarget   = [v + 58 for v in score_test_nonetarget]
        dic = dict(zip(test_target_names, score_test_target))
        dic = {**dic ,**dict(zip(test_nonetarget_names, score_test_nonetarget))}
        return dic

def fusion():
    """
    Fuses image score and sound score and makes hard decision.
    """

    imgSc = getImageScore()
    with open("image-results.txt", "w") as fus_file:
        for file in sorted(imgSc.keys()):
            fus_file.write("{name} {res_sum} {fus_res}\n".format(
                    name=file, res_sum=float(imgSc[file]), fus_res=int(float(imgSc[file]) > 0)))

    soundSc = getSoundScore()
    with open("sound-results.txt", "w") as fus_file:
        for file in sorted(soundSc.keys()):
            fus_file.write("{name} {res_sum} {fus_res}\n".format(
                    name=file, res_sum=float(soundSc[file]), fus_res=int(float(soundSc[file]) > 0)))

    assert(len(imgSc) == len(soundSc))
    res = {}
    for k in imgSc.keys():
        res[k] = imgSc[k] + soundSc[k]
    
    
    with open("results.txt", "w") as fus_file:
        for file in res.keys():
            fus_file.write("{name} {res_sum} {fus_res}\n".format(
                    name=file, res_sum=float(res[file]), fus_res=int(float(res[file]) > 0)))
    exit()

    # fusion
    border = 10
    mean1 = sum(map(abs, soundSc.values())) / len(soundSc)
    mean2 = sum(map(abs, imgSc.values()))   / len(imgSc)
    norm = mean1 / mean2
    result = {k: [v1*norm, imgSc[k]] for k, v1 in soundSc.items()}
    with open("results.txt", "w") as fus_file:
        for file, results in result.items():
            # Calculation
            res_sum = results[0] + results[1]
            # output
            print("File: - {}\nSound {}\tImage {}\tScore {}".format(file, soundSc[file], imgSc[file], res_sum))
            fus_file.write("{name} {res_sum} {fus_res}\n".format(
                name=file, res_sum=res_sum, fus_res=int(res_sum > 0)))


if __name__ == '__main__':
    main()
