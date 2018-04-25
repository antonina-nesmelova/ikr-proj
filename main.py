#!/usr/bin/env python3

import img_reader as img
import train_img as train
import snd_lib as lib
import numpy as np
import sys
import os

TRAIN = False

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

    lib.train(mergeWithin(target), mergeWithin(nontarget))

    # train
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


    # read real data
    else:
        loc = 'data'+os.sep+'test'
        data,dataname = lib.getFeatures(loc)
        
        score = {}
        for i,d in enumerate(data):
            score[ dataname[i] ] = lib.classify(d)
        print(score)

        return score

        

   

def getImageScore():
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