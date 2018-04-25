#!/usr/bin/env python3

import img_reader as img
import train_img as train
import snd_lib as lib
import numpy as np
import sys

TRAIN = True

def getSoundScore():

    # get data
    target = lib.getFeatures( lib.TARGET_DEV )
    #print(target.values())
    nontarget = lib.getFeatures( lib.NONTARGET_DEV )
    
    # this works just wierd
    #target,nontarget = lib.processFeatures(target,nontarget)

    lib.train( np.array(list(target.values())), np.array(list(nontarget.values())) )
    
    if TRAIN:
        # validate target
        target = lib.getFeatures( lib.TARGET_DEV )
        target_score = {}
        for sample in target.keys():
            target_score[sample] = lib.classify(target[sample])

        # validate nontarget
        nontarget = lib.getFeatures( lib.NONTARGET_DEV )
        nontarget_score = {}
        for sample in nontarget.keys():
            nontarget_score[sample] = lib.classify(nontarget[sample])

        # evaluate score
        ts = 0
        for c in target_score:
            if c > 0:
                ts += 1
        ns = 0
        for c in nontarget_score:
            if c <= 0:
                ns += 1

        print("target score:", ts/len(target_score) *100 )
        print("nontarget score:", ns/len(nontarget_score) *100 )

        print(target)
        


    else:
        pass
        # read real data

   

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