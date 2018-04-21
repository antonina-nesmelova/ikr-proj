#!/usr/bin/env python3

import sys
import img_reader as img
import snd_lib as lib

TRAIN = True

def getSoundScore():

    # get data
    target = lib.getFeatures( lib.TARGET_DEV )
    nontarget = lib.getFeatures( lib.NONTARGET_DEV )
    #lib.plotFeatures(target,nontarget)

    target,nontarget = lib.processFeatures(target,nontarget)

    # train
    lib.train(target, nontarget)

    if TRAIN:
        pass
        # validate target
        #target = lib.getFeatures( lib.TARGET_DEV )
        #target_score = [recog.classify(sample) for sample in target]

        # validate nontarget
        #nontarget = lib.getFeatures( lib.NONTARGET_DEV )
        #nontarget_score = [recog.classify(sample) for sample in nontarget]

        # evaluate score
    else:
        pass
        # read real data

   

def getImageScore():
	img.getVector()

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