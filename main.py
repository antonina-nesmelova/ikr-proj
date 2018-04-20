#!/usr/bin/env python3

import snd_reader as loader
import snd_recognizer as recog
import snd_lib as lib
import sys
import snd_reader as sr
import img_reader as img

def getSoundScore():
    # get data
    target = loader.getTrain_TargetFeatures()
    nontarget = loader.getTrain_NonTargetFeatures()
    lib.plotFeatures(target,nontarget)

    # train
    #recog.train(target,nontarget)

    # validate target
    #target = loader.getTest_TargetFeatures()
    #target_score = [recog.classify(sample) for sample in target]

    # validate nontarget
    #nontarget = loader.getTest_NonTargetFeatures()
    #nontarget_score = [recog.classify(sample) for sample in nontarget]


   

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
    elif:
	print('Usage: ./main [--image | --sound]', file=sys.stderr)
	exit()