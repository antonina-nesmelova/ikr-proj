#!/usr/bin/env python3

import sys
import img_reader as img
import snd_lib as lib
import ikrlib as ikr

TRAIN = True

def getSoundScore():

    # get data
    target = lib.getFeatures( lib.TARGET_TRAIN )
    nontarget = lib.getFeatures( lib.NONTARGET_TRAIN )
    
    target,nontarget = lib.processFeatures(target,nontarget)

    # train
    # not separated enough
    #lib.plotFeatures(target,nontarget,2,1)
    #lib.plotFeatures(target,nontarget,3,1)
    #lib.plotFeatures(target,nontarget,3,2)
    #lib.plotFeatures(target,nontarget,4,1)
    #lib.plotFeatures(target,nontarget,4,2)
    #lib.plotFeatures(target,nontarget,4,3)
    #lib.plotFeatures(target,nontarget,5,1)
    #lib.plotFeatures(target,nontarget,5,2)
    #lib.plotFeatures(target,nontarget,5,3)
    #lib.plotFeatures(target,nontarget,5,4)

    lib.train(target, nontarget)
    
    if TRAIN:
        # validate target
        target = lib.getFeatures( lib.TARGET_DEV )
        target_score = [lib.classify(sample) for sample in target]

        # validate nontarget
        nontarget = lib.getFeatures( lib.NONTARGET_DEV )
        nontarget_score = [lib.classify(sample) for sample in nontarget]

        # evaluate score
        print(target_score[0:50])
        print(nontarget_score[0:50])
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