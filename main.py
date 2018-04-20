#!/usr/bin/env python3

import snd_reader as loader
import snd_recognizer as recog
import snd_lib as lib

def getScore():
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


    



if __name__ == '__main__':
    getScore()