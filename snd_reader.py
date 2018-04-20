#!/usr/bin/env python3

import os
from glob import glob
from scipy.io import wavfile

TARGET_TRAIN = 'data' + os.sep + 'target_train'
NONTARGET_TRAIN = 'data' + os.sep + 'non_target_train'
TARGET_DEV = 'data' + os.sep + 'target_dev'
NONTARGET_DEV = 'data' + os.sep + 'non_target_dev'

def wav16khz2mfcc(dir_name):
    """
    Loads all *.wav files from directory dir_name (must be 16kHz), converts them into MFCC 
    features (13 coefficients) and stores them into a dictionary. Keys are the file names
    and values and 2D numpy arrays of MFCC features.
    """
    features = {}
    for f in glob(dir_name + '/*.wav'):
        print('Processing file: ', f)
        rate, s = wavfile.read(f)
        assert(rate == 16000)
        features[f] = mfcc(s, 400, 240, 512, 16000, 23, 13)
    return features

def getTrain_TargetFeatures():
    """
    Loads 
    """
    features = wav16khz2mfcc(TARGET_TRAIN)
    retu

def getTrain_NonTargetFeatures():
    raise NotImplementedError('Action not implemented yet!')

def getTest_TargetFeatures():
    raise NotImplementedError('Action not implemented yet!')

def getTest_NonTargetFeatures():
    raise NotImplementedError('Action not implemented yet!')