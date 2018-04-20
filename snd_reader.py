
"""
This module is used to extract features from source data.
It contains their getters.
"""

import snd_lib as SndLib



def getTrain_TargetFeatures():
    """
    Loads extracted features from training set of target files.
    """
    return SndLib.getFeatures( SndLib.TARGET_TRAIN )

def getTrain_NonTargetFeatures():
    """
    Loads extracted features from training set of non target files.
    """
    return SndLib.getFeatures( SndLib.NONTARGET_TRAIN )

def getTest_TargetFeatures():
    """
    Loads extracted features from testing set of target files.
    """
    return SndLib.getFeatures( SndLib.TARGET_DEV )

def getTest_NonTargetFeatures():
    """
    Loads extracted features from testing set of non target files.
    """
    return SndLib.getFeatures( SndLib.NONTARGET_DEV )