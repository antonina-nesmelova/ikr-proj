
import os
import ikrlib as ikr


TARGET_TRAIN = 'data' + os.sep + 'target_train'
NONTARGET_TRAIN = 'data' + os.sep + 'non_target_train'
TARGET_DEV = 'data' + os.sep + 'target_dev'
NONTARGET_DEV = 'data' + os.sep + 'non_target_dev'

def getTrain_TargetFeatures():
    """
    Loads extracted features from training set of target files.
    """
    features = ikr.wav16khz2mfcc(TARGET_TRAIN)
    result = []
    for feature in features.values():
        result.append(feature.tolist())
    for i in result:
        print(len(i))
    print(len(result))
    return result

def getTrain_NonTargetFeatures():
    raise NotImplementedError('Action not implemented yet!')

def getTest_TargetFeatures():
    raise NotImplementedError('Action not implemented yet!')

def getTest_NonTargetFeatures():
    raise NotImplementedError('Action not implemented yet!')