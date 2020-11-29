import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer

#-------------------------------------------------------------------------------- 

def main ():
    #runTest()
    runTrain()
def runTrain():
    
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    pathDirData = './database'
    pathFileTrain = './dataset/train_1.txt'
    pathFileVal = './dataset/val_1.txt'
    pathFileTest = './dataset/test_1.txt'

    nnArchitecture = DENSENET121
    nnIsTrained = True
    nnClassCount = 14

    trBatchSize = 16
    trMaxEpoch = 100

    imgtransResize = 256
    imgtransCrop = 224
    pathModel = 'm-' + timestampLaunch + '.pth.tar'
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)
# def runTest():
#
#     pathDirData = './database'
#     pathFileTest = './dataset/test_1.txt'
#     nnArchitecture = 'DENSE-NET-121'
#     nnIsTrained = True
#     nnClassCount = 14
#     trBatchSize = 16
#     imgtransResize = 256
#     imgtransCrop = 224
#     pathModel = './models/m-25012018-123527.pth.tar'
#     timestampLaunch = ''
#     ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)
if __name__ == '__main__':
    main()





