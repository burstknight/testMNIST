from myDataset import myDataset, myDatasetIterator
from Network import myNetwork
import numpy as np

def train():
    strTrainImagePath = "./dataset/train-images.idx3-ubyte"
    strTrainLabelPath = "./dataset/train-labels.idx1-ubyte"

    iMaxEpochs = 12000
    iBatchSize = 100
    fLearningRate = 0.01

    oDataset = myDataset(strTrainImagePath, strTrainLabelPath)
    oIterator = myDatasetIterator(oDataset, iBatchSize)
    oIterator.init()

    print("Initialize network...", end="")
    oNetwork = myNetwork()
    oNetwork.initWeight()
    print(" Done!")

    for i in range(iMaxEpochs):
        mBatchImage, mLabel = oIterator.getExample()
        mInput = mBatchImage.astype("float32")
        mInput /= 255.0

        # calculage gradient
        dctGradient = oNetwork.calcNumGradient(mInput, mLabel)

        # update network weights
        for strKey in oNetwork.dctWeights.keys():
            oNetwork.dctWeights[strKey] -= fLearningRate*dctGradient[strKey]
        # End of for-loop

        if(i % 100 == 0):
            print("Iterations: %8d, Loss: %.4f" %(i, oNetwork.calcLoss(mBatchImage, mLabel)))
        # End of if-conditon
    # End of for-loop
# End of train

def main():
    train()
    return
# End of main

if "__main__" == __name__:
    main()
# End of if-condition