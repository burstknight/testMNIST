from myDataset import myDataset, myDatasetIterator
from Network import myNetwork
import numpy as np

def train():
    strTrainImagePath = "./dataset/train-images.idx3-ubyte"
    strTrainLabelPath = "./dataset/train-labels.idx1-ubyte"

    iMaxEpochs = 12000
    iBatchSize = 100
    fLearningRate = 0.0002
    fGamma = 0.1
    fMomentom = 0.9

    viSteps = (8400, 10800)

    oDataset = myDataset(strTrainImagePath, strTrainLabelPath)
    oIterator = myDatasetIterator(oDataset, iBatchSize)
    oIterator.init()

    print("Initialize network...", end="")
    oNetwork = myNetwork()
    oNetwork.initWeight()
    print(" Done!")

    dctMomentum = {}
    for strKey in oNetwork.dctWeights.keys():
        dctMomentum[strKey] = np.zeros_like(oNetwork.dctWeights[strKey], dtype="float32")
    # End of for-loop

    for i in range(iMaxEpochs):
        mBatchImage, mLabel = oIterator.getExample()
        mInput = mBatchImage.astype("float32")
        mInput /= 255.0
        mInput -= 0.5
        mLabel = mLabel.astype("float32")

        # calculage gradient
        dctGradient = oNetwork.calcGradient(mInput.copy(), mLabel)

        # update network weights
        for strKey in oNetwork.dctWeights.keys():
            dctMomentum[strKey] = fMomentom*dctMomentum[strKey] - fLearningRate*dctGradient[strKey]
            oNetwork.dctWeights[strKey] += dctMomentum[strKey]
        # End of for-loop


        if(i % 100 == 0):
            print("Iterations: %8d, Learning rate: %.6f, Loss: %.6f, Recall: %.3f" \
                %(i, fLearningRate, oNetwork.calcLoss(mInput.copy(), mLabel), 
                oNetwork.calcRecall(mInput.copy(), mLabel)))
        # End of if-conditon

        if(i in viSteps):
            fLearningRate *= fGamma
        # End of if-condition

        oIterator.moveNext()
    # End of for-loop

    print("Final Loss: %.6f" %(oNetwork.calcLoss(mInput.copy(), mLabel)))

    oNetwork.save("model/model.pickle")
    print("Finish!")
# End of train

def main():
    train()
    return
# End of main

if "__main__" == __name__:
    main()
# End of if-condition