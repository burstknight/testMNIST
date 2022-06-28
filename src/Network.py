from collections import OrderedDict
from typing import Dict
import numpy as np
import pickle
from os.path import isfile

def sigmoid(x):
    return 1/(1 + np.exp(-x))
# End of sigmoid

def relu(x):
    return np.maximum(0, x)
# End of relu

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y
# End of softmax

def crossEntropyError(mY:np.ndarray, mT:np.ndarray):
    if(mY.ndim == 1):
        mT = mT.reshape(1, mT.size)
        mY = mY.reshape(1, mY.size)
    # Endo f if-condition

    iBatchSize = mY.shape[0]
    return -np.sum(mT*np.log(mY))/iBatchSize
# End of crossEntropyError

def calcNumGradient(func, mX:np.ndarray):
    fH = 1e-6
    vShape = mX.shape
    mX = mX.reshape((mX.size, 1))
    mGradient = np.zeros_like(mX)
    for i in range(mX.size):
        fTmpVal = mX[i]

        # calculate f(x + h)
        mX[i] = fTmpVal + fH
        fDeltaY1 = func(mX)

        # calculate f(x - h)
        mX[i] = fTmpVal - fH
        fDeltaY2 = func(mX)

        mGradient[i] = (fDeltaY1 - fDeltaY2)/(2*fH)
        mX[i] = fTmpVal
    # End of for-loop

    mX = mX.reshape(vShape)
    mGradient = mGradient.reshape(vShape)
    
    return mGradient
# End of calcNumGradient

class myAffine:
    """
    Description:
    =====================================
    This class can perform forward and backward.
    """
    def __init__(self, mWeights:np.ndarray, mBias:np.ndarray) -> None:
        self.__m_mWeights = mWeights
        self.__m_mBias = mBias
        self.__m_mX = None
        self.__m_mDWeights = None
        self.__m_mDBias = None
    # End of constructor

    @property
    def mDeltaWeights(self):
        return self.__m_mDWeights
    # End of myAffine::mDeltaWeights

    @property
    def mDeltaBias(self):
        return self.__m_mDBias
    # End of myAffine::mDeltaBias

    def forward(self, mX:np.ndarray):
        """
        Description:
        ========================================
        Perform forward.

        Args:
        ========================================
        - mX:   ptype: np.ndarray, the input data

        Returns:
        ========================================
        - rtype: np.ndarray, the result
        """
        self.__m_mX = mX
        mOut = np.dot(mX, self.__m_mWeights) + self.__m_mBias

        return mOut
    # End of myAffine::forward

    def backward(self, mDOut:np.ndarray):
        """
        Description:
        ======================================
        Use backward to calculate the current gradient.

        Args:
        ======================================
        - mDOut:    ptype: np.ndarray, the result from forward.

        Returns:
        ======================================
        - rtype: np.ndarray, the gradient of the input.
        """
        mDX = np.dot(mDOut, self.__m_mWeights.T)
        self.__m_mDWeights = np.dot(self.__m_mX.T, mDOut)
        self.__m_mDBias = np.sum(mDOut, axis=0)

        return mDX
    # End of myAffine::backward
# End of class myAffine

class Relu:
    """
    Description:
    ================================
    This class can perform Relu operation
    """
    def __init__(self) -> None:
        self.__m_mMask = None
    # End of constructor

    def forward(self, mX:np.ndarray):
        """
        Description:
        =====================================
        Perform forward.

        Args:
        =====================================
        - mX:   ptype: np.ndarray, the input data

        Returns:
        =====================================
        - rtype: np.ndarray, the result of forward
        """
        self.__m_mMask = (mX <= 0)
        mOut = mX.copy()
        mOut[self.__m_mMask] = 0
        return mOut
    # End of Relu::forward

    def backward(self, mDeltaOut:np.ndarray):
        """
        Description:
        =====================================
        Perform backward.

        Args:
        =====================================
        - mDeltaOut:    ptype: np.ndarray, the delta output.

        Args:
        =====================================
        - rtype: np.ndarray, the calculated delta input.
        """
        mDeltaOut[self.__m_mMask] = 0
        mDeltaX = mDeltaOut

        return mDeltaX
    # End of Relu::backward
# End of class Relu

class mySoftmaxLoss:
    def __init__(self) -> None:
        self.__m_fLoss = None
        self.__m_mY = None
        self.__m_mLabel = None
    # End of constructor

    def forward(self, mX:np.ndarray, mT:np.ndarray):
        self.__m_mLabel = mT
        self.__m_mY = softmax(mX)
        self.__m_fLoss = crossEntropyError(self.__m_mY, self.__m_mLabel)

        return self.__m_fLoss
    # End of mySoftmaxLoss::forward

    def backward(self, dout=1):
        iBatchSize = self.__m_mLabel.shape[0]
        mDeltaX = (self.__m_mY - self.__m_mLabel)/iBatchSize

        return mDeltaX
    # End of mySoftmaxLoss::backward
# End of class mySoftmaxLoss

class myNetwork:
    """
    Description:
    =========================================================
    This class is a neural network.
    """
    def __init__(self) -> None:
        self.__m_dctNetwork = {}
        self.__m_dctNetwork["w1"] = np.zeros((784, 50), dtype="float32")
        self.__m_dctNetwork["w2"] = np.zeros((50, 10), dtype="float32")
        self.__m_dctNetwork["b1"] = np.zeros((50, ), dtype="float32")
        self.__m_dctNetwork["b2"] = np.zeros(((10, )), dtype="float32")

        self.__m_dctLayers = OrderedDict()
        self.__m_dctLayers["affine1"] = myAffine(self.dctWeights["w1"], self.dctWeights["b1"])
        self.__m_dctLayers["relu1"] = Relu()
        self.__m_dctLayers["affine2"] = myAffine(self.dctWeights["w2"], self.dctWeights["b2"])

        self.__m_oLastLayer = mySoftmaxLoss()
    # End of constructor

    @property
    def dctWeights(self) -> Dict[str, np.ndarray]:
        return self.__m_dctNetwork
    # End of myNetwork::dctWeights

    def save(self, strPath:str):
        """
        Description:
        ========================================================
        Save network weights.

        Args:
        ========================================================
        - strPath:  ptype: str, the path to store network weights.

        Returns:
        ========================================================
        - rtype: void
        """
        with open(strPath, "wb") as oWriter:
            pickle.dump(self.dctWeights, oWriter)
        # End of with-block
    # End of myNetwork::save

    def load(self, strPath:str):
        """
        Description:
        =========================================================
        Load network weights.

        Args:
        =========================================================
        - strPath:  ptype: str, the weights file path

        Returns:
        =========================================================
        - rtype: bool, return True if succeed to load weights, otherwise return False
        """
        if(False == isfile(strPath)):
            return False
        # End of if-condition

        with open(strPath, "rb") as oReader:
            self.dctWeights = pickle.load(oReader)
        # End of with-block

        return True
    #  End of myNetwork::load

    def initWeight(self):
        """
        Description:
        ==================================================
        Initialize the wieghts for training.
        """
        for strKey in self.__m_dctNetwork.keys():
            if("b" in strKey):
                continue
            # End of if-condition

            vPreShape = self.__m_dctNetwork[strKey].shape
            self.__m_dctNetwork[strKey] = self.__m_dctNetwork[strKey].reshape((self.__m_dctNetwork[strKey].size, 1))
            for i in range(self.__m_dctNetwork[strKey].size):
                self.__m_dctNetwork[strKey][i] = np.random.uniform()*0.01
            # End of for-loop
            self.__m_dctNetwork[strKey] = self.__m_dctNetwork[strKey].reshape(vPreShape)
        # End of for-loop
    # End of myNetwork::initWeight

    def predict(self, mX:np.ndarray) -> np.ndarray:
        """
        Description:
        ======================================================
        Predict the result.

        Args:
        ======================================================
        - mX:   ptype: np.ndarray, (batch_size, 784), the input data

        Returns:
        ======================================================
        - rtype: np.ndarray, (batch_size, 10), the predicted result
        """
        for oLayer in self.__m_dctLayers.values():
            mX = oLayer.forward(mX)
        # End of for-loop

        return mX
    # End of myNetwork::predict

    def calcLoss(self, mX:np.ndarray, mT:np.ndarray) -> float:
        """
        Description:
        ==============================================
        Calculate the loss for the given input data.

        Args:
        ==============================================
        - mX:   ptype:  np.ndarray, (batch_size, 784), the input data
        - mT:   ptype:  np.ndarray, (batchsize, ), the label of the input data

        Returns:
        ==============================================
        - rtype: float, the current loss
        """
        mY = self.predict(mX)

        return self.__m_oLastLayer.forward(mY, mT)
    # End of myNetwork::calcLoss

    def calcNumGradient(self, mX:np.ndarray, mT:np.ndarray) -> Dict[str, np.ndarray]:
        """
        Description:
        ===================================================================
        Calculate the current gradient of the network weights with numerical
        method.

        Args:
        ===================================================================
        - mX:   ptype:  np.ndarray, (batch_size, 784), the input data
        - mT:   ptype:  np.ndarray, (batchsize, ), the label of the input data

        Returns:
        ==============================================
        - rtype: dict, the current gradient
        """
        loss_W = lambda W: self.calcLoss(mX, mT)

        dctGradient = {}
        dctGradient["w1"] = calcNumGradient(loss_W, self.__m_dctNetwork["w1"])
        dctGradient["b1"] = calcNumGradient(loss_W, self.__m_dctNetwork["b1"])
        dctGradient["w2"] = calcNumGradient(loss_W, self.__m_dctNetwork["w2"])
        dctGradient["b2"] = calcNumGradient(loss_W, self.__m_dctNetwork["b2"])

        return dctGradient
    # End of myNetwork::calcNumGradient

    def calcGradient(self, mX:np.ndarray, mT:np.ndarray):
        """
        Description:
        ==========================================
        Use backford to calculate gradient.

        Args:
        ==========================================
        - mX:   ptype:  np.ndarray, (batch_size, 784), the input data
        - mT:   ptype:  np.ndarray, (batchsize, ), the label of the input data

        Returns:
        ==============================================
        - rtype: dict, the current gradient
        """
        # forward
        self.calcLoss(mX, mT)

        # backward
        dout = 1
        dout = self.__m_oLastLayer.backward(dout)

        voLayers = list(self.__m_dctLayers.values())
        voLayers.reverse()
        for oLayer in voLayers:
            dout = oLayer.backward(dout)
        # End of for-loop

        dctGradient = {}
        dctGradient["w1"] = self.__m_dctLayers["affine1"].mDeltaWeights
        dctGradient["b1"] = self.__m_dctLayers["affine1"].mDeltaBias
        dctGradient["w2"] = self.__m_dctLayers["affine2"].mDeltaWeights
        dctGradient["b2"] = self.__m_dctLayers["affine2"].mDeltaBias

        return dctGradient
    # End of myNetwork::calcGradient
# End of class myNetwork