from typing import Dict
import numpy as np

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
    return -np.sum(np.log(mY[np.arange(iBatchSize), mT]))/iBatchSize
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

class myNetwork:
    """
    Description:
    =========================================================
    This class is a neural network.
    """
    def __init__(self) -> None:
        self.__m_dctNetwork = {}
        self.__m_dctNetwork["w1"] = np.zeros((784, 50), dtype="float32")
        self.__m_dctNetwork["w2"] = np.zeros((50, 100), dtype="float32")
        self.__m_dctNetwork["w3"] = np.zeros((100, 10), dtype="float32")
        self.__m_dctNetwork["b1"] = np.zeros((50, ), dtype="float32")
        self.__m_dctNetwork["b2"] = np.zeros(((100, )), dtype="float32")
        self.__m_dctNetwork["b3"] = np.zeros((10, ), dtype="float32")
    # End of constructor

    @property
    def dctWeights(self) -> Dict[str, np.ndarray]:
        return self.__m_dctNetwork
    # End of myNetwork::dctWeights

    def initWeight(self):
        """
        Description:
        ==================================================
        Initialize the wieghts for training.
        """
        for strKey in self.__m_dctNetwork.keys():
            vPreShape = self.__m_dctNetwork[strKey].shape
            self.__m_dctNetwork[strKey] = self.__m_dctNetwork[strKey].reshape((self.__m_dctNetwork[strKey].size, 1))
            for i in range(self.__m_dctNetwork[strKey].size):
                self.__m_dctNetwork[strKey][i] = np.random.uniform(-1.0, 1.0)
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
        mW1 = self.__m_dctNetwork["w1"]
        mW2 = self.__m_dctNetwork["w2"]
        mW3 = self.__m_dctNetwork["w3"]

        b1 = self.__m_dctNetwork["b1"]
        b2 = self.__m_dctNetwork["b2"]
        b3 = self.__m_dctNetwork["b3"]

        mA1 = np.dot(mX, mW1) + b1
        mZ1 = sigmoid(mA1)
        mA2 = np.dot(mZ1, mW2) + b2
        mZ2 = sigmoid(mA2)
        mA3 = np.dot(mZ2, mW3) + b3
        mY = softmax(mA3)

        return mY
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

        return crossEntropyError(mY, mT)
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
        dctGradient["w3"] = calcNumGradient(loss_W, self.__m_dctNetwork["w3"])
        dctGradient["b3"] = calcNumGradient(loss_W, self.__m_dctNetwork["b3"])

        return dctGradient
    # End of myNetwork::calcNumGradient
# End of class myNetwork