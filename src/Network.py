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

    def initWeight(self):
        """
        Description:
        ==================================================
        Initialize the wieghts for training.
        """
        for strKey in self.__m_dctNetwork.keys():
            for i in range(self.__m_dctNetwork[strKey].size):
                self.__m_dctNetwork[strKey].data[i] = np.random.uniform(-1.0, 1.0)
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
# End of class myNetwork