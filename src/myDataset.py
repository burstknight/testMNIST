from typing import Tuple
from attr import s
import numpy as np
import cv2

class myDataset:
    """
    Description:
    ============================================
    This class can read MNIST dataset.
    """

    def __init__(self, strImagePath:str, strLabelPath:str) -> None:
        """
        Description:
        ============================================================
        Initialize this object of this class.

        Args:
        ============================================================
        - strImagePath: ptype: str, the path of the images for mnist dataset
        - strLabelPath: ptype: str, the path of the labels for mnist dataset
        """
        self.__m_vmImages = []
        self.__m_viLabels = []
        self.__m_iRows = 0
        self.__m_iCols = 0

        self.__load(strImagePath, strLabelPath)
    # End of constructor

    @property
    def iRows(self):
        return self.__m_iRows
    # End of myDataset::iRows(getter)

    @property
    def iCols(self):
        return self.__m_iCols
    # End of myDataset::iCols(getter)

    @property
    def iNumOfItems(self):
        return len(self.__m_vmImages)
    # End of myDataset::iNumOfItems(getter)

    def getItem(self, index:int) -> Tuple[np.ndarray, int]:
        """
        Description:
        ============================================================
        Get an item in the dataset.

        Args:
        ============================================================
        - index:    ptype: int, the index of the item.

        Returns:
        ============================================================
        - rtype: np.ndarray, the image
        - rytpe: np.ndarray, the label
        """
        if(index >= self.iNumOfItems):
            raise IndexError("The index is out of the range in myDataset::getItem(): %d >= %d" %(index, self.iNumOfItems))
        # End of if-condition

        return self.__m_vmImages[index], self.__m_viLabels[index]
    # End of myDataset::getItem

    def __load(self, strImagePath:str, strLabelPath:str):
        """
        Description:
        ============================================================
        Load mnist dataset.

        Args:
        ============================================================
        - strImagePath: ptype: str, the path of the images for mnist dataset
        - strLabelPath: ptype: str, the path of the labels for mnist dataset
        """
        print("Loading dataset...", end="")
        with open(strImagePath, "rb") as oRead:
            iMagicNumber = int.from_bytes(oRead.read(4), "big")
            iNumOfImages = int.from_bytes(oRead.read(4), "big")
            self.__m_iRows = int.from_bytes(oRead.read(4), "big")
            self.__m_iCols = int.from_bytes(oRead.read(4), "big")
            iSizeOfImage = self.__m_iRows*self.__m_iCols
            for _ in range(iNumOfImages):
                mImage = np.zeros((iSizeOfImage, ), dtype="uint8")
                for j in range(iSizeOfImage):
                    mImage[j] = int.from_bytes(oRead.read(1), "big")
                # End of for-loop
                self.__m_vmImages.append(mImage)
            # End of for-loop
        # End of with-block

        with open(strLabelPath, "rb") as oRead:
            iMagicNumber = int.from_bytes(oRead.read(4), "big")
            iNumOfItems = int.from_bytes(oRead.read(4), "big")
            for _ in range(iNumOfItems):
                iLabel = int.from_bytes(oRead.read(1), "big")
                self.__m_viLabels.append(iLabel)
            # End of for-loop
        # End of with-block

        print(" Done!")
    # End of myDataset::load
# End of class myDataset

class myDatasetIterator:
    """
    Description:
    =========================================================
    This class is an iterator that can travel whole dataset.
    """
    def __init__(self, oDataset:myDataset, iBatchSize:int=1) -> None:
        self.__m_oDataset = oDataset
        self.__m_viIndex = []
        self.__m_iIndex = 0
        self.__m_iBatchSize = iBatchSize
    # End of constructor

    @property
    def iNumOfItems(self):
        return self.__m_oDataset.iNumOfItems
    # End of myDatasetIterator::iNumOfItems(getter)

    def init(self, isShuffle:bool=True):
        """
        Description:
        ======================================================
        Initialize fields to travel dataset.

        Args:
        ======================================================
        - isShuffle:    ptype: bool, this flag can control to shuffle the dataset or not.

        Returns:
        ======================================================
        - rtype: void
        """
        self.__m_viIndex.clear()
        for i in range(self.__m_oDataset.iNumOfItems):
            self.__m_viIndex.append(i)
        # End of for-loop

        if(False == isShuffle):
            return
        # End of if-condition

        for i in range(len(self.__m_viIndex)):
            index = np.random.randint(0, self.__m_oDataset.iNumOfItems)
            tmp = self.__m_viIndex[i]
            self.__m_viIndex[i] = self.__m_viIndex[index]
            self.__m_viIndex[index] = tmp
        # End of for-loop
    # End of myDatasetIterator::init

    def getExample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Description:
        ======================================================
        Get examples.

        Args:
        ======================================================
        - isFlatten:    ptype: bool, this flag can control to return a 1D array of the image.

        Returns: 
        ======================================================
        - rtype: np.ndarray, the batch images
        - rtype: np.ndarray, the batch labels
        """
        mBatchImage = np.zeros((self.__m_iBatchSize, self.__m_oDataset.iCols*self.__m_oDataset.iRows), dtype="uint8")
        mLabel = np.zeros((self.__m_iBatchSize, 10), dtype="uint8")
        for i in range(self.__m_iBatchSize):
            mImage, iLabel = self.__m_oDataset.getItem(self.__m_viIndex[self.__m_iIndex + i])
            mBatchImage[i, :] = mImage 
            mLabel[i, iLabel] = 1
        # End of for-loop

        return mBatchImage, mLabel
    # End of myDatasetIterator::getExample

    def moveNext(self):
        """
        Description:
        ======================================================
        Move the index to next item.
        """
        if(self.__m_iIndex + self.__m_iBatchSize >= len(self.__m_viIndex)):
            return
        # End of if-condition

        self.__m_iIndex += self.__m_iBatchSize
# End of class myDatasetIterator

def main():
    strImagePath = "./dataset/train-images.idx3-ubyte"
    strLabelPath = "./dataset/train-labels.idx1-ubyte"
    oDataset = myDataset(strImagePath, strLabelPath)
    oIter = myDatasetIterator(oDataset)
    oIter.init()
    
    for i in range(oIter.iNumOfItems):
        mBatchImage, mLabel = oIter.getExample()
        print("%08d: shape: %s, label: %d" %(i, str(mBatchImage.shape), mLabel[0]))
        oIter.moveNext()
    # End of for-loop
    return
# End of main

if "__main__" == __name__:
    main()