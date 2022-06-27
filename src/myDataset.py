from typing import Tuple
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

def main():
    strImagePath = "./dataset/train-images.idx3-ubyte"
    strLabelPath = "./dataset/train-labels.idx1-ubyte"
    oDataset = myDataset(strImagePath, strLabelPath)
    return
# End of main

if "__main__" == __name__:
    main()