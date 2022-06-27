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