
#idea things to do for analysis:
# 1. get All data in a unique matrix and two matrices for fake and good values
# 3. Plot the histograms one by one for both matrices and the scatter plots too. 
# 4.

import numpy as np
import matplotlib.pyplot as plt


def loadData(fileName):
    completeSamplesMatrix = []
    genuineSamplesMatrix = []
    fakeSamplesMatrix = []

    with open(fileName) as file:
        for line in file:
            vectorLoaded = line.replace(" ", "").replace("\n", "").split(",")
            completeSamplesMatrix.append(vectorLoaded)
            if vectorLoaded[-1]=="1":
                genuineSamplesMatrix.append(vectorLoaded[0:-1])
            if vectorLoaded[-1]=="0":
                fakeSamplesMatrix.append(vectorLoaded[0:-1])
    return completeSamplesMatrix, genuineSamplesMatrix, fakeSamplesMatrix

def plotAllhist(trueSampleMatrix, fakeSampleMatrix):
    #matrices data of the same type are in the same column, so I get the transposed matrix. 
    # allSampleMatrixTransposed = np.array(allSampleMatrix).T
    trueSampleMatrixTransposed = np.array(trueSampleMatrix).T.astype(float)
    fakeSampleMatrixTransposed = np.array(fakeSampleMatrix).T.astype(float)

    #rows goes from 0 to 5 in the transposed matrix: one for each characteristic!
    for i in range(len(trueSampleMatrixTransposed)):
        rowTrue = trueSampleMatrixTransposed[i]
        rowFalse = fakeSampleMatrixTransposed[i]
        plt.figure()
        plt.xlabel("datus %d" % (i+1))
        plt.hist(rowTrue, bins=10, density=True, alpha=0.4, label= "Genuine samples", color="green")
        plt.hist(rowFalse, bins=10, density=True, alpha=0.4, label= "Counterfeit samples", color="red")
        plt.legend()
        plt.tight_layout() 
        plt.title("hist: %d" % (i+1))
        print("row true:")
        print(rowTrue)
        print("row false:")
        print(rowFalse)



def plotAllScattered(trueSampleMatrix, fakeSampleMatrix):
    trueSampleMatrixTransposed = np.array(trueSampleMatrix).T.astype(float)
    fakeSampleMatrixTransposed = np.array(fakeSampleMatrix).T.astype(float)
    for i in range(0, len(trueSampleMatrixTransposed), 2):
        rowTrue1 = trueSampleMatrixTransposed[i]
        rowFalse1 = fakeSampleMatrixTransposed[i]
        rowTrue2 = trueSampleMatrixTransposed[i+1]
        rowFalse2 = fakeSampleMatrixTransposed[i+1]

        plt.figure()
        plt.title("scatter: %d - %d" % (i+1, i+2))
        plt.xlabel("datus %d" % (i+1))
        plt.ylabel=("datus %d" % (i+2))
        plt.scatter(rowTrue1, rowTrue2, label="Genuine sample", color="green")
        plt.scatter(rowFalse1, rowFalse2, label="Counterfeit sample", color="red")
        plt.legend()
        plt.tight_layout() 
        plt.title("scatter: %d - %d" % (i+1, i+2))


if __name__ == '__main__':
    completeSamplesMatrix, genuineSamplesMatrix, fakeSamplesMatrix = loadData('trainData.txt')
    plotAllhist(genuineSamplesMatrix, fakeSamplesMatrix)
    plotAllScattered(genuineSamplesMatrix, fakeSamplesMatrix)
    plt.show()


