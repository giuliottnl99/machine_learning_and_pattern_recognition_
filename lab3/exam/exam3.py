# import utils as ut
import numpy as np
import sys
import utils as ut
import matplotlib.pyplot as plt


def plot6Directions(DProj, L):
    for i in range(DProj.shape[0]):
        rowCharacteristic = DProj[i, :]
        arrGood = rowCharacteristic[:, L==1]
        arrFake = rowCharacteristic[:, L==0]
        plt.figure()
        plt.title('Characteristic number: %d of relevance' % (i+1))
        ut.plotHist(arrGood, c="green", l="good samples", i=10)
        ut.plotHist(arrFake, c="red", l="fake samples", i=10)
    plt.show()

def computeAndPlotPCA6Dim(D, L, toPlot=False):
    PCACompleteReducingMatrix = ut.computePCA_ReducingMatrix(D, L, dim=6)
    DProj6PCA = PCACompleteReducingMatrix @ D
    plot6Directions(DProj6PCA, L)

def computeAndPlotLDA6Dim(D, L, toPlot=False):
    LDACompleteReducingMatrix = ut.computeLDA_ReducingMatrix(D, L, dim=6)
    DProj6LDA = LDACompleteReducingMatrix @ D
    plot6Directions(DProj6LDA, L)


def findMaximumThreshold(D, L):
    maximumThreshold = None
    maximumScore = 0.0
    for thr in np.arange(-4, 4.01, 0.01):
        score = ut.doBinaryClassification_PCA_LDA(D, L, toPlot=False, toPrint=False, chosenMethod='LDA', LValueTrue=1, LValueFalse=0, threshold=thr)
        if (score > maximumScore):
            maximumScore = score
            maximumThreshold = thr
    return maximumThreshold, maximumScore

def applyPCAFirstAndThenLDA(D, L):
    for i in [5, 4, 3, 2]:
        print("Try reducing to %d using PCA before LDA" % (i) )
        percAcc = ut.doBinaryClassification_PCA_LDA(D, L, toPlot=False, toPrint=True, chosenMethod='bothPCA_LDA', dimensionsPCA=i, LValueTrue=1, LValueFalse=0, threshold=None)




if __name__ == '__main__':
    D = np.load('general_utils/D_exam_train.npy')
    L = np.load('general_utils/L_exam_train.npy')
    #first plot PCA on 6 dimensions:
    computeAndPlotPCA6Dim(D, L, toPlot=False)
    #then plot LDA on 6 dimensions:
    computeAndPlotLDA6Dim(D, L, toPlot=False)
    #use LDA as classifier:
    #Binary classification applying LDA and using normal threshold:
    ut.doBinaryClassification_PCA_LDA(D, L, toPlot=False, chosenMethod='LDA', LValueTrue=1, LValueFalse=0)
    print("try using median:")
    ut.doBinaryClassification_PCA_LDA(D, L, toPlot=False, chosenMethod='LDA', LValueTrue=1, LValueFalse=0, threshold='Median')
    print("try finding the best threshold with findMaximumThreshold method")
    # maxTh, maxScore = findMaximumThreshold(D, L)
    # print("optimizing threshold is %f with a score of %f" % (maxTh, maxScore*2000))
    applyPCAFirstAndThenLDA(D, L)
