import numpy as np
import sys
import matplotlib.pyplot as plt
import utils as utG
# import scipy
# import scipy.linalg

def load_iris(): # Same as in pca script
    
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']


def plotPCA(D, L):
    reducingMatrix = utG.computePCA_ReducingMatrix(D, L, dim=2)
    D_PCA = reducingMatrix @ D
    plt.figure()
    # plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    utG.plotScatter(D_PCA[0, L==0], D_PCA[1, L==0], l='Setosa', c="blue")
    utG.plotScatter(D_PCA[0, L==1], D_PCA[1, L==1], l='Versicolor', c="orange")
    utG.plotScatter(D_PCA[0, L==2], D_PCA[1, L==2], l='virginica', c="green")
    plt.plot()
    plt.show()

def plotLDA(D, L):
    reducingMatrix = utG.computeLDA_ReducingMatrix(D, L)
    projD = reducingMatrix @ D
    plt.figure()
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    utG.plotScatter(projD[0, L==0], projD[1, L==0], l='Setosa', c="blue")
    utG.plotScatter(projD[0, L==1], projD[1, L==1], l='Versicolor', c="yellow")
    utG.plotScatter(projD[0, L==2], projD[1, L==2], l='virginica', c="green")
    plt.plot()
    plt.show()

def computeAndRevertProj_DT_DV(DT, DV, LT, reducingMatrix):
    projDT = reducingMatrix @ DT
    if projDT[0, LT==1].mean(1)[0, 0] > projDT[0, LT==2].mean(1)[0, 0]:    
        reducingMatrix = - reducingMatrix
    projDT = reducingMatrix @ DT
    projDV = reducingMatrix @ DV
    return projDT, projDV


def doBinaryClassification(D, L, toPlot=False, chosenMethod='LDA'):
    #first of all, remove data where the data label is ==0 keeping 1 and 2 only
    dataSetFiltered = D[:, L!=0]
    labelSetFiltered = L[L!=0]
    #divide training and test set:
    (DT, LT), (DV, LV) = utG.divideSamplesRandomly(dataSetFiltered, labelSetFiltered)
    #doLDA
    reducingMatrix = None
    projDT = None
    projDL = None
    if chosenMethod == 'LDA':
        reducingMatrix = utG.computeLDA_ReducingMatrix(DT, LT, dim=1)
    if chosenMethod == 'PCA':
        reducingMatrix = utG.computePCA_ReducingMatrix(DT, LT, dim=1)

    if chosenMethod=='bothPCA_LDA':
        reducingMatrixPCA = utG.computePCA_ReducingMatrix(DT, LT, dim=2)
        projDT, projDV = computeAndRevertProj_DT_DV(DT, DV, LT, reducingMatrixPCA)
        reducingMatrixLDA = utG.computeLDA_ReducingMatrix(projDT, LT, dim=1)
        projDT, projDV = computeAndRevertProj_DT_DV(projDT, projDV, LT, reducingMatrixLDA)
    else:
        projDT, projDV = computeAndRevertProj_DT_DV(DT, DV, LT, reducingMatrix)

    #check if virginica class samples are on the right of the Versicolor samples:
    threshold = ( projDT[0, LT==1].mean() + projDT[0, LT==2].mean() ) / 2.0
    #compute number of times projDV is good
    PV = np.matrix(np.zeros(shape=LV.shape))
    PV[projDV >= threshold] = 2
    PV[projDV < threshold] = 1
    matrixValidSamples = [PV[0, i] for i in range(len(LV)) if LV[i]==PV[0, i]]
    #expected: 32 out of 34
    print("there are %d matches out of %d classes" % (len(matrixValidSamples), len(LV)))

    if toPlot:
        #first plot test sample reduced:
        plt.figure()
        plt.title("Training set")
        utG.plotHist(projDT[0, LT==1], c="orange", i=5, l="Versicolor")
        utG.plotHist(projDT[0, LT==2], c="green", i=5, l="Virginica")
        #then plot validation sample reduced:
        plt.figure()
        plt.title("Validation set")
        utG.plotHist(projDV[0, LV==1], c="orange", i=5, l="Versicolor")
        utG.plotHist(projDV[0, LV==2], c="green", i=5, l="Virginica")
        
        plt.show()

if __name__ == '__main__':
    D, L = load_iris()
    #plot using PCA: works
    # plotPCA(D, L)

    #plot using LDA: works:
    # plotLDA(D, L)

    #then we can procede with binary classification for LDA:
    print("apply LDA")
    doBinaryClassification(D, L, toPlot=False, chosenMethod='LDA')
    # and for PCA:
    print("apply PCA")
    doBinaryClassification(D, L, toPlot=False, chosenMethod='PCA')
    print("apply both PCA and LDA")
    doBinaryClassification(D, L, toPlot=False, chosenMethod='bothPCA_LDA')
