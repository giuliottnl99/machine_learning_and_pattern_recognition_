import numpy as np
import sys
import matplotlib.pyplot as plt
import lab3Utils as ut
# import scipy
# import scipy.linalg


def load(fileName):
    matrixResult = []
    completeMatrixResult =  []
    with open(fileName) as file:
        for line in file:
            dataLine = line.replace("\n", "").split(",")
            matrixResult.append(dataLine[0:-1])
            completeMatrixResult.append(dataLine)
    return matrixResult, completeMatrixResult

def getCovarianceMatrix(pythonMatrix):
    matrix = np.array(pythonMatrix).astype(float).T
    avgVector= np.mean(matrix, axis=1)
    diffMatrix = matrix - avgVector.reshape(avgVector.size, 1)
    covarianceMatrix = np.dot(diffMatrix, diffMatrix.T) / matrix.shape[1]
    return covarianceMatrix

def plot(arrXtoPlot, arrYtoPlot, color, label):
        #plot x of matrixX and y of the specific matrix of the Y
        plt.xlabel('caratt 1')
        plt.ylabel('caratt 2')
        plt.scatter(np.ravel(arrXtoPlot).astype(float), np.ravel(arrYtoPlot).astype(float), label=label, color=color)
        plt.tight_layout()
        plt.legend()


def doPCA(dataSetMatrix, completeDataSetMatrix, plotGraph=True, reducedDim=2):
    covMatrix = getCovarianceMatrix(dataSetMatrix)
    print('cov matrix:')
    print(covMatrix)

    #find more important columns:
    #eigen decomposition:
    U, s, Vh = np.linalg.svd(covMatrix)
    print('vector of singular values:')
    print(s)
    #try to plot
    #vector of singular values: [4.20005343 0.24105294 0.0776881  0.02367619]
    #so get only first 2 dimensions:
    Vreduced = Vh.T[:, :reducedDim]
    print("shapes: ")
    print(Vreduced.shape)
    print(covMatrix.shape)

    reducedCovMatrix = covMatrix @ Vreduced
    print("reduced Cov Matrix:")
    print(reducedCovMatrix)
    #vector of singular values: [4.20005343 0.24105294 0.0776881  0.02367619]   
    #note: i see that only the first 2 values of the matrix are relevant, so I can get only the first 2 parts:
    #plot:
    setosaMatrix, versicolorMatrix, virginicaMatrix = ut.divideMatrix(completeDataSetMatrix)
    #try now if in this way it works:
    toPojectVector = Vh.T[:, 0:reducedDim]
    print('dimensions:')
    print(np.array(setosaMatrix).shape)
    print(np.array(toPojectVector).shape)
    print(np.array(setosaMatrix))
    print(np.array(toPojectVector))

    #fix here!   -> Try to use np.array!  
    setosaReducedMatrix = np.array(setosaMatrix) @ np.array(toPojectVector)
    versicolorReducedMatrix = versicolorMatrix @ toPojectVector
    virginicaReducedMatrix = virginicaMatrix @ toPojectVector
    
    if plotGraph==True:
        plt.figure()
        plt.gca().invert_yaxis()   
        plot(np.matrix(setosaReducedMatrix)[:, 0], np.matrix(setosaReducedMatrix)[:, 1], 'blue', 'setosa')
        plot(np.matrix(versicolorReducedMatrix)[:, 0], np.matrix(versicolorReducedMatrix)[:, 1], 'orange', 'versicolor')
        plot(np.matrix(virginicaReducedMatrix)[:, 0], np.matrix(virginicaReducedMatrix)[:, 1],'green', 'virginica')
        plt.show()
    return setosaReducedMatrix, versicolorReducedMatrix, virginicaReducedMatrix

def doLDA(completeDataSetMatrix, plotGraph=True, reducedDim=2):
    matrixSetosa, matrixVersicolor, matrixVirginica = ut.divideMatrix(completeDataSetMatrix)
    withinCovMatrix = ut.calculateWithinCovarianceMatrix(np.array(matrixSetosa).astype(float).T, np.array(matrixVersicolor).astype(float).T, np.array(matrixVirginica).astype(float).T)
    print('within cov matrix:')
    print(withinCovMatrix)
    betweenCovMatrix = ut.calculateBetweenCovarianceMatrix(np.array(matrixSetosa).astype(float).T, np.array(matrixVersicolor).astype(float).T, np.array(matrixVirginica).astype(float).T, np.array(dataSetMatrix).astype(float).T)
    print('between cov matrix:')
    print(betweenCovMatrix)

    #now I can do generalized eigenvalue problem
    rightSingularVectors, singularValuesWithin, _ = np.linalg.svd(withinCovMatrix)
    singularValuesMatrixSquared = np.diag(1.0/(singularValuesWithin**0.5))
    P1Matrix = rightSingularVectors @ singularValuesMatrixSquared @ rightSingularVectors.T

    covMatrixBetweenTransformed = P1Matrix @ betweenCovMatrix @ P1Matrix.T
    finalSingularRightsVectors, finalSingularValues, _ = np.linalg.svd(np.array(covMatrixBetweenTransformed))
    #choose how much you should reduce:
    finalResult = finalSingularRightsVectors[:, 0:reducedDim]
    print(finalResult)

    LDASubspaceSetosa = finalResult.T @ P1Matrix @ matrixSetosa.T
    LDASubspaceVersicolor = finalResult.T @ P1Matrix @ matrixVersicolor.T
    LDASubspaceVirginica = finalResult.T @ P1Matrix @ matrixVirginica.T

    if plotGraph==True:
        plt.figure()
        plt.gca().invert_xaxis()   
        plt.gca().invert_yaxis()   
        plot(np.matrix(LDASubspaceSetosa)[0, :], np.matrix(LDASubspaceSetosa)[1, :], 'blue', 'setosa')
        plot(np.matrix(LDASubspaceVersicolor)[0, :], np.matrix(LDASubspaceVersicolor)[1, :], 'orange', 'versicolor')
        plot(np.matrix(LDASubspaceVirginica)[0, :], np.matrix(LDASubspaceVirginica)[1, :], 'green', 'virginica')
        plt.show()
    return LDASubspaceSetosa, LDASubspaceVersicolor, LDASubspaceVirginica

def doBinaryClassification(completeDataSetMatrix):
    matrixSetosa, matrixVersicolor, matrixVirginica = ut.divideMatrix(completeDataSetMatrix)
    trainDataVersicolor, testDataVersicolor = ut.divideSamples(matrixVersicolor)
    trainDataVirginica, testDataVirginica = ut.divideSamples(matrixVirginica)
    ut.plot_hist(trainDataVersicolor, 'Versicolor', trainDataVirginica, 'Virginica', 5, 'Model training set')
    ut.plot_hist(testDataVersicolor, 'Versicolor', testDataVirginica, 'Virginica', 5, 'Model validation set')
    plt.show()

def doPCAAndLDAClassification(completeDataSetMatrix):
    #try to understand if I understood professor's task:
    matrixSetosa, matrixVersicolor, matrixVirginica = ut.divideMatrix(completeDataSetMatrix)
    setosaReducedMatrix, versicolorReducedMatrix, virginicaReducedMatrix = doPCA(dataSetMatrix, completeDataSetMatrix, plotGraph=False, reducedDim=3)
    finalSetosaReducedMatrix, finalVersicolorReducedMatrix, finalVirginicaReducedMatrix = doLDA(setosaReducedMatrix, versicolorReducedMatrix, virginicaReducedMatrix)
    return finalSetosaReducedMatrix, finalVersicolorReducedMatrix, finalVirginicaReducedMatrix

if __name__ == '__main__':
    dataSetMatrix, completeDataSetMatrix = load('..\lab2\iris.csv')
    doPCA(dataSetMatrix, completeDataSetMatrix)

    #now we can use LDA to improve everything:
    doLDA(completeDataSetMatrix)

    doBinaryClassification(completeDataSetMatrix)