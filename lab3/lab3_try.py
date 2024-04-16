import numpy as np
import sys
import matplotlib.pyplot as plt
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

def divideMatrix(completeMatrixResult):
    matrixSetosa = []
    matrixVersicolor = []
    matrixVirginica = []
    for element in completeMatrixResult:
         rowType = element[-1]
         if rowType=='Iris-setosa':
            matrixSetosa.append(element[0:-1])
         if rowType=='Iris-versicolor':
            matrixVersicolor.append(element[0:-1])
         if rowType=='Iris-virginica':
            matrixVirginica.append(element[0:-1])
    return np.array(matrixSetosa).astype(float), np.array(matrixVersicolor).astype(float), np.array(matrixVirginica).astype(float)

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


def doPCA(dataSetMatrix, completeDataSetMatrix):
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
    Vreduced = Vh.T[:, :2]
    print("shapes: ")
    print(Vreduced.shape)
    print(covMatrix.shape)

    reducedCovMatrix = covMatrix @ Vreduced
    print("reduced Cov Matrix:")
    print(reducedCovMatrix)
    #vector of singular values: [4.20005343 0.24105294 0.0776881  0.02367619]   
    #note: i see that only the first 2 values of the matrix are relevant, so I can get only the first 2 parts:
    #plot:
    setosaMatrix, versicolorMatrix, virginicaMatrix = divideMatrix(completeDataSetMatrix)
    #try now if in this way it works:
    toPojectVector = Vh.T[:, 0:2]
    print('dimensions:')
    print(np.array(setosaMatrix).shape)
    print(np.array(toPojectVector).shape)
    print(np.array(setosaMatrix))
    print(np.array(toPojectVector))

    #fix here!   -> Try to use np.array!  
    setosaReducedMatrix = np.array(setosaMatrix) @ np.array(toPojectVector)
    versicolorReducedMatrix = versicolorMatrix @ toPojectVector
    virginicaReducedMatrix = virginicaMatrix @ toPojectVector
       
    plt.figure()
    plt.gca().invert_yaxis()   
    plot(np.matrix(setosaReducedMatrix)[:, 0], np.matrix(setosaReducedMatrix)[:, 1], 'blue', 'setosa')
    plot(np.matrix(versicolorReducedMatrix)[:, 0], np.matrix(versicolorReducedMatrix)[:, 1], 'orange', 'versicolor')
    plot(np.matrix(virginicaReducedMatrix)[:, 0], np.matrix(virginicaReducedMatrix)[:, 1],'green', 'virginica')
    plt.show()

#pass transposed matrix!
def calculateWithinCovarianceMatrix(matrixSetosaT, matrixVersicolorT, matrixVirginicaT):
    # betweenClassCovMatrix
    #first compute average for each class:
    avgSetosa = np.mean(matrixSetosaT, axis=1)
    avgVersicolor = np.mean(matrixVersicolorT, axis=1)
    avgVirginica = np.mean(matrixVirginicaT, axis=1)

    #calculate the within diff matrices:
    matrixDiffSetosa = matrixSetosaT - avgSetosa.reshape(avgSetosa.size, 1)
    matrixDiffVersicolor = matrixVersicolorT - avgVersicolor.reshape(avgVersicolor.size, 1)
    matrixDiffVirginica = matrixVirginicaT - avgVirginica.reshape(avgVirginica.size, 1)
    
    #calculate the within components of cov matrix one by one:
    matrixPartSetosa = matrixDiffSetosa @ matrixDiffSetosa.T
    matrixPartVersicolor = matrixDiffVersicolor @ matrixDiffVersicolor.T
    matrixPartVirginica = matrixDiffVirginica @ matrixDiffVirginica.T

    withinCovMatrix = (matrixPartSetosa+matrixPartVersicolor+matrixPartVirginica) / (150)
    return withinCovMatrix



def calculateBetweenCovarianceMatrix(matrixSetosaT, matrixVersicolorT, matrixVirginicaT, matrixGlobalT):
    #calculate between cov matrix:
    avgGlobal = np.mean(matrixGlobalT, axis=1)
    avgSetosa = np.mean(matrixSetosaT, axis=1)
    avgVersicolor = np.mean(matrixVersicolorT, axis=1)
    avgVirginica = np.mean(matrixVirginicaT, axis=1)

    
    #calculate diff matrices
    vectorDiffSetosa = avgSetosa.reshape(avgSetosa.size, 1) - avgGlobal.reshape(avgGlobal.size, 1)
    vectorDiffVersicolor = avgVersicolor.reshape(avgVersicolor.size, 1) - avgGlobal.reshape(avgGlobal.size, 1)
    vectorDiffVirginica = avgVirginica.reshape(avgVirginica.size, 1) - avgGlobal.reshape(avgGlobal.size, 1)
    
    #calculate the between components of cov matrix one by one:
    matrixPartSetosa =  vectorDiffSetosa @ vectorDiffSetosa.T
    matrixPartVersicolor = vectorDiffVersicolor @ vectorDiffVersicolor.T
    matrixPartVirginica = vectorDiffVirginica @ vectorDiffVirginica.T
    #sum and make average:
    betweenMatrixTotal = (matrixPartSetosa*matrixPartSetosa.shape[0] + matrixPartVersicolor*matrixPartVersicolor.shape[0] + matrixPartVirginica*matrixPartVirginica.shape[0])
    betweenCovMatrix = betweenMatrixTotal / (matrixPartSetosa.shape[0] + matrixPartVersicolor.shape[0] + matrixPartVirginica.shape[0])
    return betweenCovMatrix

def doLDA(completeDataSetMatrix):
    matrixSetosa, matrixVersicolor, matrixVirginica = divideMatrix(completeDataSetMatrix)
    withinCovMatrix = calculateWithinCovarianceMatrix(np.array(matrixSetosa).astype(float).T, np.array(matrixVersicolor).astype(float).T, np.array(matrixVirginica).astype(float).T)
    print('within cov matrix:')
    print(withinCovMatrix)
    betweenCovMatrix = calculateBetweenCovarianceMatrix(np.array(matrixSetosa).astype(float).T, np.array(matrixVersicolor).astype(float).T, np.array(matrixVirginica).astype(float).T, np.array(dataSetMatrix).astype(float).T)
    print('between cov matrix:')
    print(betweenCovMatrix)

    #now I can do generalized eigenvalue problem
    rightSingularVectors, singularValuesWithin, _ = np.linalg.svd(withinCovMatrix)
    singularValuesMatrixSquared = np.diag(1.0/(singularValuesWithin**0.5))
    P1Matrix = rightSingularVectors @ singularValuesMatrixSquared @ rightSingularVectors.T

    covMatrixBetweenTransformed = P1Matrix @ betweenCovMatrix @ P1Matrix.T
    finalSingularRightsVectors, finalSingularValues, _ = np.linalg.svd(np.array(covMatrixBetweenTransformed))
    #choose how much you should reduce:
    finalResult = finalSingularRightsVectors[:, 0:2]
    print(finalResult)

    LDASubspaceSetosa = finalSingularRightsVectors.T @ P1Matrix @ matrixSetosa
    LDASubspaceVersicolor = finalSingularRightsVectors.T @ P1Matrix @ matrixVersicolor
    LDASubspaceVirginica = finalSingularRightsVectors.T @ P1Matrix @ matrixVirginica

    #try to plot: will it works?
    plt.figure()
    plt.gca().invert_yaxis()   
    plot(np.matrix(LDASubspaceSetosa)[:, 0], np.matrix(LDASubspaceSetosa)[:, 1], 'blue', 'setosa')
    plot(np.matrix(LDASubspaceVersicolor)[:, 0], np.matrix(LDASubspaceVersicolor)[:, 1], 'blue', 'setosa')
    plot(np.matrix(LDASubspaceVirginica)[:, 0], np.matrix(LDASubspaceVirginica)[:, 1], 'blue', 'setosa')
    plt.show()


if __name__ == '__main__':
    dataSetMatrix, completeDataSetMatrix = load('..\lab2\iris.csv')
    # doPCA(dataSetMatrix, completeDataSetMatrix)

    #now we can use LDA to improve everything:
    doLDA(completeDataSetMatrix)
