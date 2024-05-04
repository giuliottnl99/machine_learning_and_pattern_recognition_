import numpy as np
import matplotlib.pyplot as plt


def divideSamples(allSamples):
    allSamplesArray = np.ravel(allSamples)
    permutedSequence =  np.random.permutation(allSamplesArray)
    totalSize = len(permutedSequence)
    trainElemQt = totalSize*2/3
    trainData = []
    testData = []
    for i in range(totalSize):
        elem = permutedSequence[i]
        if i<trainElemQt:
            trainData.append(elem)
        else:
            testData.append(elem)
    return trainData, testData    

def plot_hist(arrXtoPlot, labelX, arrYToPlot, labelY, intervals, title):
    plt.figure()
    plt.title(title)
    plt.hist(np.ravel(arrXtoPlot).astype(float), bins=intervals, density=True, ec='black', alpha = 0.4, color='red', label=labelX)
    plt.hist(np.ravel(arrYToPlot).astype(float), bins=intervals, density=True, ec='black', alpha = 0.4, color='blue', label=labelY)

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
