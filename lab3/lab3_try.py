import numpy as np
import sys
import matplotlib.pyplot as plt


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
    return matrixSetosa, matrixVersicolor, matrixVirginica

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


def loadMatrixTemp2():
    dataSetMatrix, completeDataSetMatrix = load('..\lab2\iris.csv')
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
    plt.figure()
    plot(np.matrix(setosaMatrix)[:, 0], np.matrix(setosaMatrix)[:, 1], 'blue', 'setosa')
    plot(np.matrix(versicolorMatrix)[:, 0], np.matrix(versicolorMatrix)[:, 1], 'orange', 'versicolor')
    plot(np.matrix(virginicaMatrix)[:, 0], np.matrix(virginicaMatrix)[:, 1],'green', 'virginica')
    plt.show()


if __name__ == '__main__':
    # loadMatrixTemp2()
    #issue: graph result is not the same as the wanted one. 
    #Anyway, covariance matrix is fine



