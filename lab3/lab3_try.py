import numpy as np

def getCovarianceMatrix(pythonMatrix):
    matrix = np.array(pythonMatrix)
    avgVector= matrix.mean(1)
    diffMatrix = matrix - avgVector.reshape(avgVector.size, 1)
    covarianceMatrix = np.dot(diffMatrix, diffMatrix.T) #/ avgVector.size
    return covarianceMatrix


if __name__ == '__main__':
    arrResult = np.load('IRIS_PCA_matrix_m4.npy')
    print("array:")
    print(arrResult) 
    covMatrix = getCovarianceMatrix(arrResult)
    print('cov matrix:')
    print(covMatrix)
    covMatrixNp = np.cov(arrResult, rowvar=False)
    print('cov matrix by numpy:')
    print(covMatrixNp)
