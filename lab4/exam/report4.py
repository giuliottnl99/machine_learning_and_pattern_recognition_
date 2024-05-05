import numpy as np
import matplotlib.pyplot as plt
import sys


def loadReport():
    matrixLoaded = []
    with open('trainData.txt') as file:
        for line in file:
            matrixLoaded.append(line.split(" , "))
    return np.array(matrixLoaded).astype(float)


def logpdf_GAU_ND(x, mu, C):
    Cinv = np.linalg.inv(C)
    CLogDet = np.linalg.slogdet(C)[1]
    _ , CInvLogDet = np.linalg.slogdet(Cinv)
    M =  x.shape[0]
    component1 = - (M/2)*np.log(np.pi*2)
    component2 = -0.5 * CLogDet
    component3 = - (1/2) * ((x - mu)* (Cinv @(x-mu))).sum(0) #note: 
    
    return component1 + component2 + component3

def plotGraphHistWithGaussian(x, mu, cov, i):
    plt.figure()
    plt.title("hist for characteristic %d" %(i))
    plt.hist(x.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(XPlot.reshape(1, XPlot.size), mu, cov)))
    


#load all dataset:
if __name__ == '__main__':
    transposedMatrix = (np.transpose(loadReport()))[0:-1, :]
    #plot:
    for i in range(len(transposedMatrix)):
        row = transposedMatrix[i, :].reshape(1, len(transposedMatrix[i])) #resize the row
        mu = row.mean(1).reshape(1, 1)
        print(mu)
        cov = ((row-mu) @ (row - mu).T) / row.shape[1]
        print(cov)
        plotGraphHistWithGaussian(row, mu, cov, i+1)
    plt.show()

    


    

