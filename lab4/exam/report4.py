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

def plotGraphHistWithGaussian(x, mu, cov, i, color, label):
    plt.hist(x.ravel(), bins=50, density=True, alpha=0.4, color=color, label=label)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(XPlot.reshape(1, XPlot.size), mu, cov)))
    


#load all dataset:
if __name__ == '__main__':
    D, L = np.load("general_utils/D_exam_train.npy"), np.load("general_utils/L_exam_train.npy")
    #plot:
    DTrue = D[:, L==1]
    DFalse = D[:, L==0]
    for i in range(D.shape[0]): #for each row (feature)
        plt.figure()
        plt.title("hist for feature %d" %(i+1))

        rowTrue = DTrue[i, :].reshape(1, len(DTrue[i])) #resize the row
        mu = rowTrue.mean(1).reshape(1, 1)
        print(mu)
        cov = ((rowTrue-mu) @ (rowTrue - mu).T) / rowTrue.shape[1]
        print(cov)
        plotGraphHistWithGaussian(rowTrue, mu, cov, i+1, "green", "Genuine sample")

        rowFalse = DFalse[i, :].reshape(1, len(DFalse[i])) #resize the row
        mu = rowFalse.mean(1).reshape(1, 1)
        print(mu)
        cov = ((rowFalse-mu) @ (rowFalse - mu).T) / rowFalse.shape[1]
        print(cov)
        plotGraphHistWithGaussian(rowFalse, mu, cov, i+1, "red", "Cunterfeit sample")
        plt.legend()

    plt.show()
    


    

