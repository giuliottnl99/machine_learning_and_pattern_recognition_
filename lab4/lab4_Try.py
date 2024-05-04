import numpy as np
import matplotlib.pyplot as plt
import pdb




def logpdf_GAU_ND(x, mu, C):
    # print(x)
    Cinv = np.linalg.inv(C)
    CLogDet = np.linalg.slogdet(C)[1]
    _ , CInvLogDet = np.linalg.slogdet(Cinv)
    M =  x.shape[0]
    component1 = - (M/2)*np.log(np.pi*2)
    component2 = -0.5 * CLogDet
    component3 = - (1/2) * ((x - mu)* (Cinv @(x-mu))).sum(0) #note: 
    
    return component1 + component2 + component3

 #x is the matrix of elements (the array of the arrays of data)
def logLikelihood(x, hist=False):
    mu = x.mean(1).reshape(x.shape[0], 1)
    cov = ( (x-mu) @ ((x-mu).T) )/(x.shape[1])
    #it's the sum for each element of the log-density
    result = logpdf_GAU_ND(x, mu, cov).sum()

    #let's hist:
    if hist:
        plt.figure()
        plt.hist(x.ravel(), bins=50, density=True)
        XPlot = np.linspace(-8, 12, 1000)
        plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(XPlot.reshape(1, XPlot.size), mu, cov)))
        plt.show()
    return result
    
if __name__ == '__main__':
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(XPlot.reshape(1, XPlot.size), m, C)))
    plt.show()
    #for XND
    XND = np.load('XND.npy')
    print('result for XND:')
    print(logLikelihood(XND, hist=False)) 

    #then for X1D:
    X1D = np.load('X1D.npy')
    print('result for XND:')
    print(logLikelihood(X1D, hist=True))