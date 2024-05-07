import numpy as np

def computeMuAndCov(arr):
    x= np.array(arr).T
    muArr = x.mean(1)
    mu = np.matrix(x.mean(1)).reshape(len(muArr), 1)
    cov = ((x - mu) @ (x-mu).T) / len(arr)
    return mu, cov

#dataset splitted is: [0-1-2 as name] [correspondent matrix of elements]
def splitDataSets(fullArr, labels):
    dataSetSplitted = [[], [], []]
    for  i in range(len(labels)):
        row = fullArr[:, i]
        label = labels[i]
        if label==0:
            dataSetSplitted[0].append(row)
        elif label==1:
            dataSetSplitted[1].append(row)
        elif label==2:
            dataSetSplitted[2].append(row)
    return dataSetSplitted    

#each class is rappresented by a row -> Each row contains mu and cov

def logpdf_GAU_ND(x, mu, C):
    Cinv = np.linalg.inv(C)
    CLogDet = np.linalg.slogdet(C)[1]
    _ , CInvLogDet = np.linalg.slogdet(Cinv)
    M =  x.shape[0]
    component1 = - (M/2)*np.log(np.pi*2)
    component2 = -0.5 * CLogDet
    component3 = - (1/2) * ((x - mu) @ (Cinv @(x-mu))).sum(0) #note: 
    
    return component1 + component2 + component3

