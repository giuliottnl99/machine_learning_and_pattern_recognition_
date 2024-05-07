import sklearn.datasets
import numpy as np 
import utils as ut

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def split_DB_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

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


def logpdf_GAU_ND(x, mu, C):
    Cinv = np.linalg.inv(C)
    CLogDet = np.linalg.slogdet(C)[1]
    _ , CInvLogDet = np.linalg.slogdet(Cinv)
    M =  x.shape[0]
    component1 = - (M/2)*np.log(np.pi*2)
    component2 = -0.5 * CLogDet
    component3 = - (1/2) * ((x - mu).T @ (Cinv @(x-mu))).sum(0) #note: 
    
    return component1 + component2 + component3
    
#return of type class -> array of likelihoods 
def computeLogLikelihoodForEachClass(dataSetsSplitted, muAndCovDivided):
    scoreMatrix = []
    for i in range(len(dataSetsSplitted)):
        classDataSet = dataSetsSplitted[i]
        mu, cov = muAndCovDivided[i][0], muAndCovDivided[i][1]
        scoreMatrix.append(logpdf_GAU_ND(classDataSet, mu, cov))
    print('scoreMatrix:')
    print(scoreMatrix)
    return scoreMatrix

def computeJointDensities(scoreMatrix):
    SJoint = []
    #note: classData is a matrix!
    for classData in scoreMatrix:
        classDataToArr = classData.ravel()
        SJoint.append(classDataToArr/classDataToArr.shape[1])
    return SJoint

#classs -> Transposed matrix
def trasposeDataSetSplitted(dataSetSplitted):
    dataSetSplittedTransposed = []
    for i in range(len(dataSetSplitted)):
        matrixDataTransposed = np.matrix(dataSetSplitted[i]).T
        dataSetSplittedTransposed.append(matrixDataTransposed)
    return dataSetSplittedTransposed

def computeMuAndCov(x):
    # x= np.array(arr)
    muArr = x.mean(1)
    mu = np.matrix(x.mean(1)).reshape(len(muArr), 1)
    cov = ((x - mu) @ (x-mu).T) / x.shape[1]
    return mu, cov


def computeMuAndCovForClass(dataSetSplitted):
    muAndCovForElement = []
    for classParameters in dataSetSplitted:
        mu, cov = computeMuAndCov(classParameters)
        muAndCovForElement.append([mu, cov])
    return muAndCovForElement



if __name__ == '__main__':
    D, L = load_iris()
    (DTR, LTR), (DVAL, LVAL) = split_DB_2to1(D, L)

    #first try: split dataset based on whole matrix;
    dataSetsSplitted = splitDataSets(DVAL, LVAL)
    dataSetsSplittedTransposed = trasposeDataSetSplitted(dataSetsSplitted)
    #I think mistakes starts from here:
    muAndCovDivided = computeMuAndCovForClass(dataSetsSplittedTransposed)
    #first: compute the likelihoods:
    scoreMatrix = computeLogLikelihoodForEachClass(dataSetsSplittedTransposed, muAndCovDivided)
    #now compute joint densities:
    SJoint = computeJointDensities(scoreMatrix)
    #now compare:
    print('SJoint found by me:')
    print(SJoint)
    print('SJoint proposed by corrections:')
    print(np.load('SJoint_MVG.npy'))





