import sklearn.datasets
import numpy as np 
import utils as ut
import scipy

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
    P = np.linalg.inv(C)
    # return -0.5*x.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x-mu).T @ (P @ (x-mu))).sum(0)
    return -0.5*x.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * (np.array(x-mu) * (np.array(P) @ np.array(x-mu))).sum(0)    
#return of type class -> array of likelihoods 
def computeLogLikelihoodForEachClassWithClasses(dataSetsSplitted, muAndCovDivided):
    scoreMatrix = []
    for i in range(len(dataSetsSplitted)):
        classDataSet = dataSetsSplitted[i]
        mu, cov = muAndCovDivided[i][0], muAndCovDivided[i][1] #here it breaks!
        scoreMatrix.append(logpdf_GAU_ND(classDataSet, mu, cov))
    print('scoreMatrix:')
    print(scoreMatrix)
    return scoreMatrix

def computeLogLikelihoodForEachClassWithoutClasses(AllSamples, muAndCovDivided):
    scoreMatrix = []

    for i in range(len(muAndCovDivided)):
        mu, cov = muAndCovDivided[i][0], muAndCovDivided[i][1] #here it breaks!
        scoreMatrix.append(logpdf_GAU_ND(AllSamples, mu, cov))
    print('scoreMatrix:')
    print(scoreMatrix)
    return scoreMatrix


def computeLogPosterior(logScoreMatrix, v_prior):
    #note: classData is a matrix!
    v_priorLog = np.log(v_prior)
    logSJoint = logScoreMatrix + v_priorLog.reshape(v_priorLog.size, 1)
    print('SJoint found:')
    print(np.exp(logSJoint))
    logSMarginalToReshape = scipy.special.logsumexp(logSJoint, axis=0)
    logSmarginal = (logSMarginalToReshape).reshape(1, logSMarginalToReshape.size)
    logSPost = logSJoint - logSmarginal
    return logSPost


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
    dataSetsTrainSplitted = splitDataSets(DTR, LTR)
    dataSetsTrainSplittedTransposed = trasposeDataSetSplitted(dataSetsTrainSplitted)

    muAndCovDivided = computeMuAndCovForClass(dataSetsTrainSplittedTransposed)
    #first: compute the likelihoods:
    logscoreMatrix = computeLogLikelihoodForEachClassWithoutClasses(DVAL, muAndCovDivided)
    logPosterior = computeLogPosterior(logscoreMatrix, np.ones(3)/3.)
    #now compare: matrices are identical!
    # print('logPosterior found by me:')
    # print(logPosterior)
    # print('logPosterior solution:')
    # print(np.load('logPosterior_MVG.npy'))

    #phase 3 of lab:
    


