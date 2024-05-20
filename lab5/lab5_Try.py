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

def computeLogLikelihoodForEachClassWithoutClasses(AllSamples, muAndCovDivided, doPrint=False):
    scoreMatrix = []

    for i in range(len(muAndCovDivided)):
        mu, cov = muAndCovDivided[i][0], muAndCovDivided[i][1] #here it breaks!
        scoreMatrix.append(logpdf_GAU_ND(AllSamples, mu, cov))
    if doPrint:
        print('scoreMatrix:')
        print(scoreMatrix)
    return scoreMatrix


def computeLogPosterior(logScoreMatrix, v_prior, toPrint=False):
    #note: classData is a matrix!
    v_priorLog = np.log(v_prior)
    logSJoint = logScoreMatrix + v_priorLog.reshape(v_priorLog.size, 1)
    logSMarginalToReshape = scipy.special.logsumexp(logSJoint, axis=0)
    logSmarginal = (logSMarginalToReshape).reshape(1, logSMarginalToReshape.size)
    logSPost = logSJoint - logSmarginal
    if toPrint:
        print('SJoint found:')
        print(np.exp(logSJoint))
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


def computeMuAndCovForClass(dataSetSplitted, chosenCase='default'):
    muAndCovForElement = []
    betweenCovMatrix = np.zeros((dataSetSplitted[0].shape[0], dataSetSplitted[0].shape[0]))
    muForBetweenCovMatrix = []
    totalSizeForBetweenCovMatrix = 0
    for classParameters in dataSetSplitted:
        if chosenCase=='binaryClassification' and classParameters.shape[0]==0:
            continue
        mu, cov = computeMuAndCov(classParameters)
        if chosenCase=='default' or chosenCase=='binaryClassification':
            muAndCovForElement.append([mu, cov])
        elif chosenCase=='naive':
            #maybe shape is a issue (I don't think!)
            muAndCovForElement.append([mu, cov * np.eye(classParameters.shape[0])])
        elif chosenCase=='tied':
            betweenCovMatrix += cov * classParameters.shape[1]
            muForBetweenCovMatrix.append(mu)
            totalSizeForBetweenCovMatrix += classParameters.shape[1]

    if chosenCase=='tied':
        betweenCovMatrix = betweenCovMatrix / totalSizeForBetweenCovMatrix
        for mu in muForBetweenCovMatrix:
            #mu is always the same but, in order to make it easier, I repeat n times
            muAndCovForElement.append([mu, betweenCovMatrix])
    return muAndCovForElement


def proofOfNine(posteriorProbMatrix):
    print('posterior matrix:')
    print(posteriorProbMatrix)
    #sum of each element:
    sumOfPosteriorRows = []
    for row in posteriorProbMatrix: 
        if (len(sumOfPosteriorRows)==0):
            sumOfPosteriorRows.append(row)
        else:
            sumOfPosteriorRows += row
    print('prove of nine: should be a matrix of all to one:') 
    print(sumOfPosteriorRows)


def computeAccuracy(posteriorProbMatrix, labels):
    prevalentValueArray = []
    for j in range(posteriorProbMatrix.shape[1]):
        maxValueForElement = 0
        winnerClass = None
        for i in range(posteriorProbMatrix.shape[0]):
            if(maxValueForElement<=posteriorProbMatrix[i][j]):
                maxValueForElement, winnerClass = posteriorProbMatrix[i][j], i
        prevalentValueArray.append(winnerClass)
    #then compute accuracy:
    accuracyArray = prevalentValueArray == labels
    return np.count_nonzero(accuracyArray) / len(prevalentValueArray) * 100


if __name__ == '__main__':
    D, L = load_iris()
    (DTR, LTR), (DVAL, LVAL) = split_DB_2to1(D, L)

    #first try: split dataset based on whole matrix;
    dataSetsTrainSplitted = splitDataSets(DTR, LTR)
    dataSetsTrainSplittedTransposed = trasposeDataSetSplitted(dataSetsTrainSplitted)

    muAndCovDividedML = computeMuAndCovForClass(dataSetsTrainSplittedTransposed)
    #first: compute the likelihoods:
    logScoreMatrixML = computeLogLikelihoodForEachClassWithoutClasses(DVAL, muAndCovDividedML)
    logPosteriorML = computeLogPosterior(logScoreMatrixML, np.ones(3)/3.)
    posteriorProbMatrixML = np.exp(logPosteriorML)
    # proofOfNine(logPosterior)
    accuracyML = computeAccuracy(posteriorProbMatrixML, LVAL)
    print('accuracy for ML solution:')
    print(accuracyML)

    #phase 2.2 of lab: naive Bayes Gaussian Classifier:
    muAndCovDividedNaive = computeMuAndCovForClass(dataSetsTrainSplittedTransposed, chosenCase='naive')
    logScoreMatrixNaive = computeLogLikelihoodForEachClassWithoutClasses(DVAL, muAndCovDividedNaive)
    logPosteriorNaive = computeLogPosterior(logScoreMatrixNaive, np.ones(3)/3.)
    posteriorProbMatrixNaive = np.exp(logPosteriorNaive)
    accuracyNaive = computeAccuracy(posteriorProbMatrixNaive, LVAL)
    print('accuracy for naive:')
    print(accuracyNaive)

    #now compute accuracy for tied:
    muAndCovDividedTied = computeMuAndCovForClass(dataSetsTrainSplittedTransposed, chosenCase='tied')
    print('cov tied:')
    print(muAndCovDividedTied[0][1]) #same for each class!
    logScoreMatrixTied = computeLogLikelihoodForEachClassWithoutClasses(DVAL, muAndCovDividedTied)
    logPosteriorTied = computeLogPosterior(logScoreMatrixTied, np.ones(3)/3.)
    posteriorProbMatrixTied = np.exp(logPosteriorTied)
    accuracyTied = computeAccuracy(posteriorProbMatrixTied, LVAL)
    print('accuracy for Tied:')
    print(accuracyTied)


    #class problem, ML model
    #logPdf of class 2 - logPdf of class 1 (pdfGau of class 1 / pdfGau of class 2)
    DReduced = D[:, L!=0]
    LReduced = L[L!=0]
    (DTRReduced, LTRReduced), (DVALReduced, LVALReduced) = split_DB_2to1(DReduced, LReduced)
    dataSetsTrainSplitted = splitDataSets(DTRReduced, LTRReduced)
    dataSetsTrainSplittedTransposed = trasposeDataSetSplitted(dataSetsTrainSplitted)

    muAndCovDividedML = computeMuAndCovForClass(dataSetsTrainSplittedTransposed, chosenCase='binaryClassification')
    logScoreMatrixML = computeLogLikelihoodForEachClassWithoutClasses(DVALReduced, muAndCovDividedML)
    LLR = logScoreMatrixML[1] - logScoreMatrixML[0]
    oneAnd2SamplesArrays = np.zeros(DVALReduced.shape[1], dtype=np.int32)
    oneAnd2SamplesArrays[LLR>=0] = 2
    oneAnd2SamplesArrays[LLR<0] = 1
    indexesOfMatchedElements = np.where(oneAnd2SamplesArrays == LVALReduced)[0]
    arrayOfMatchedElements = LVALReduced[indexesOfMatchedElements]
    print('accuracy of class-problem division: ')
    print(len(arrayOfMatchedElements) / len(LVALReduced))









