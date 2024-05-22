import numpy as np
import matplotlib.pyplot as plt
import scipy

def vcol(x): # Same as in pca script
    return x.reshape((x.size, 1))

def vrow(x): # Same as in pca script
    return np.matrix(x.reshape((1, x.size)))

#given D with rows = characteristic and col = sample, get the cov matrix
def computeCovMatrix(D, L):
    muGlobal = np.matrix(D).mean(1)
    diff = D - muGlobal
    C = ( diff @ diff.T ) / D.shape[1]
    return C

#computes an array in the form classLabel -> matrixAssociated
#dim = 3 [classLabel][characteristic][valueOfTheCharacteristic]
def divideByClass(D, L):
    classesMatricesArray = {}
    for label in np.unique(L):
        classesMatricesArray[label] = np.matrix(D[:, L==label])
    return classesMatricesArray

def divideSamplesRandomly(allSamples, L, seed=0):
    np.random.seed(seed)
    permutedSequence =  np.random.permutation(allSamples.shape[1])
    totalSize = permutedSequence.size
    nTrain = int(totalSize*2/3)
    idxTrain = permutedSequence[0:nTrain]
    idxTest = permutedSequence[nTrain:]

    dataTrain = allSamples[:, idxTrain]
    dataValidation = allSamples[:, idxTest]
    labelTrain = L[idxTrain]
    labelValidation = L[idxTest]

    return(dataTrain, labelTrain), (dataValidation , labelValidation)


def compute_Sb_Sw(D, L):
    classesMatrices = divideByClass(D, L)
    muGlobal = np.matrix(D).mean(1)
    Sb = 0
    Sw = 0
    for classMatrix in classesMatrices.values():
        muClass = classMatrix.mean(1)
        muBetweenDiff = muClass - muGlobal
        Sb += ( muBetweenDiff @ muBetweenDiff.T ) * classMatrix.shape[1]
        muWithinDiff = classMatrix - muClass
        Sw += muWithinDiff @ muWithinDiff.T
    return Sb / D.shape[1], Sw / D.shape[1]

#get the "reducing matrix" applying LDA. Just multiply reducingMatrix @ x for the array x to obtain 
def computeLDA_ReducingMatrix(D, L, dim=2):
    Sb, Sw = compute_Sb_Sw(D, L)
    U, s, _ = np.linalg.svd(Sw)
    P1 = U @ (np.diag(1.0 / s**0.5)) @ U.T
    Sb2 = P1 @ Sb @ P1.T
    P2Complete, eigenVectorsComplete, _ = np.linalg.svd(Sb2)
    P2 = P2Complete[:, 0:dim]
    return P2.T @ P1

#get reducing matrix applying PCA: just compute: reducingMatrix @ D to obtain reduction
#Note: usually you compute Vh.T (getting columns instead of rows) and then retranspose it to apply PCA, but it makes not sense!
def computePCA_ReducingMatrix(D, L, dim=2):
    C = computeCovMatrix(D, L)
    U, s, Vh = np.linalg.svd(C)
    # reducingMatrix = Vh.T[:, 0:dim]
    reducingMatrix = Vh[0:dim, :]
    return reducingMatrix



def plotHist(array, c="red", l="", i=5):
    plt.hist(np.ravel(array), bins=i, density=True, ec='black', alpha = 0.4, color=c, label=l)

def plotScatter(arrayX, arrayY, l='', c="red"):
    plt.xlabel('caratt 1')
    plt.ylabel('caratt 2')
    plt.scatter(np.ravel(arrayX).astype(float), np.ravel(arrayY).astype(float), label=l, color=c)

#I will load the file as an array in D_and_L.npy
def loadFile(path):
    D = []
    L = []
    with open(path, 'r') as f:
        for line in f:
               elements = line.split(" , ")
               D.append(elements[0:-1])
               L.append(elements[-1])
    return np.matrix(D).T, np.ravel(L)
               
def loadFileAndSave(pathFrom, pathTo1, pathTo2):
     D, L = loadFile(pathFrom)
     np.save(pathTo1, D)
     np.save(pathTo2, L)

#want true at the right and false at the left!
def computeAndRevertProj_DT_DV(DT, DV, LT, reducingMatrix, lblTrue, lblFalse):
    projDT = reducingMatrix @ DT
    if projDT[0, LT==lblFalse].mean(1)[0, 0] > projDT[0, LT==lblTrue].mean(1)[0, 0]:    
        reducingMatrix = - reducingMatrix
    projDT = reducingMatrix @ DT
    projDV = reducingMatrix @ DV
    return projDT, projDV


#False on the left, True on the right is the config chosen
def doBinaryClassification_PCA_LDA(DBinary, LBinary, toPlot=False, toPrint=True, chosenMethod='LDA', dimensionsPCA=2, LValueTrue=1, LValueFalse=0, threshold=None):
    #first of all, remove data where the data label is ==0 keeping 1 and 2 only
    #divide training and test set:
    (DT, LT), (DV, LV) = divideSamplesRandomly(DBinary, LBinary)
    #doLDA
    reducingMatrix = None
    projDT = None
    projDV = None
    if chosenMethod == 'LDA':
        reducingMatrix = computeLDA_ReducingMatrix(DT, LT, dim=1)
    if chosenMethod == 'PCA':
        reducingMatrix = computePCA_ReducingMatrix(DT, LT, dim=1)

    if chosenMethod=='bothPCA_LDA':
        reducingMatrixPCA = computePCA_ReducingMatrix(DT, LT, dim=dimensionsPCA)
        projDT, projDV = computeAndRevertProj_DT_DV(DT, DV, LT, reducingMatrixPCA, LValueTrue, LValueFalse)
        reducingMatrixLDA = computeLDA_ReducingMatrix(projDT, LT, dim=1)
        projDT, projDV = computeAndRevertProj_DT_DV(projDT, projDV, LT, reducingMatrixLDA, LValueTrue, LValueFalse)
    else:
        projDT, projDV = computeAndRevertProj_DT_DV(DT, DV, LT, reducingMatrix, LValueTrue, LValueFalse)

    #check if virginica class samples are on the right of the Versicolor samples:
    if threshold == None:
        threshold = ( projDT[0, LT==LValueTrue].mean() + projDT[0, LT==LValueFalse].mean() ) / 2.0
    elif threshold=="Median":
        threshold = ( np.median(projDT[0, LT==LValueTrue], axis=1) + np.median(projDT[0, LT==LValueFalse], axis=1) ) / 2.0
    #compute number of times projDV is good
    PV = np.matrix(np.zeros(shape=LV.shape))
    PV[projDV >= threshold] = LValueTrue
    PV[projDV < threshold] = LValueFalse
    matrixValidSamples = [PV[0, i] for i in range(len(LV)) if LV[i]==PV[0, i]]
    #expected: 32 out of 34
    if toPrint:
        print("there are %d matches out of %d classes" % (len(matrixValidSamples), len(LV)))

    if toPlot:
        #first plot test sample reduced:
        plt.figure()
        plt.title("Training set")
        plotHist(projDT[0, LT==LValueTrue], c="green", i=5, l="True")
        plotHist(projDT[0, LT==LValueFalse], c="red", i=5, l="False")
        #then plot validation sample reduced:
        plt.figure()
        plt.title("Validation set")
        plotHist(projDV[0, LV==LValueTrue], c="green", i=5, l="True")
        plotHist(projDV[0, LV==LValueFalse], c="red", i=5, l="False")
        
        plt.show()
    return len(matrixValidSamples) / len(LV)

def computeMuAndCovForClass(D, L, chosenCase='ML'):
    muAndCovForClass = {}
    covTied = np.zeros((D.shape[0], D.shape[0]))
    muTied = {}
    for className in np.unique(L):
        classMatrix = D[:, L==className]
        muC = classMatrix.mean(1)
        covC = computeCovMatrix(D[:, L==className], L)
        if chosenCase=='ML':
            muAndCovForClass[className] = [muC, covC]
        if chosenCase == 'naive':
            muAndCovForClass[className] = [muC, np.array(covC) * np.eye(classMatrix.shape[0])] #what does change from ML?
        if chosenCase == 'tied':
            covTied += covC * classMatrix.shape[1]
            muTied[className] = muC
    
    if chosenCase == 'tied':
        for className in np.unique(L):
            muAndCovForClass[className] = [muTied[className], covTied/D.shape[1]]
    return muAndCovForClass

#try transformed version too!    
def logpdf_GAU_ND(x, mu, C):
    P = np.linalg.inv(C)
    # return -0.5*x.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x-mu).T @ (P @ (x-mu))).sum(0)
    return -0.5*x.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * (np.array(x-vcol(mu)) * (np.array(P) @ np.array(x-vcol(mu)))).sum(0)    


#return logScoreMatrix. 
def computeLogScoreMatrix(D, L, muAndCovDividedForClass):
    logScoreMatrix = []
    classNamesArr = []
    for className in np.unique(L):
       mu, cov =  muAndCovDividedForClass[className][0], muAndCovDividedForClass[className][1]
       logScoreMatrix.append(logpdf_GAU_ND(D, mu, cov))
       classNamesArr.append(className) 
    return np.matrix(logScoreMatrix), classNamesArr

def computePosterior(D, L, logScoreMatrix):
    logPosterior = computeLogPosterior(D, L, logScoreMatrix)
    return np.exp(logPosterior)

def computeLogPosterior(D, L, logScoreMatrix):
    V_priorLog = np.log(np.ones(logScoreMatrix.shape[0]) / logScoreMatrix.shape[0])
    logSJoint = logScoreMatrix + vcol(V_priorLog)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    return logSJoint - logSMarginal

def computePrevisionArray(D, L, posteriorProbMatrix, classNamesArr):
    #get classes for each
    #for each sample get the maximum probability
    previsionArray = []
    for j in range(posteriorProbMatrix.shape[1]):
        bestClassIndex = np.argmax(posteriorProbMatrix[:, j])
        bestClass = classNamesArr[bestClassIndex]
        previsionArray.append(bestClass)
    return previsionArray

def computeAccuracy(previsionArray, L):
    validElements = [ L[i] for i in range(len(L)) if L[i]==previsionArray[i]]
    return len(validElements) / len(L) 

def createAndApplyMVG(D, L, chosenCase='ML'):
    (DT, LT), (DV, LV) = divideSamplesRandomly(D, L)

    muAndCovDivided = computeMuAndCovForClass(DT, LT, chosenCase=chosenCase)
    #first: compute the likelihoods:
    logScoreMatrix, classNamesArr = computeLogScoreMatrix(DV, LV, muAndCovDivided)
    posteriorProbMatrix = computePosterior(DV, LV, logScoreMatrix)
    previsionArray = computePrevisionArray(DV, LV, posteriorProbMatrix, classNamesArr)
    accuracy = computeAccuracy(previsionArray, LV)
    return accuracy

#it is the same as calling createAndApplyMVG
def computeAccuracyUsingBinaryDivision_MVG(DBinary, LBinary, labelTrue, labelFalse, chosenCase='ML'):
    (DT, LT), (DV, LV) = divideSamplesRandomly(DBinary, LBinary)

    muAndCovDivided = computeMuAndCovForClass(DT, LT, chosenCase=chosenCase)
    logScoreMatrix, classNamesArr = computeLogScoreMatrix(DV, LV, muAndCovDivided)
    LLR = vcol(logScoreMatrix[1] - logScoreMatrix[0])
    previsionArr = np.zeros((DV.shape[1], 1), dtype=np.int32)
    previsionArr[LLR>=0] = labelTrue 
    previsionArr[LLR<0] = labelFalse
    arrayMatches = [ previsionArr[i] for i in range(len(LV)) if previsionArr[i]==LV[i] ]
    return len(arrayMatches) / len(LV)

def computePearsonCorrCoeff(covMatrix):
    corr = covMatrix / ( vcol(np.asarray(covMatrix.diagonal())**0.5) * vrow(np.asarray(covMatrix.diagonal())**0.5) )
    return corr


# if __name__ == "__main__":
#     loadFileAndSave('trainData.txt', 'D_exam_train.npy', 'L_exam_train.npy') 