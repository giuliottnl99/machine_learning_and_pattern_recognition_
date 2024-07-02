import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn.datasets

def vcol(x): # Same as in pca script
    return x.reshape((x.size, 1))

def vrow(x): # Same as in pca script
    return np.matrix(x.reshape((1, x.size)))

def vcol_arr(x):
    return np.array(x.reshape((x.size, 1)))

def vrow_arr(x):
    return np.array(x.reshape((1, x.size)))

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

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
#Note: theoretically you compute Vh.T (getting columns instead of rows) and then retranspose it to apply PCA, but it makes not sense!
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

def plotScatterUsingScale(arrayX, arrayY, title='', c="red", xLabel='caratt 1', yLabel='caratt 2'):
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xscale('log', base=10)
    plt.title(title)
    plt.scatter(np.ravel(arrayX).astype(float), np.ravel(arrayY).astype(float), color=c)


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
def doBinaryClassification_PCA_LDA(DBinary, LBinary, toPlot=False, toPrint=True, chosenMethod='LDA', dimensionsPCA=2, LValueTrue=1, LValueFalse=0, threshold=None, reduceDataset=False):
    #first of all, remove data where the data label is ==0 keeping 1 and 2 only
    #divide training and test set:
    (DT, LT), (DV, LV) = divideSamplesRandomly(DBinary, LBinary)
    if reduceDataset:
        DT = DT[:, ::50]
        LT = LT[::50]
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
        plt.legend()
        #then plot validation sample reduced:
        plt.figure()
        plt.title("Validation set")
        plotHist(projDV[0, LV==LValueTrue], c="green", i=5, l="True")
        plotHist(projDV[0, LV==LValueFalse], c="red", i=5, l="False")
        plt.legend()

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

def computePosterior(logScoreMatrix):
    logPosterior = computeLogPosterior(logScoreMatrix)
    return np.exp(logPosterior)

def computeLogPosterior(logScoreMatrix, prior=None):
    if prior==None:
        prior = np.ones(logScoreMatrix.shape[0]) / logScoreMatrix.shape[0]
    V_priorLog = np.log(prior)
    logSJoint = logScoreMatrix + vcol(V_priorLog)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    return logSJoint - logSMarginal

def computePrevisionArray(posteriorProbMatrix, classNamesArr):
    #get classes for each
    #for each sample get the maximum probability
    previsionArray = []
    if classNamesArr == None:
        classNamesArr = []
        for j in range(posteriorProbMatrix.shape[1]):
           classNamesArr.append(j) 

    for j in range(posteriorProbMatrix.shape[1]):
        bestClassIndex = np.argmax(posteriorProbMatrix[:, j])
        bestClass = classNamesArr[bestClassIndex]
        previsionArray.append(bestClass)
    return previsionArray

def computeAccuracy(previsionArray, L):
    validElements = [ L[i] for i in range(len(L)) if L[i]==previsionArray[i]]
    return len(validElements) / len(L) 

#Log score matrix is usually known as LLR
def applyMVGToComputeLogScoreMatrix(D, L, chosenCase='ML'):
    (DT, LT), (DV, LV) = divideSamplesRandomly(D, L)
    muAndCovDivided = computeMuAndCovForClass(DT, LT, chosenCase=chosenCase)
    #first: compute the likelihoods:
    logScoreMatrix, classNamesArr = computeLogScoreMatrix(DV, LV, muAndCovDivided)
    return logScoreMatrix

#LLR = logPosteriorMatrix binarized!
def applyMVGToComputeLLR_Binary(D, L, chosenCase='ML'):
    (DT, LT), (DV, LV) = divideSamplesRandomly(D, L)
    muAndCovDivided = computeMuAndCovForClass(DT, LT, chosenCase=chosenCase)
    #first: compute the likelihoods:
    logScoreMatrix, classNamesArr = computeLogScoreMatrix(DV, LV, muAndCovDivided)
    logPosteriorProbMatrix = computeLogPosterior(logScoreMatrix)
    LLR = logPosteriorProbMatrix[0, :] - logPosteriorProbMatrix[1, :]
    return LLR


def createAndApplyMVG(D, L, chosenCase='ML'):
    (DT, LT), (DV, LV) = divideSamplesRandomly(D, L)

    muAndCovDivided = computeMuAndCovForClass(DT, LT, chosenCase=chosenCase)
    #first: compute the likelihoods:
    logScoreMatrix, classNamesArr = computeLogScoreMatrix(DV, LV, muAndCovDivided)
    posteriorProbMatrix = computePosterior(logScoreMatrix)
    previsionArray = computePrevisionArray(posteriorProbMatrix, classNamesArr)
    accuracy = computeAccuracy(previsionArray, LV)
    return accuracy, previsionArray

#it is the same as calling createAndApplyMVG but works only for binarys
def computeAccuracyUsingBinaryDivision_MVG(DBinary, LBinary, labelTrue, labelFalse, chosenCase='ML', reduceDataset=False):
    (DT, LT), (DV, LV) = divideSamplesRandomly(DBinary, LBinary)
    if reduceDataset:
        DT = DT[:, ::50]
        LT = LT[::50]

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

#previsionArray must be passed as a 1-d array with same values as predictions should!
def computeConfusionMatrix(previsionArray, LVAL):
    confusionMatrix = np.zeros((len(np.unique(LVAL)), len(np.unique(LVAL))))
    previsionArrayRaveled = np.ravel(previsionArray)
    mapLabelsToIndex = {} #contains className -> Associated index
    nClass=0
    for className in np.unique(LVAL):
        mapLabelsToIndex[className] = nClass
        nClass += 1
    for i in range(len(LVAL)):
        predictedLabel = previsionArrayRaveled[i]
        actualLabel = LVAL[i]
        indexPredLabel = mapLabelsToIndex[predictedLabel]
        indexActualLabel = mapLabelsToIndex[actualLabel]
        confusionMatrix[indexPredLabel, indexActualLabel] += 1
    return confusionMatrix

def computeOptimalThresholdUsingCosts_Binary(prior, Cfn, Cfp):
    return -np.log( (prior * Cfn) / ((1 - prior) * Cfp) )

#remember that conventionally true case is on the left of the threshold and false is on the right!
def computePrevisionMatrix_Binary(llrArray, threshold, trueValue=None, falseValue=None):
    previsionArrayBinary = llrArray < threshold
    previsionArrayBinaryWithValues = None
    if(trueValue!=None or falseValue!=None):
        previsionArrayBinaryWithValues = np.zeros(len(previsionArrayBinary))
        previsionArrayBinaryWithValues[previsionArrayBinary==True] = trueValue
        previsionArrayBinaryWithValues[previsionArrayBinary==False] = falseValue
    else:
        previsionArrayBinaryWithValues = previsionArrayBinary
    return previsionArrayBinaryWithValues
    
#llrArray should be an array with 1-d only -> That's important!
def computePrevisionMatrixUsingCosts_Binary(llrArray, prior, Cfn, Cfp, trueValue=None, falseValue=None):
    llrArrayRavel = np.ravel(llrArray)
    threshold = computeOptimalThresholdUsingCosts_Binary(prior, Cfn, Cfn)
    return computePrevisionMatrix_Binary(llrArrayRavel, threshold, trueValue=trueValue,falseValue=falseValue)

def computeDCFBayesError_Binary(confusionMatrix, prior: float, costFalseNeg: float, costFalsePos: float, normalize=True):
    M = confusionMatrix
    probFalseNeg = M[0,1] / (M[0,1] + M[1,1])
    probFalsePos = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * costFalseNeg * probFalseNeg + (1-prior) * costFalsePos * probFalsePos
    if normalize:
        return bayesError / np.minimum(prior * costFalseNeg, (1-prior)*costFalsePos)
    return bayesError

#compute and index of (Detection cost function) -> if >1 the system is not good at all!
#in the priors_array pass first 1-p and then p
def computeDCFBayesError_Multiclass(confusionMatrix, priors_array, costsMatrix, normalize=True):
    #convert in case you need:
    priors_array = np.array(priors_array)
    errorsArray = np.array(confusionMatrix / vrow(confusionMatrix.sum(0)))
    costsMatrix = np.array(costsMatrix)
    bayesError = ((np.multiply(errorsArray, costsMatrix)).sum(0)  * priors_array.ravel()).sum()
    if normalize:
        return bayesError / np.min(costsMatrix @ vcol(priors_array))
    return bayesError

#try to compute DCF for every threshold until it finds min threshold and min DCF. 
def compute_minDCF_binary_slow(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):    
    thresholds = np.concatenate([np.array([-np.inf]), np.ravel(llr), np.array([np.inf])])
    dcfMin = None
    dcfTh = None
    for th in thresholds:
        predictedLabels = np.int32(llr > th)
        confusionMatrix = computeConfusionMatrix(predictedLabels, classLabels)
        dcf = computeDCFBayesError_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=True)
        if dcfMin is None or dcf < dcfMin:
            dcfMin = dcf
            dcfTh = th
    if returnThreshold:
        return dcfMin, dcfTh
    else:
        return dcfMin

#hard copied by solution, but I need fast version for the lab:
def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):

    llrSorter = np.argsort(llr)
    llrSorted = []
    # classLabelsSorted = []
    if llrSorter.shape[0]==1:
        llrSorted = vrow_arr(llr[0, llrSorter]).ravel() 
    else:
        llrSorted = llr[llrSorter] # We sort the llrs
    classLabelsSorted = classLabels[llrSorter].ravel() # we sort the labels so that they are aligned to the llrs

    Pfp = []
    Pfn = []
    
    nTrue = (classLabelsSorted==1).sum()
    nFalse = (classLabelsSorted==0).sum()
    nFalseNegative = 0 # With the left-most theshold all samples are assigned to class 1
    nFalsePositive = nFalse
    
    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)
    
    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    #The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
    #Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    #Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    llrSorted = np.concatenate([-np.array([np.inf]), llrSorted])

    # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx+1] != llrSorted[idx]: # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])
            
    return np.array(PfnOut), np.array(PfpOut), np.array(thresholdsOut) # we return also the corresponding thresholds


def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):

    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / np.minimum(prior * Cfn, (1-prior)*Cfp) # We exploit broadcasting to compute all DCFs for all thresholds
    idx = np.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]


#note: L must contains only labels 0 and 1!!!!
#if pT==None we have a normal log reg, otherwise a weighted log ref
def trainLogRegBinary(DTR, LTR, lambd, pT=None, toPrint=False):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    epsTrue = 1
    epsFalse = 1
    if pT!=None:
        epsTrue = pT / (ZTR>0).sum()
        epsFalse = (1-pT) / (ZTR<0).sum()


    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = np.dot(vcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)
        #if we are in normal case, this won't affect. Otherwise multiply for esp
        loss[ZTR>0] *= epsTrue
        loss[ZTR<0] *= epsFalse
        
        G = -ZTR / (1.0 + np.exp(ZTR * s))
        G[ZTR > 0] *= epsTrue
        G[ZTR < 0] *= epsFalse
        GW = (vrow_arr(G) * DTR).mean(1) + lambd * w.ravel()
        #but if pT !=None GW must be redefined
        if pT!=None:
            GW = (vrow_arr(G) * DTR).sum(1) + lambd * w.ravel()

        Gb = G.mean()
        if pT!=None:
            Gb = G.sum()
            return loss.sum() + lambd / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])
        return loss.mean() + lambd / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = np.zeros(DTR.shape[0]+1))[0]
    if toPrint and pT==None:
        print ("Log-reg - lambda = %e - J*(w, b) = %e" % (lambd, logreg_obj_with_grad(vf)[0]))
    elif toPrint:
        print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, lambd, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

def computeQuadraticXforLogReg(dataSet):
    resultMatrix = np.zeros((dataSet.shape[0]**2 + dataSet.shape[0], dataSet.shape[1]))
    for j in range(dataSet.shape[1]): #j is a sample, so a column
        x = dataSet[:, j]
        productColsXAsMatrix = vcol_arr(x) @ vcol_arr(x).T
        productColsXAsArray = productColsXAsMatrix.ravel()
        #add productColsXAsArray
        for i in range(len(productColsXAsArray)): #i is the row
            resultMatrix[i, j] = productColsXAsArray[i]
        for i in range(len(x)): #i is the row
            resultMatrix[i+len(productColsXAsArray), j] = x[i]
    return np.array(resultMatrix)

def computeQuadraticXforSMV(dataSet):
    resultMatrix = np.zeros((dataSet.shape[0]**2 + dataSet.shape[0] +1, dataSet.shape[1]))
    for j in range(dataSet.shape[1]): #j is a sample, so a column
        x = dataSet[:, j]
        productColsXAsMatrix = vcol_arr(x) @ vcol_arr(x).T
        productColsXAsArray = productColsXAsMatrix.ravel()
        #add productColsXAsArray
        for i in range(len(productColsXAsArray)): #i is the row
            resultMatrix[i, j] = productColsXAsArray[i]
        for i in range(len(x)): #i is the row
            resultMatrix[i+len(productColsXAsArray), j] = (2**0.5) * x[i]
        resultMatrix[len(resultMatrix)-1, j] = 1 
    return np.array(resultMatrix)

    

def train_dual_SVM_linear(DTR, LTR, C, K = 1):
    
    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    DTR_EXT = np.vstack([DTR, np.ones((1,DTR.shape[1])) * K])
    H = np.dot(DTR_EXT.T, DTR_EXT) * vcol_arr(ZTR) * vrow_arr(ZTR)

    # Dual objective with gradient
    def fOpt(alpha): 
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)
    
    # Primal loss
    def primalLoss(w_hat):
        S = (vrow_arr(w_hat) @ DTR_EXT).ravel()
        return 0.5 * np.linalg.norm(w_hat)**2 + C * np.maximum(0, 1 - ZTR * S).sum()

    # Compute primal solution for extended data matrix
    w_hat = (vrow_arr(alphaStar) * vrow_arr(ZTR) * DTR_EXT).sum(1)
    
    # Extract w and b - alternatively, we could construct the extended matrix for the samples to score and use directly v
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K # b must be rescaled in case K != 1, since we want to compute w'x + b * K

    primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0]
    # print ('SVM - C %e - K %e - primal loss %e - dual loss %e - duality gap %e' % (C, K, primalLoss, dualLoss, primalLoss - dualLoss))

    
    return w, b, primalLoss, dualLoss

def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0):

    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    K = kernelFunc(DTR, DTR) + eps
    H = vcol_arr(ZTR) * vrow_arr(ZTR) * K

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)

    # print ('SVM (kernel) - C %e - dual loss %e' % (C, -fOpt(alphaStar)[0]))

    # Function to compute the scores for samples in DTE
    def fScore(DTE):
        
        K = kernelFunc(DTR, DTE) + eps
        H = vcol_arr(alphaStar) * vcol_arr(ZTR) * np.array(K)
        return H.sum(0)

    return fScore, -fOpt(alphaStar)[0] # we directly return the function to score a matrix of test samples

def polyKernel(degree, c):
    
    def polyKernelFunc(D1, D2):
        return ((D1.T @ D2) + c) ** degree
    return polyKernelFunc

def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol_arr(D1Norms) + vrow_arr(D2Norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-gamma * Z)

    return rbfKernelFunc



# if __name__ == "__main__":
#     loadFileAndSave('trainData.txt', 'D_exam_train.npy', 'L_exam_train.npy')

