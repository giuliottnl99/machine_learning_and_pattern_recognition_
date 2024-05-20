import numpy as np
import matplotlib.pyplot as plt

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
def getPCAReducingMatrix(D, L, dim=2):
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

