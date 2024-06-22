import scipy
import numpy as np
import sys
sys.path.append('C:\\Users\\giuli\\OneDrive\\Documenti\\poliTo 2024\\machine_learning_and_pattern_recognition\\general_utils\\')
import utils as ut
import matplotlib.pyplot as plt


if __name__ == "__main__":
    D, L = np.load("general_utils/D_exam_train.npy"), np.load("general_utils/L_exam_train.npy")
    (DT, LT), (DV, LV) = ut.divideSamplesRandomly(D, L)
    #get items
    arrayYActual = []
    arrayYMin = []
    arrayXLambda = []

    for lamb in np.logspace(-4, 2, 13):
        #first try with pT=0.1
        pT=0.1 #this is prior without weighted system
        #we use pT but not for log red
        # w, b = ut.trainLogRegBinary(DT, LT, lamb, pT=pT) 
        w, b = ut.trainLogRegBinary(DT, LT, lamb)
        logPosterior = w.T @ DV + b
        ssLLR = logPosterior - np.log(pT / (1-pT))
        th = -np.log((pT * 1.0) / ((1.0 - pT) * 1.0))
        previsionArray = (ssLLR > th) *1
        confusionMatrix = ut.computeConfusionMatrix(previsionArray, LV)
        # print("DCF pT - 0.8  =  %f " % (ut.computeDCFBayesError_Binary(confusionMatrix, pT, 1.0, 1.0)))
        # print("DCF min - pT 0.8 = %f " % (ut.compute_minDCF_binary_slow(ssLLR, LV, pT, 1.0, 1.0)))
        arrayXLambda.append(lamb)
        arrayYMin.append(ut.compute_minDCF_binary_fast(ssLLR, LV, pT, 1.0, 1.0))
        arrayYActual.append(ut.computeDCFBayesError_Binary(confusionMatrix, pT, 1.0, 1.0))
    #and now let's plot:
    plt.figure()
    ut.plotScatterUsingScale(arrayXLambda, arrayYMin, title='Plot lambda vs DCF min', c='red', xLabel='Lambda', yLabel='min DCF')
    plt.figure()
    ut.plotScatterUsingScale(arrayXLambda, arrayYActual, title='Plot lambda vs actual DCF', c='green', xLabel='Lambda', yLabel='actual DCF')
    print("DCFs mins for each lambda, full dataset, not weighted: ")
    print(arrayYMin)
    print("DCFs actual for each lambda, full dataset, not weighted: ")
    print(arrayYActual)

    #part 2: redo using reduced DT:
    arrayYActual = []
    arrayYMin = []
    arrayXLambda = []
    DTRed = DT[:, ::50]
    LTRed = LT[::50]

    for lamb in np.logspace(-4, 2, 13):
        #first try with pT=0.1
        pT=0.1 #this is prior without weighted system
        #first of all: train with pT 
        # w, b = ut.trainLogRegBinary(DTRed, LTRed, lamb, pT=pT)
        w, b = ut.trainLogRegBinary(DTRed, LTRed, lamb)
        logPosterior = w.T @ DV + b
        ssLLR = logPosterior - np.log(pT / (1-pT))
        th = -np.log((pT * 1.0) / ((1.0 - pT) * 1.0))
        previsionArray = (ssLLR > th) *1
        confusionMatrix = ut.computeConfusionMatrix(previsionArray, LV)
        # print("DCF pT - 0.8  =  %f " % (ut.computeDCFBayesError_Binary(confusionMatrix, pT, 1.0, 1.0)))
        # print("DCF min - pT 0.8 = %f " % (ut.compute_minDCF_binary_slow(ssLLR, LV, pT, 1.0, 1.0)))
        arrayXLambda.append(lamb)
        arrayYMin.append(ut.compute_minDCF_binary_fast(ssLLR, LV, pT, 1.0, 1.0))
        arrayYActual.append(ut.computeDCFBayesError_Binary(confusionMatrix, pT, 1.0, 1.0))
    #and now let's plot:
    plt.figure()
    ut.plotScatterUsingScale(arrayXLambda, arrayYMin, title='Plot lambda vs DCF min with 1/50 samples', c='red', xLabel='Lambda', yLabel='min DCF')
    plt.figure()
    ut.plotScatterUsingScale(arrayXLambda, arrayYActual, title='Plot lambda vs actual DCF with 1/50 samples', c='green', xLabel='Lambda', yLabel='actual DCF')
    
    print("DCFs mins for each lambda, reduced dataset, not weighted: ")
    print(arrayYMin)
    print("DCFs actual for each lambda, reduced dataset, not weighted: ")
    print(arrayYActual)

    #part 3: using prior-weighted-regression:
    arrayYActual = []
    arrayYMin = []
    arrayXLambda = []

    for lamb in np.logspace(-4, 2, 13):
        #first try with pT=0.1
        pT=0.1 #this is prior without weighted system
        #first of all: train with pT 
        w, b = ut.trainLogRegBinary(DTRed, LTRed, lamb, pT=pT)
        logPosterior = w.T @ DV + b
        ssLLR = logPosterior - np.log(pT / (1-pT))
        th = -np.log((pT * 1.0) / ((1.0 - pT) * 1.0))
        previsionArray = (ssLLR > th) *1
        confusionMatrix = ut.computeConfusionMatrix(previsionArray, LV)
        arrayXLambda.append(lamb)
        arrayYMin.append(ut.compute_minDCF_binary_fast(ssLLR, LV, pT, 1.0, 1.0))
        arrayYActual.append(ut.computeDCFBayesError_Binary(confusionMatrix, pT, 1.0, 1.0))
    #and now let's plot:
    plt.figure()
    ut.plotScatterUsingScale(arrayXLambda, arrayYMin, title='Plot lambda vs DCF min using weighted', c='red', xLabel='Lambda', yLabel='min DCF')
    plt.figure()
    ut.plotScatterUsingScale(arrayXLambda, arrayYActual, title='Plot lambda vs actual DCF using weighted', c='green', xLabel='Lambda', yLabel='actual DCF')
    print("DCFs mins for each lambda, full dataset, weighted: ")
    print(arrayYMin)
    print("DCFs actual for each lambda, full dataset, weighted: ")
    print(arrayYActual)

    print("lambdas")
    print(arrayXLambda)
    plt.show()

    remainedGraphDebug = "" 




