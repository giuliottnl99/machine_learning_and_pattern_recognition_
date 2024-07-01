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
        w, b = ut.trainLogRegBinary(DT, LT, lamb)
        logPosterior = w.T @ DV + b
        #must use empirical! -> % of true samples! CORRECT???
        pEmpirical = DV[:, LV==0].shape[1] / DV.shape[1]
        ssLLR = logPosterior - np.log(pEmpirical / (1-pEmpirical))
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
    plt.title("linear case, full dataset")
    plt.xlabel("lambda")
    plt.ylabel("DCF")
    plt.xscale('log', base=10)
    plt.scatter(np.ravel(arrayXLambda).astype(float), np.ravel(arrayYActual).astype(float), color="red", label="actual DCF")
    plt.scatter(np.ravel(arrayXLambda).astype(float), np.ravel(arrayYMin).astype(float), color="blue", label="min DCF")
    plt.legend()

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
        w, b = ut.trainLogRegBinary(DTRed, LTRed, lamb)  
        logPosterior = w.T @ DV + b
        #must use empirical! -> % of true samples! 
        pEmpirical = DV[:, LV==0].shape[1] / DV.shape[1]
        ssLLR = logPosterior - np.log(pEmpirical / (1-pEmpirical)) 
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
    plt.title("linear case, reduced dataset")
    plt.xlabel("lambda")
    plt.ylabel("DCF")
    plt.xscale('log', base=10)
    plt.scatter(np.ravel(arrayXLambda).astype(float), np.ravel(arrayYActual).astype(float), color="red", label="actual DCF")
    plt.scatter(np.ravel(arrayXLambda).astype(float), np.ravel(arrayYMin).astype(float), color="blue", label="min DCF")
    plt.legend()

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
        w, b = ut.trainLogRegBinary(DT, LT, lamb, pT=pT)
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
    plt.title("weighted case, full dataset")
    plt.xlabel("lambda")
    plt.ylabel("DCF")
    plt.xscale('log', base=10)
    plt.scatter(np.ravel(arrayXLambda).astype(float), np.ravel(arrayYActual).astype(float), color="red", label="actual DCF")
    plt.scatter(np.ravel(arrayXLambda).astype(float), np.ravel(arrayYMin).astype(float), color="blue", label="min DCF")
    plt.legend()

    print("DCFs mins for each lambda, full dataset, weighted: ")
    print(arrayYMin)
    print("DCFs actual for each lambda, full dataset, weighted: ")
    print(arrayYActual)

    print("lambdas")
    print(arrayXLambda)

    #QUADRATIC logistic regression
    arrayYActual = []
    arrayYMin = []
    arrayXLambda = []

    DTQuadratic = ut.computeQuadraticX(DT)
    DVQuadratic = ut.computeQuadraticX(DV)

    for lamb in np.logspace(-4, 2, 13):
        #first try with pT=0.1
        pT=0.1 #this is prior without weighted system
        #we use pT but not for log red
        w, b = ut.trainLogRegBinary(DTQuadratic, LT, lamb)
        logPosterior = w.T @ DVQuadratic + b
        #can I just use DVQuadratic??
        pEmpirical = DVQuadratic[:, LV==0].shape[1] / DVQuadratic.shape[1]
        ssLLR = logPosterior - np.log(pEmpirical / (1-pEmpirical))
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
    plt.title("quadratic case, full dataset")
    plt.xlabel("lambda")
    plt.ylabel("DCF")
    plt.xscale('log', base=10)
    plt.scatter(np.ravel(arrayXLambda).astype(float), np.ravel(arrayYActual).astype(float), color="red", label="actual DCF")
    plt.scatter(np.ravel(arrayXLambda).astype(float), np.ravel(arrayYMin).astype(float), color="blue", label="min DCF")
    plt.legend()

    print("DCFs mins for each lambda, full dataset, quadratic: ")
    print(arrayYMin)
    print("DCFs actual for each lambda, full dataset, quadratic: ")
    print(arrayYActual)

    #centered linear model
    arrayYActual = []
    arrayYMin = []
    arrayXLambda = []
    DTCentered = DT - np.mean(DT, axis=0)
    DVCentered = DV - np.mean(DV, axis=0)
    for lamb in np.logspace(-4, 2, 13):
        #first try with pT=0.1
        pT=0.1 #this is prior without weighted system
        #we use pT but not for log red
        w, b = ut.trainLogRegBinary(DTCentered, LT, lamb)
        logPosterior = w.T @ DVCentered + b
        pEmpirical = DVCentered[:, LV==0].shape[1] / DVCentered.shape[1]
        ssLLR = logPosterior - np.log(pEmpirical / (1-pEmpirical))
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
    plt.title("centered case, full dataset")
    plt.xlabel("lambda")
    plt.ylabel("DCF")
    plt.xscale('log', base=10)
    plt.scatter(np.ravel(arrayXLambda).astype(float), np.ravel(arrayYActual).astype(float), color="red", label="actual DCF")
    plt.scatter(np.ravel(arrayXLambda).astype(float), np.ravel(arrayYMin).astype(float), color="blue", label="min DCF")
    plt.legend()

    print("DCFs mins for each lambda, full dataset, centered: ")
    print(arrayYMin)
    print("DCFs actual for each lambda, full dataset, centered: ")
    print(arrayYActual)
    plt.show()
    stopDebug = ""






