import sys
import scipy
import numpy as np
sys.path.append('C:\\Users\\giuli\\OneDrive\\Documenti\\poliTo 2024\\machine_learning_and_pattern_recognition\\general_utils\\')
import utils as ut


def f(x):
    y,z = x
    return (y+3)**2 + np.sin(y) + (z+1)**2

def fprime(x):
    y,z = x
    return np.array([2*(y+3) + np.cos(y), 2 * (z+1)])

if __name__ == "__main__":
    DComplete, LComplete = ut.load_iris()
    D = DComplete[:, LComplete!=0]
    L = LComplete[LComplete!=0]
    L[L==2] = 0
    (DT, LT), (DV, LV) = ut.divideSamplesRandomly(D, L)

    #let's try first with "try" function:
    print (scipy.optimize.fmin_l_bfgs_b(func = f, approx_grad = True, x0 = np.zeros(2)))
    print (scipy.optimize.fmin_l_bfgs_b(func = f, fprime = fprime, x0 = np.zeros(2)))
    
    for lamb in [1e-3, 1e-1, 1.0]:
        #I want L to be 1 or 0
        w, b = ut.trainLogRegBinary(DT, LT, lamb, toPrint=True) #compute w and b
        logPosterior = w.T @ DV + b #compute scores on validation
        previsionArray = (logPosterior>0)*1
        err = previsionArray[LV==previsionArray] / previsionArray.size

        prior = LT[LT==1].size / LT.size  #prior
        ssLLR = logPosterior - np.log(prior/ (1-prior))
        th = -np.log((prior * 1.0) / ((1 - prior) * 1.0))
        previsionArray2 = (ssLLR>th)*1 #should I use this instead?
        #note well: previsionArray1 and previsionArray2 are equivalent! previsionArray2 system is used for minDCF

        th = -np.log((0.5 * 1.0) / ((1 - 0.5) * 1.0))
        previsionArraySol = (ssLLR>th)*1 #should I use this instead with prior=0.5

        #compute DCF and min DCF:
        # confusionMatrix = ut.computeConfusionMatrix(previsionArray, LV) #is my interpretation wrong?
        confusionMatrix = ut.computeConfusionMatrix(previsionArraySol, LV)
        #note: we use 0.5 as prior, but actual prior is 
        print("DCF: %f " % (ut.computeDCFBayesError_Binary(confusionMatrix, 0.5, 1.0, 1.0)))
        print("DCF min: %f " % (ut.compute_minDCF_binary_slow(ssLLR, LV, 0.5, 1.0, 1.0)))

        #try with weighted: pT=0.8
        pT=0.8
        w, b = ut.trainLogRegBinary(DT, LT, lamb, toPrint=True, pT=pT)
        logPosterior = w.T @ DV + b
        ssLLR = logPosterior - np.log(pT / (1-pT))
        th = -np.log((pT * 1.0) / ((1.0 - pT) * 1.0))
        previsionArray = (ssLLR > th) *1
        confusionMatrix = ut.computeConfusionMatrix(previsionArray, LV)
        print("DCF pT - 0.8  =  %f " % (ut.computeDCFBayesError_Binary(confusionMatrix, pT, 1.0, 1.0)))
        print("DCF min - pT 0.8 = %f " % (ut.compute_minDCF_binary_slow(ssLLR, LV, pT, 1.0, 1.0)))



