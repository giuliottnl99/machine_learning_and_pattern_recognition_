import sys
import scipy
import numpy as np
sys.path.append('C:\\Users\\giuli\\OneDrive\\Documenti\\poliTo 2024\\machine_learning_and_pattern_recognition\\general_utils\\')
import utils as ut


if __name__ == "__main__":
    DComplete, LComplete = ut.load_iris()
    D = DComplete[:, LComplete!=0]
    L = LComplete[LComplete!=0]
    L[L==2] = 0

    (DT, LT), (DV, LV) = ut.divideSamplesRandomly(D, L)
    for k in [1, 10]:
        for c in [0.1, 1.0, 10.0]:
            w, b, primalLoss, dualLoss = ut.train_dual_SVM_linear(DT, LT, c, K=k)
            scores = (ut.vrow(w) @ DV + b).ravel()
            previsionArray = ((scores > 0) * 1).ravel()
            err = 1 - (previsionArray[previsionArray==LV].size / LV.size)
            confusionMatrix = ut.computeConfusionMatrix(previsionArray, LV)
            actualDCF = ut.computeDCFBayesError_Binary(confusionMatrix, 0.5, 1.0, 1.0)
            minDCF = ut.compute_minDCF_binary_fast(scores, LV, 0.5, 1.0, 1.0)
            print("SMV linear, for C: %f, k: %f => DCF min: %f, DCF actual: %f, primalLoss: %e, dualLoss: %e, error: %f"
                  % (c, k, minDCF, actualDCF,primalLoss, dualLoss, err ))

    #now for kernel:
    i=0
    for kernelFunc in [ut.polyKernel(2, 0), ut.polyKernel(2, 1), ut.rbfKernel(1.0), ut.rbfKernel(10.0)]:
        for eps in [0.0, 1.0]:
            funcScore, dualLoss = ut.train_dual_SVM_kernel(DT, LT, 1.0, kernelFunc, eps)
            scores = funcScore(DV)
            previsionArray = (scores > 0) * 1
            err = 1 - (LV[previsionArray==LV].size / LV.size)
            confusionMatrix = ut.computeConfusionMatrix(previsionArray, LV)
            actualDCF = ut.computeDCFBayesError_Binary(confusionMatrix, 0.5, 1.0, 1.0)
            minDCF = ut.compute_minDCF_binary_fast(scores, LV, 0.5, 1.0, 1.0)
            print("SMV kernel, epsilon: %f, function number: %d => DCF min: %f, DCF actual: %f,dualLoss: %e, error: %f"
                  % (eps, i, minDCF, actualDCF, dualLoss, err))
        i+=1
