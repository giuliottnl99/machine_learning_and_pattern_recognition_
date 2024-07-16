import numpy as np
import sys
sys.path.append('C:\\Users\\giuli\\OneDrive\\Documenti\\poliTo 2024\\machine_learning_and_pattern_recognition\\general_utils\\')
import utils as ut
import matplotlib.pyplot as plt
erwada

def loadScoresLinear(DT, LT, DV):
    primalLosses = []
    dualLosses = []
    k=1
    i=0
    for c in np.logspace(-5, 0, 11):
        w, b, primalLoss, dualLoss = ut.train_dual_SVM_linear(DT, LT, c, K=k)
        scores = (ut.vrow(w) @ DV + b).ravel()
        np.save("lab9/exam/save/scoresLinear_" + str(i) + ".npy", scores)
        primalLosses.append(primalLoss)
        dualLosses.append(dualLoss[0, 0])
        i+=1
    np.save("lab9/exam/save/arrLossesLinear.npy", np.vstack((np.array(primalLosses), np.array(dualLosses), np.array(np.logspace(-5, 0, 11)))))

    #kernel:
    #kernel d=2, c=1, eps=0, c=1, 
def loadScoresKernelPoly(DT, LT, DV):
    i=0
    kernelFunc  = ut.polyKernel(2, 1)
    eps = 0.0
    dualLosses = []
    for C in np.logspace(-5, 0, 11):
        funcScore, dualLoss = ut.train_dual_SVM_kernel(DT, LT, C, kernelFunc, eps)
        scores = funcScore(DV)
        np.save("lab9/exam/save/scoreKernel_" + str(i) + ".npy", scores)
        dualLosses.append(dualLoss[0, 0])
        i+=1
    np.save("lab9/exam/save/arrLossesKernel.npy", np.vstack((np.array(dualLosses), np.array(np.logspace(-5, 0, 11)))))




if __name__ == "__main__":
    #remember:
    #K=1, different values of C, piT=0
    D, L = np.load("general_utils/D_exam_train.npy"), np.load("general_utils/L_exam_train.npy")
    (DT, LT), (DV, LV) = ut.divideSamplesRandomly(D, L)
    pT=0.1

    k=1.0
    
    plt.figure()
    plt.title("")
    plt.xscale('log', base=10)
    plt.xlabel("value of regulizer C")
    plt.ylabel("DCF actual/min")

    #code to load scores (comment if scores are loaded):
    # loadScoresLinear(DT, LT, DV)
    # loadScoresKernelPoly(DT, LT, DV)

    losses = np.load("lab9/exam/save/arrLossesLinear.npy")
    actualDCFs = []
    minDCFs = []
    for i in range(11):
        strSave = "lab9/exam/save/scoresLinear_" + str(i) + ".npy"
        scores = np.load(strSave)
        primalLoss = losses[0, i]
        dualLoss = losses[1, i]
        c = losses[2, i]
        previsionArray = ((scores > 0) * 1).ravel()
        err = 1 - (previsionArray[previsionArray==LV].size / LV.size)
        confusionMatrix = ut.computeConfusionMatrix(previsionArray, LV)
        actualDCF = ut.computeDCFBayesError_Binary(confusionMatrix, pT, 1.0, 1.0)
        minDCF = ut.compute_minDCF_binary_fast(scores, LV, pT, 1.0, 1.0)
        print("SMV linear, for C: %f, k: %f => DCF min: %f, DCF actual: %f, primalLoss: %e, dualLoss: %e, error: %f"
            % (c, k, minDCF, actualDCF,primalLoss, dualLoss, err ))
        actualDCFs.append(actualDCF)
        minDCFs.append(minDCF)
    plt.scatter(np.ravel(losses[2, :]), actualDCFs, color="red", label = "actual DCF")
    plt.scatter(np.ravel(losses[2, :]), minDCFs, color="blue", label="minimum DCF")
    #second to delete:
    plt.figure()
    plt.title("")
    plt.xscale('log', base=10)
    plt.xlabel("value of regulizer C")
    plt.ylabel("DCF actual")
    plt.scatter(np.ravel(losses[2, :]), actualDCFs, color="red", label = "actual DCF")

    plt.show()
    stopPlot = 0

    #Kernel:
    arrLosses = np.load("lab9/exam/save/arrLossesKernel.npy")
    actualDCFs = []
    minDCFs = []
    for i in range(11):
        scores = np.load("lab9/exam/save/scoreKernel_" + str(i) + ".npy")
        dualLoss = arrLosses[0, i]
        C = arrLosses[1, i]
        previsionArray = (scores > 0) * 1
        err = 1 - (LV[previsionArray==LV].size / LV.size)
        confusionMatrix = ut.computeConfusionMatrix(previsionArray, LV)
        actualDCF = ut.computeDCFBayesError_Binary(confusionMatrix, pT, 1.0, 1.0)
        minDCF = ut.compute_minDCF_binary_fast(scores, LV, pT, 1.0, 1.0)
        print("SMV kernel, epsilon: %f, C function number: %d => DCF min: %f, DCF actual: %f,dualLoss: %e, error: %f"
            % (0.0, i, minDCF, C, actualDCF, dualLoss, err))
        actualDCFs.append(actualDCF)
        minDCFs.append(minDCF)



    # i=1
    # for c in np.logspace(-5, 0, 11):
    #     w, b, primalLoss, dualLoss = ut.train_dual_SVM_linear(DT, LT, c, K=k)
    #     scores = (ut.vrow(w) @ DV + b).ravel()
    #     strSave = "lab9/exam/scores/exam9Scores_" + str(i) + ".npy"
    #     np.save(strSave, scores)
    #     previsionArray = ((scores > 0) * 1).ravel()
    #     err = 1 - (previsionArray[previsionArray==LV].size / LV.size)
    #     confusionMatrix = ut.computeConfusionMatrix(previsionArray, LV)
    #     actualDCF = ut.computeDCFBayesError_Binary(confusionMatrix, pT, 1.0, 1.0)
    #     minDCF = ut.compute_minDCF_binary_fast(scores, LV, pT, 1.0, 1.0)
    #     print("SMV linear, for C: %f, k: %f => DCF min: %f, DCF actual: %f, primalLoss: %e, dualLoss: %e, error: %f"
    #         % (c, k, minDCF, actualDCF,primalLoss, dualLoss, err ))
    #     plt.scatter(actualDCF, np.ravel(c).astype(float), color="red", label = "actual DCF")
    #     plt.scatter(minDCF, np.ravel(c).astype(float), color="blue", label="minimum DCF")
    #     i+=1
    # plt.show()    
