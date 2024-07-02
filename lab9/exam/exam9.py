import numpy as np
import sys
sys.path.append('C:\\Users\\giuli\\OneDrive\\Documenti\\poliTo 2024\\machine_learning_and_pattern_recognition\\general_utils\\')
import utils as ut
import matplotlib.pyplot as plt
import os


def loadScoresLinear(DT, LT, DV, path):
    primalLosses = []
    dualLosses = []
    k=1
    i=0
    for c in np.logspace(-5, 0, 11):
        w, b, primalLoss, dualLoss = ut.train_dual_SVM_linear(DT, LT, c, K=k)
        scores = (ut.vrow(w) @ DV + b).ravel()
        np.save(path + "scoresLinear_" + str(i) + ".npy", scores)
        primalLosses.append(primalLoss)
        dualLosses.append(dualLoss[0, 0])
        i+=1
    np.save(path + "arrLossesLinear.npy", np.vstack((np.array(primalLosses), np.array(dualLosses), np.array(np.logspace(-5, 0, 11)))))

    #kernel:
    #kernel d=2, c=1, eps=0, c=1, 
def loadScoresKernelPoly(DT, LT, DV, path):
    i=0
    kernelFunc  = ut.polyKernel(2, 1)
    eps = 0.0
    dualLosses = []
    for C in np.logspace(-5, 0, 11):
        funcScore, dualLoss = ut.train_dual_SVM_kernel(DT, LT, C, kernelFunc, eps)
        scores = funcScore(DV)
        np.save(path + "scoreKernel_" + str(i) + ".npy", scores)
        dualLosses.append(dualLoss[0, 0])
        i+=1
    np.save(path + "arrLossesKernel.npy", np.vstack((np.array(dualLosses), np.array(np.logspace(-5, 0, 11)))))

#TODO: function to load with RBF. Differences: 
def loadScoresKernelRBF(DT, LT, DV):
    i=0
    eps = 1.0
    dualLosses = []
    #in array put: gamma, C, 
    #you cannot properly plot, but you can print!
    for gamma in np.logspace(-4, -1, 4):
        j=0
        dualLosses.append([])
        for C in np.logspace(-3, 2, 11):
            kernelFunc = ut.rbfKernel(gamma)
            funcScore, dualLoss = ut.train_dual_SVM_kernel(DT, LT, C, kernelFunc, eps)
            scores = funcScore(DV)
            np.save("lab9/exam/save/kernelRBF/gamma_case_" + str(i) +"/scoreKernelRBF_" + str(j) + ".npy", scores)
            dualLosses[i].append(dualLoss[0, 0])
            j+=1
        i+=1
    np.save("lab9/exam/save/kernelRBF/matrixLossesKernelRBF.npy", np.vstack((np.array(dualLosses), np.array(np.logspace(-5, 0, 11)))))

if __name__ == "__main__":
    #remember:
    #K=1, different values of C, piT=0
    D, L = np.load("general_utils/D_exam_train.npy"), np.load("general_utils/L_exam_train.npy")
    (DT, LT), (DV, LV) = ut.divideSamplesRandomly(D, L)
    pT=0.1

    k=1.0
    DTCentered = DT - np.mean(DT, axis=0)
    DVCentered = DV - np.mean(DV, axis=0)  

    DTQuadratic = ut.computeQuadraticXforSMV(DT)  
    DVQuadratic = ut.computeQuadraticXforSMV(DV)  

    #code to load scores (comment if scores are loaded):
    # loadScoresLinear(DT, LT, DV, "lab9/exam/save/linear/")
    # loadScoresLinear(DTCentered, LT, DVCentered, "lab9/exam/save/linear_centered/") #centered case
    # loadScoresKernelPoly(DT, LT, DV, "lab9/exam/save/kernelPoly/")
    # loadScoresKernelPoly(DTQuadratic, LT, DVQuadratic, "lab9/exam/save/kernelPoly_quadratic/") #quadratic case
    # loadScoresKernelRBF(DT, LT, DV) #TODO: to be fixed!

    #linear case:
    losses = np.load("lab9/exam/save/linear/arrLossesLinear.npy")
    actualDCFs = []
    minDCFs = []
    for i in range(11):
        strSave = "lab9/exam/save/linear/scoresLinear_" + str(i) + ".npy"
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
    
    plt.figure()
    plt.title("DCF acutal/min for linear SVM")
    plt.xscale('log', base=10)
    plt.xlabel("value of regulizer C")
    plt.ylabel("DCF actual/min")

    plt.scatter(np.ravel(losses[2, :]), actualDCFs, color="red", label="actual DCF")
    plt.scatter(np.ravel(losses[2, :]), minDCFs, color="blue", label="minimum DCF")
    plt.legend()

    #linear case, but centered:
    losses = np.load("lab9/exam/save/linear_centered/arrLossesLinear.npy")
    actualDCFs = []
    minDCFs = []
    for i in range(11):
        strSave = "lab9/exam/save/linear_centered/scoresLinear_" + str(i) + ".npy"
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
    
    plt.figure()
    plt.title("DCF acutal/min for linear centered SVM")
    plt.xscale('log', base=10)
    plt.xlabel("value of regulizer C")
    plt.ylabel("DCF actual/min")

    plt.scatter(np.ravel(losses[2, :]), actualDCFs, color="red", label="actual DCF")
    plt.scatter(np.ravel(losses[2, :]), minDCFs, color="blue", label="minimum DCF")
    plt.legend()

    #Kernel poly linear:
    losses = np.load("lab9/exam/save/kernelPoly/arrLossesKernel.npy")
    actualDCFs = []
    minDCFs = []
    for i in range(11):
        scores = np.load("lab9/exam/save/kernelPoly/scoreKernel_" + str(i) + ".npy")
        dualLoss = losses[0, i]
        C = losses[1, i]
        previsionArray = (scores > 0) * 1
        err = 1 - (LV[previsionArray==LV].size / LV.size)
        confusionMatrix = ut.computeConfusionMatrix(previsionArray, LV)
        actualDCF = ut.computeDCFBayesError_Binary(confusionMatrix, pT, 1.0, 1.0)
        minDCF = ut.compute_minDCF_binary_fast(scores, LV, pT, 1.0, 1.0)
        print("SMV kernel poly, epsilon: %f, C: %e, function number: %d => DCF min: %f, DCF actual: %f,dualLoss: %e, error: %f"
            % (0.0, C, i, minDCF, actualDCF, dualLoss, err))
        actualDCFs.append(actualDCF)
        minDCFs.append(minDCF)
    #now plot:
    plt.figure()
    plt.title("DCF actual/min for kernel poly SVM")
    plt.xscale('log', base=10)
    plt.xlabel("value of regulizer C")
    plt.ylabel("DCF actual/min")

    plt.scatter(np.ravel(losses[1, :]), actualDCFs, color="red", label = "actual DCF")
    plt.scatter(np.ravel(losses[1, :]), minDCFs, color="blue", label="minimum DCF")
    plt.legend()

    #Kernel poly quadratic:
    losses = np.load("lab9/exam/save/kernelPoly_quadratic/arrLossesKernel.npy")
    actualDCFs = []
    minDCFs = []
    for i in range(11):
        scores = np.load("lab9/exam/save/kernelPoly_quadratic/scoreKernel_" + str(i) + ".npy")
        dualLoss = losses[0, i]
        C = losses[1, i]
        previsionArray = (scores > 0) * 1
        err = 1 - (LV[previsionArray==LV].size / LV.size)
        confusionMatrix = ut.computeConfusionMatrix(previsionArray, LV)
        actualDCF = ut.computeDCFBayesError_Binary(confusionMatrix, pT, 1.0, 1.0)
        minDCF = ut.compute_minDCF_binary_fast(scores, LV, pT, 1.0, 1.0)
        print("SMV kernel poly quadratic, epsilon: %f, C: %e, function number: %d => DCF min: %f, DCF actual: %f,dualLoss: %e, error: %f"
            % (0.0, C, i, minDCF, actualDCF, dualLoss, err))
        actualDCFs.append(actualDCF)
        minDCFs.append(minDCF)
    #now plot:
    plt.figure()
    plt.title("DCF actual/min for kernel poly quadratic SVM")
    plt.xscale('log', base=10)
    plt.xlabel("value of regulizer C")
    plt.ylabel("DCF actual/min")

    plt.scatter(np.ravel(losses[1, :]), actualDCFs, color="red", label = "actual DCF")
    plt.scatter(np.ravel(losses[1, :]), minDCFs, color="blue", label="minimum DCF")
    plt.legend()

    #Kernel RBF must be done!
    losses = np.load("lab9/exam/save/kernelRBF/matrixLossesKernelRBF.npy")
    #for those matrices each row is a value of gamma, each column a value of C
    actualDCFsMatrix = np.zeros((4, 11)) 
    minDCFsMatrix = np.zeros((4, 11)) 
    plt.figure()
    plt.title("DCF actual/min for RFB kernel SVM")
    plt.xscale('log', base=10)
    plt.xlabel("value of regulizer C")
    plt.ylabel("DCF actual/min")
    colorsMinDCF = ["midnightblue", "blue", "slateblue", "indigo"]
    colorsActDCF = ["lightcoral", "red", "orangered", "saddlebrown"]
    gammaRanges = np.logspace(-4, -1, 4)

    for gammaCount in range(4):
        for cIndex in range(11):
            scores = np.load("lab9/exam/save/kernelRBF/gamma_case_" + str(gammaCount) +"/scoreKernelRBF_" + str(cIndex) + ".npy")
            dualLoss = losses[gammaCount, cIndex]
            C = losses[losses.shape[0]-1, cIndex]
            previsionArray = (scores > 0) * 1
            err = 1 - (LV[previsionArray==LV].size / LV.size)
            confusionMatrix = ut.computeConfusionMatrix(previsionArray, LV)
            actualDCF = ut.computeDCFBayesError_Binary(confusionMatrix, pT, 1.0, 1.0)
            minDCF = ut.compute_minDCF_binary_fast(scores, LV, pT, 1.0, 1.0)
            print("SMV kernel poly quadratic, epsilon: %f, C: %e, function number: %d => DCF min: %f, DCF actual: %f,dualLoss: %e, error: %f"
                % (0.0, C, cIndex, minDCF, actualDCF, dualLoss, err))
            actualDCFsMatrix[gammaCount, cIndex] = actualDCF
            minDCFsMatrix[gammaCount, cIndex] = minDCF
        #for each gamma, a different plot (scatter and lines):
        plt.scatter(np.ravel(losses[losses.shape[0]-1, :]), actualDCFsMatrix[gammaCount, :], color=colorsActDCF[gammaCount] )
        plt.scatter(np.ravel(losses[losses.shape[0]-1, :]), minDCFsMatrix[gammaCount, :], color=colorsMinDCF[gammaCount])
        plt.plot(np.ravel(losses[losses.shape[0]-1, :]), actualDCFsMatrix[gammaCount, :], color=colorsActDCF[gammaCount], label = "actual DCF for gamma = " + str(gammaRanges[gammaCount]), linestyle='dashed')
        plt.plot(np.ravel(losses[losses.shape[0]-1, :]), minDCFsMatrix[gammaCount, :], color=colorsMinDCF[gammaCount], label = "minimum DCF for gamma = " + str(gammaRanges[gammaCount]), linestyle='dashed')

    plt.legend()
    plt.show()
    stopDebug = ""
