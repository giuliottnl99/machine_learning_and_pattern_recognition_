import numpy as np
import sys
sys.path.append('C:\\Users\\giuli\\OneDrive\\Documenti\\poliTo 2024\\machine_learning_and_pattern_recognition\\general_utils\\')
print (sys.path)
import utils as ut
import matplotlib.pyplot as plt

def printEffPriors():
    for prior, Cfn, Cfp in [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)]:
        effPrior = ut.effectivePriorRappresentation(prior, Cfn, Cfp)
        print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp, "effectivePrior: ", effPrior)


#If I well understood, I should compute: all the 3 bayes cases using PCA, computing actual DCF and minimum DCF
#I need to understand what is LLRs -> It looks like a "1-d" array with the same function of the logPosteriorProbMatrix, but it's exp is not between 0 and 1
if __name__ == "__main__":
    DComplete, L = np.load("general_utils/D_exam_train.npy"), np.load("general_utils/L_exam_train.npy")
    (DTComplete, LT), (DVComplete, LV) = ut.divideSamplesRandomly(DComplete, L)
    #just write effective priors
    printEffPriors()
    print("start computation of DCFs with MVG and other variants")
    
    for prior in [0.1, 0.5, 0.9]:
        print("")
        print("start prior: ", prior)
        for variantChosen in ["ML", "naive", "tied"]:
            print("")
            print("starts: ", variantChosen)
            for PCADimension in [6, 4, 2]:
            #it looks like other priors are not useful
                print("prior: %f, variant of MVG picked: %s, PCA reduces to %d dimensions" % (prior, variantChosen, PCADimension))
                DPCA = DComplete
                llr_binary =  ut.applyMVGToComputeLLR_Binary(DPCA, L, chosenCase=variantChosen, PCAdimensions=PCADimension) #samples are divided inside the method! 
                #prevision matrix:
                predictionArr = ut.computePrevisionMatrixUsingCosts_Binary(llr_binary, prior, 1, 1, trueValue=1, falseValue=0)
                confusionMatrix = ut.computeConfusionMatrix(predictionArr, LV)
                # print("Confusion matrix:")
                # print(confusionMatrix)
                actualDCF = ut.computeDCFBayesError_Binary(confusionMatrix, prior, 1, 1, normalize=True)
                #fix bugs: verify confusion matrix in particular: it should make sense!
                minDCF = ut.compute_minDCF_binary_fast(llr_binary, LV, prior, 1, 1, returnThreshold=False)
                print("Actual DCF: %f, Min DCF: %f" % (actualDCF, minDCF))

    print("\n\n\n starts analysis for different prior log odds")
    #now consider PCA==6 (the best case)    
    plt.figure()
    plt.title("Prior log odd compared to DCF:")
    plt.xlabel("prior log odd")
    plt.ylabel("DCFs")

    colorsInfo = {"ML" : ["lightcoral", "royalblue", "MVG"], 
        "naive" : ["red", "midnightblue", "naive"], 
        "tied" : ["orangered", "slateblue", "tied"]}

    for variantChosen in ["ML", "naive", "tied"]:
        arrayOfMinDCFs = []
        arrayOfActualDCFs =  []
        for priorLogOdd in np.arange(-4, 5): #between -4 and +4 INCLUDED!
            prior = 1 / (1 + np.exp(-priorLogOdd))
            llr_binary =  ut.applyMVGToComputeLLR_Binary(DComplete, L, chosenCase=variantChosen) #samples are divided inside the method! 
            #prevision matrix:
            predictionArr = ut.computePrevisionMatrixUsingCosts_Binary(llr_binary, prior, 1, 1, trueValue=1, falseValue=0)
            confusionMatrix = ut.computeConfusionMatrix(predictionArr, LV)
            actualDCF = ut.computeDCFBayesError_Binary(confusionMatrix, prior, 1, 1, normalize=True)
            minDCF = ut.compute_minDCF_binary_fast(llr_binary, LV, prior, 1, 1, returnThreshold=False)
            arrayOfActualDCFs.append(actualDCF)
            arrayOfMinDCFs.append(minDCF)
        colorInfo = colorsInfo[variantChosen]
        plt.scatter(np.arange(-4, 5).astype(float), np.ravel(arrayOfActualDCFs).astype(float), color=colorInfo[0])
        plt.scatter(np.arange(-4, 5).astype(float), np.ravel(arrayOfMinDCFs).astype(float), color=colorInfo[1])
        plt.plot(np.arange(-4, 5).astype(float), np.ravel(arrayOfActualDCFs).astype(float), color=colorInfo[0], label="actual DCF for " + colorInfo[2], linestyle='dashed')
        plt.plot(np.arange(-4, 5).astype(float), np.ravel(arrayOfMinDCFs).astype(float), color=colorInfo[1], label="min DCF for " + colorInfo[2], linestyle='dashed')

    plt.legend()
    plt.show()
    stopDebug = ""
