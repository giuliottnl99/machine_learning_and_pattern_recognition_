import numpy as np
import sys
sys.path.append('C:\\Users\\giuli\\OneDrive\\Documenti\\poliTo 2024\\machine_learning_and_pattern_recognition\\general_utils\\')
print (sys.path)
import utils as ut



#If I well understood, I should compute: all the 3 bayes cases using PCA, computing actual DCF and minimum DCF
#I need to understand what is LLRs -> It looks like a "1-d" array with the same function of the logPosteriorProbMatrix, but it's exp is not between 0 and 1
if __name__ == "__main__":
    DComplete, L = np.load("general_utils/D_exam_train.npy"), np.load("general_utils/L_exam_train.npy")
    (DTComplete, LT), (DVComplete, LV) = ut.divideSamplesRandomly(DComplete, L)


    for variantChosen in ["ML", "naive", "tied"]:
        for PCADimension in [2, 4, 6]:
            #it looks like other priors are not useful
            # for prior, Cfn, Cfp in [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)]:
            for prior, Cfn, Cfp in [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0)]:
                print("variant of MVG picked: %s" % (variantChosen))
                DPCA = DComplete
                if PCADimension!=6:
                    pcaReducingMatrix = ut.computePCA_ReducingMatrix(DComplete, L, dim=PCADimension)
                    DPCA = pcaReducingMatrix @ DComplete
                    print("Applied PCA with reduction to %d " % (PCADimension))
                else:
                    print("PCA not applied")
                llr_binary =  ut.applyMVGToComputeLLR_Binary(DPCA, L, chosenCase=variantChosen) #samples are divided inside the method! 
                print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp)
                #prevision matrix:
                predictionArr = ut.computePrevisionMatrixUsingCosts_Binary(llr_binary, prior, Cfn, Cfp, trueValue=1, falseValue=0)
                confusionMatrix = ut.computeConfusionMatrix(predictionArr, LV)
                print("Confusion matrix:")
                print(confusionMatrix)
                DCF = ut.computeDCFBayesError_Binary(confusionMatrix, prior, Cfn, Cfp, normalize=False)
                print("DCF: %f" % (DCF))
                #fix bugs: verify confusion matrix in particular: it should make sense!
                minDCF = ut.compute_minDCF_binary_slow(llr_binary, LV, prior, Cfn, Cfp, returnThreshold=False)

        