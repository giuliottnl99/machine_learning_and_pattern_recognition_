import numpy as np
import sys
sys.path.append('C:\\Users\\giuli\\OneDrive\\Documenti\\poliTo 2024\\machine_learning_and_pattern_recognition\\general_utils\\')
import utils as ut

if __name__ == "__main__":
    D, L = ut.load_iris()
    (DT, LV), (DV, LV) = ut.divideSamplesRandomly(D, L)
    _, previsionArrayMVG = ut.createAndApplyMVG(D, L, chosenCase='ML')
    confusionMatrixMVG = ut.computeConfusionMatrix(previsionArrayMVG, LV)
    print("confusion matrix MVG:")
    print(confusionMatrixMVG)

    _, previsionArrayTied = ut.createAndApplyMVG(D, L, chosenCase='tied')
    confusionMatrixTied = ut.computeConfusionMatrix(previsionArrayTied, LV)
    print("confusion matrix tied:")
    print(confusionMatrixTied)

    #now compute confusion matrix for the comedia:
    comediaLogScoreMatrix = np.load('lab7/data/commedia_ll.npy')  
    comediaLabels = np.load('lab7/data/commedia_labels.npy')  
    comediaPosteriorProbMatrix = ut.computePosterior(comediaLogScoreMatrix)
    comediaPrevisionArray = ut.computePrevisionArray(comediaPosteriorProbMatrix, None)
    comediaConfusionMatrix = ut.computeConfusionMatrix(comediaPrevisionArray, comediaLabels)
    print("confusion matrix for Comedia:")
    print(comediaConfusionMatrix)

    #binary division:
    print("begin binary division:")
    llrArrBin = np.load('lab7/data/commedia_llr_infpar.npy')
    labelsBin = np.load('lab7/data/commedia_labels_infpar.npy')

    #cfn = cost false negativem Cfp = cost false position, prior = % of elements of the specific class
    for prior, Cfn, Cfp in [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]:
        print()
        print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp)
        predBin = ut.computePrevisionMatrixUsingCosts_Binary(llrArrBin, prior, Cfn, Cfp)
        confusionMatrixBin = ut.computeConfusionMatrix(predBin, labelsBin)
        print("Confusion matrix:")
        print(confusionMatrixBin)

        print("DCF not normalized binary: ")
        bayesRisk1 = ut.computeDCFBayesError_Binary(confusionMatrixBin, prior, Cfn, Cfp, normalize=False)
        print(bayesRisk1)

        print("DCF not normalized multiclass (should be the same as binary: we have only 2 classes): ")
        bayesRisk2 = ut.computeDCFBayesError_Multiclass(confusionMatrixBin, [1-prior, prior], np.matrix([ [0, Cfn], [Cfp, 0] ]), normalize=False)
        print(bayesRisk2)

        print("DCF normalized binary: ")
        bayesRisk3 = ut.computeDCFBayesError_Binary(confusionMatrixBin, prior, Cfn, Cfp, normalize=True)
        print(bayesRisk3)

        print("DCF normalized multiclass (should be the same as binary: we have only 2 classes): ")
        bayesRisk4 = ut.computeDCFBayesError_Multiclass(confusionMatrixBin, [prior, 1-prior],  np.matrix([ [0, Cfn], [Cfp, 0] ]), normalize=True)
        print(bayesRisk4)

        minDCFNormalized, thresholdForMinDCF = ut.compute_minDCF_binary_slow(llrArrBin, labelsBin, prior, Cfn, Cfp, returnThreshold=True)
        print("Min DCF normalized: %f\n Correspondent threshold: %f" %  (minDCFNormalized, thresholdForMinDCF))


        # minDCF, minDCFThreshold = compute_minDCF_binary_slow(commedia_llr_binary, labelsBin, prior, Cfn, Cfp, returnThreshold=True)
        # print('MinDCF (normalized, slow): %.3f (@ th = %e)' % (minDCF, minDCFThreshold))
        # minDCF, minDCFThreshold = compute_minDCF_binary_fast(commedia_llr_binary, labelsBin, prior, Cfn, Cfp, returnThreshold=True)
        # print('MinDCF (normalized, fast): %.3f (@ th = %e)' % (minDCF, minDCFThreshold))
    
