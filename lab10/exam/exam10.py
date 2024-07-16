import sys
sys.path.append('C:\\Users\\giuli\\OneDrive\\Documenti\\poliTo 2024\\machine_learning_and_pattern_recognition\\general_utils\\')
import utils as ut
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    D, L = np.load("general_utils/D_exam_train.npy"), np.load("general_utils/L_exam_train.npy")
    (DT, LT), (DV, LV) = ut.divideSamplesRandomly(D, L)
    for covType in ['full', 'diagonal']:
        print()
        print("start %s model" % (covType))
        for nComp in [2, 4, 8, 16, 32]:
            gmmTrue = ut.train_GMM_LBG_EM(DT[:, LT==1], LT[LT==1], nComp, covType=covType, verbose=False, psiEig = 0.01)
            gmmFalse = ut.train_GMM_LBG_EM(DT[:, LT==0], LT[LT==0], nComp, covType=covType, verbose=False, psiEig = 0.01)
            SLLR = ut.logpdf_GMM(DV, gmmTrue) - ut.logpdf_GMM(DV, gmmFalse)
            decisions = (SLLR > 0) * 1
            accuracy = len(decisions[LV==decisions]) / len(LV)
            minDCF = ut.compute_minDCF_binary_fast(SLLR, LV, 0.5, 1, 1)
            confusionMatrix = ut.computeConfusionMatrix(decisions, LV) 
            actDcf = ut.computeDCFBayesError_Binary(confusionMatrix, 0.5, 1.0, 1.0)
            print("for %d components: accuracy: %f, minDCF: %f, actDCF: %f" % (nComp, accuracy, minDCF, actDcf) )