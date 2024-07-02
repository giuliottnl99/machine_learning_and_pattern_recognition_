import sys
sys.path.append('C:\\Users\\giuli\\OneDrive\\Documenti\\poliTo 2024\\machine_learning_and_pattern_recognition\\general_utils\\')
import utils as ut
import numpy as np



def computeMVG_tied_naiveThenPCA(D, L):
    arrOfMatrices = [ D[:-2, :], D[:2, :], D[2:4, :], D[ 2:, : ] ]
    arrOfLogs = ["the first 4 features", "only 1 and 2 feature jointly", "only 3 and 4 feature jointly", "only features 3-4-5-6"]

    for i in range(len(arrOfLogs)):
        matrix = arrOfMatrices[i]
        log = arrOfLogs[i]
        
        accML = ut.computeAccuracyUsingBinaryDivision_MVG(matrix, L, 1, 0, chosenCase='ML', reduceDataset=True)
        print("Using %s the MVG solution error rate is: %f" % (log, ((1-accML)*100)))
        accTied = ut.computeAccuracyUsingBinaryDivision_MVG(matrix, L, 1, 0, chosenCase='tied', reduceDataset=True)
        print("Using %s the tied solution error rate is: %f" % (log, ((1-accTied)*100)))
        accNaive = ut.computeAccuracyUsingBinaryDivision_MVG(matrix, L, 1, 0, chosenCase='naive', reduceDataset=True)
        print("Using %s the naive solution error rate is: %f" % (log, ((1-accNaive)*100)))

    #apply PCA to reduce to 4 and then use ML and tied
    PCAReducingMatrix = ut.computePCA_ReducingMatrix(D, L, dim=4)
    matrixReduced = PCAReducingMatrix @ D
    accPCAML = ut.computeAccuracyUsingBinaryDivision_MVG(matrixReduced, L, 1, 0, chosenCase='ML', reduceDataset=True)
    print("Using the application of PCA (dim=4) and then MVG have as an error rate of: %f" % (((1-accPCAML)*100)))
    accPCATied = ut.computeAccuracyUsingBinaryDivision_MVG(matrixReduced, L, 1, 0, chosenCase='tied', reduceDataset=True)
    print("Using the application of PCA (dim=4) and then tied MVG have as an error rate of: %f" % (((1-accPCATied)*100)))
    accPCANaive = ut.computeAccuracyUsingBinaryDivision_MVG(matrixReduced, L, 1, 0, chosenCase='naive', reduceDataset=True)
    print("Using the application of PCA (dim=4) and then naive have as an error rate of: %f" % (((1-accPCANaive)*100)))
    



#here I experiment using a reduced sample!
if __name__ == '__main__':
    D, L = np.load("general_utils/D_exam_train.npy"), np.load("general_utils/L_exam_train.npy")
    #compute error rate using ML
    accML = ut.computeAccuracyUsingBinaryDivision_MVG(D, L, 1, 0, chosenCase='ML', reduceDataset=True)
    print("using MVG model the error rate is: %f" % ((1-accML)*100) )
    #compute error rate using tied
    accTied= ut.computeAccuracyUsingBinaryDivision_MVG(D, L, 1, 0, chosenCase='tied', reduceDataset=True)
    print("using tied Gaussian model the error rate is: %f" % ((1-accTied)*100) )
    #compute error rate using naive
    accNaive = ut.computeAccuracyUsingBinaryDivision_MVG(D, L, 1, 0, chosenCase='naive', reduceDataset=True)
    print("using naive Gaussian model the error rate is: %f" % ((1-accNaive)*100) )
    
    #compute error rate using LDA
    accLDA = ut.doBinaryClassification_PCA_LDA(D, L, chosenMethod='LDA', LValueTrue=1, LValueFalse=0, reduceDataset=True)
    print("accuracy using LDA reduction: %f" % ((1-accLDA)*100))


    #find pearson matrix:
    covTrue = ut.computeCovMatrix(D[:, L==1], L[L==1])
    covFalse = ut.computeCovMatrix(D[:, L==0], L[L==0])
    persCorrTrue = ut.computePearsonCorrCoeff(covTrue)
    persCorrFalse = ut.computePearsonCorrCoeff(covFalse)

    print("Covariance of True samples is:")
    print(covTrue)
    print("Pearson coefficient of True samples is:")
    print(persCorrTrue)

    print("Covariance of False samples is:")
    print(covFalse)
    print("Pearson coefficient of False samples is:")
    print(persCorrFalse)

    computeMVG_tied_naiveThenPCA(D, L)

