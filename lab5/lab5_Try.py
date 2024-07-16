import sklearn.datasets
import numpy as np 
import scipy
import sys
sys.path.append('C:\\Users\\giuli\\OneDrive\\Documenti\\poliTo 2024\\machine_learning_and_pattern_recognition\\general_utils\\')
import utils as ut

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L


def ApplyMVGLab5():
    D, L = load_iris()
    accML, _ = ut.createAndApplyMVG(D, L, chosenCase='ML')
    print("Accuracy for ML solution: %f" % (accML))
    accNaive, _  = ut.createAndApplyMVG(D, L, chosenCase='naive')
    print("Accuracy for Naive solution: %f" % (accNaive))
    accTied, _  = ut.createAndApplyMVG(D, L, chosenCase='tied')
    print("Accuracy for Tied solution: %f" % (accTied))
    #class-problem divsion:
    DReduced = D[:, L!=0]
    LReduced = L[L!=0]
    accBinary, _  = ut.createAndApplyMVG(DReduced, LReduced, chosenCase='ML')
    print("Accuracy for binary using ML: %f" % (accBinary))
    accBinaryDiv = ut.computeAccuracyUsingBinaryDivision_MVG(DReduced, LReduced, 2, 1, chosenCase='ML')
    print("Accuracy for binary using binary division: %f" % (accBinaryDiv))



if __name__ == '__main__':
    # oldMethods()
    ApplyMVGLab5()





