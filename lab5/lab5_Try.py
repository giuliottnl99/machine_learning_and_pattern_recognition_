import sklearn.datasets
import numpy as np 
import utils as ut

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def split_DB_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)
    



if __name__ == '__main__':
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_DB_2to1(D, L)

    #first try: split dataset based on whole matrix;
    setosa, versicolor, virginica = ut.splitDataSets(D, L)
    mu1, cov1 = ut.computeMuAndCov(setosa)
    mu2, cov2 = ut.computeMuAndCov(versicolor)
    mu3, cov3 = ut.computeMuAndCov(virginica)

    print('setosa:')
    print(mu1)
    print(cov1)
    print('versicolor:')
    print(mu2)
    print(cov2)
    print('virginica:')
    print(mu3)
    print(cov3)




