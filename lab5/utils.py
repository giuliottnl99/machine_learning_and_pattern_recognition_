import numpy as np

def computeMuAndCov(arr):
    x= np.array(arr).T
    muArr = x.mean(1)
    mu = np.matrix(x.mean(1)).reshape(len(muArr), 1)
    cov = ((x - mu) @ (x-mu).T) / len(arr)
    return mu, cov

def splitDataSets(fullArr, labels):
    setosa =  []
    versicolor = []
    virginica = []
    for  i in range(len(labels)):
        row = fullArr[:, i]
        label = labels[i]
        if label==0:
            setosa.append(row)
        elif label==1:
            versicolor.append(row)
        elif label==2:
            virginica.append(row)

    return setosa, versicolor, virginica    

