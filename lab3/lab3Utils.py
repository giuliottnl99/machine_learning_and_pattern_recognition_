import numpy as np
import matplotlib.pyplot as plt


def divideSamples(allSamples):
    allSamplesArray = np.ravel(allSamples)
    permutedSequence =  np.random.permutation(allSamplesArray)
    totalSize = len(permutedSequence)
    trainElemQt = totalSize*2/3
    trainData = []
    testData = []
    for i in range(totalSize):
        elem = permutedSequence[i]
        if i<trainElemQt:
            trainData.append(elem)
        else:
            testData.append(elem)
    return trainData, testData    


# def plot(arrXtoPlot, arrYtoPlot, color, label):
#         #plot x of matrixX and y of the specific matrix of the Y
#         plt.xlabel('caratt 1')
#         plt.ylabel('caratt 2')
#         plt.scatter(np.ravel(arrXtoPlot).astype(float), np.ravel(arrYtoPlot).astype(float), label=label, color=color)
#         plt.tight_layout()
#         plt.legend()


def plot_hist(arrXtoPlot, labelX, arrYToPlot, labelY, intervals, title):
    plt.figure()
    plt.title(title)
    plt.hist(np.ravel(arrXtoPlot).astype(float), bins=intervals, density=True, ec='black', alpha = 0.4, color='red', label=labelX)
    plt.hist(np.ravel(arrYToPlot).astype(float), bins=intervals, density=True, ec='black', alpha = 0.4, color='blue', label=labelY)
