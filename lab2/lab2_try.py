import sys
import numpy
import matplotlib
import matplotlib.pyplot as plt
import os


class IrisSetosa:
    def __init__(self, sepalLength, sepalWidth, petalLength, petalWidth, category):
        self.sepalLength = sepalLength
        self.sepalWidth = sepalWidth
        self.petalLength = petalLength
        self.petalWidth = petalWidth
        self.category = category


class HistVector:
    def __init__(self, setosaArray, versicolorArray, virginicaArray):
        self.setosaArray = setosaArray
        self.versicolorArray = versicolorArray
        self.virginicaArray = virginicaArray


def convertFromClassNameToClassNumericIdentified(className):
    if className=='Iris-setosa':
        return 0
    elif className=='Iris-versicolor':
        return 1
    elif className=='Iris-virginica':
        return 2

def load(fileName):
    matrixResult = []
    vectorClassesInt = []
    irisArray = []
    with open(fileName) as file:
        for line in file:
            dataLine = line.replace("\n", "").split(",")
            irisArray.append(
                IrisSetosa(dataLine[0], dataLine[1], dataLine[2], dataLine[3], dataLine[4])
            )
            matrixResult.append(dataLine[0:-1])
            vectorClassesInt.append(convertFromClassNameToClassNumericIdentified(dataLine[-1]))
    return numpy.hstack(matrixResult), numpy.array(vectorClassesInt), irisArray


def plotSingle_Hist(toPlotMatrix, label="no inserted label"):
    plt.figure()
    plt.xlabel(label)
    commonVector = numpy.concatenate((toPlotMatrix.setosaArray, toPlotMatrix.versicolorArray, toPlotMatrix.virginicaArray))
    minOfAll = min(commonVector)
    maxOfAll = max(commonVector)
    print(minOfAll)
    print(maxOfAll)

    intervals = numpy.linspace(float(minOfAll), float(maxOfAll), 8)
    plt.hist(numpy.ravel(toPlotMatrix.setosaArray).astype(float), bins=intervals, density=True, ec='black', alpha = 0.4, color='red', label="setosa")
    plt.hist(numpy.ravel(toPlotMatrix.versicolorArray).astype(float), bins=intervals, density=True, ec='black', alpha = 0.4,  color='yellow', label="versicolor")
    plt.hist(numpy.ravel(toPlotMatrix.virginicaArray).astype(float), bins=intervals, density=True, ec='black', alpha = 0.4, color='green', label="virginica")
    plt.legend()
    plt.tight_layout()
    directory = 'saved_figures/hist/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig('saved_figures/hist/hist%s.pdf' % (label))
    plt.show()


def plotSingleScatter(matrixX, matricesY, labelX, labelsY):
    #for the given matrix to rappresent:
    #foreach quality not passed
    for i in range(len(matricesY)):
        matrixY = matricesY[i]
        labelY = labelsY[i]
        #plot x of matrixX and y of the specific matrix of the Y
        plt.figure()
        plt.xlabel(labelX)
        plt.ylabel(labelY)
        # plt.gca().invert_yaxis()

        plt.scatter(numpy.ravel(matrixX.setosaArray).astype(float), numpy.ravel(matrixY.setosaArray).astype(float), label='Setosa', color='red')
        plt.scatter(numpy.ravel(matrixX.versicolorArray).astype(float), numpy.ravel(matrixY.versicolorArray).astype(float), label='Versicolor', color='blue')
        plt.scatter(numpy.ravel(matrixX.virginicaArray).astype(float), numpy.ravel(matrixY.virginicaArray).astype(float), label='Virginica', color='green')
        plt.tight_layout()
        plt.legend()
        directory = ('saved_figures/scatter/x%s' % labelX)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig('saved_figures/scatter/x%s/scatter_x%s_y%s.pdf' % (labelX, labelX, labelY))
        plt.show()


def divideArrayByHistCategory(irisArray):
    sepalLengthMatrix, sepalWidthMatrix, petalLengthMatrix, petalWidthMatrix  = HistVector([], [], []), HistVector([], [], []), HistVector([], [], []), HistVector([], [], [])
    for row in irisArray:
        if(row.category=='Iris-setosa'):
            sepalLengthMatrix.setosaArray.append(row.sepalLength)
            sepalWidthMatrix.setosaArray.append(row.sepalWidth)
            petalLengthMatrix.setosaArray.append(row.petalLength)
            petalWidthMatrix.setosaArray.append(row.petalWidth)
        if(row.category=='Iris-versicolor'):
            sepalLengthMatrix.versicolorArray.append(row.sepalLength)
            sepalWidthMatrix.versicolorArray.append(row.sepalWidth)
            petalLengthMatrix.versicolorArray.append(row.petalLength)
            petalWidthMatrix.versicolorArray.append(row.petalWidth)
        if(row.category=='Iris-virginica'):
            sepalLengthMatrix.virginicaArray.append(row.sepalLength)
            sepalWidthMatrix.virginicaArray.append(row.sepalWidth)
            petalLengthMatrix.virginicaArray.append(row.petalLength)
            petalWidthMatrix.virginicaArray.append(row.petalWidth)
    return sepalLengthMatrix, sepalWidthMatrix, petalLengthMatrix, petalWidthMatrix


def plotAllhist(sepalLengthMatrix, sepalWidthMatrix, petalLengthMatrix, petalWidthMatrix): #toend
    plotSingle_Hist(sepalLengthMatrix, label="sepal Length")
    plotSingle_Hist(sepalWidthMatrix, label="sepal Width")
    plotSingle_Hist(petalLengthMatrix, label="petal Length")
    plotSingle_Hist(petalWidthMatrix, label="petal Width")

def plotAllScatter(sepalLengthMatrix, sepalWidthMatrix, petalLengthMatrix, petalWidthMatrix): #toend
    plotSingleScatter(sepalLengthMatrix, [sepalWidthMatrix, petalLengthMatrix, petalWidthMatrix], "sepal Length", ["sepal Width", "petal Length", "petal Width"])
    plotSingleScatter(sepalWidthMatrix, [sepalLengthMatrix, petalLengthMatrix, petalWidthMatrix], "sepal Width", ["sepal Length", "petal Length", "petal Width"])
    plotSingleScatter(petalLengthMatrix, [sepalWidthMatrix, sepalLengthMatrix, petalWidthMatrix], "petal Length", ["sepal Width", "sepal Length", "petal Width"])
    plotSingleScatter(petalWidthMatrix, [sepalWidthMatrix, petalLengthMatrix, sepalLengthMatrix], "petal Width", ["sepal Width", "petal Length", "sepal Length"])

#TODO: sepalLegthMatrix
def computeAverage(descriptorMatrix):
    completeArray = numpy.ravel(numpy.concatenate(descriptorMatrix.setosaArray ,descriptorMatrix.versicolorArray ,descriptorMatrix.virginicaArray))
    return completeArray.mean(1)



if __name__ == '__main__':
    matrix, vector, irisArray = load(sys.argv[1])
    #print(matrix)
    #print(vector)
    #plot
    sepalLengthMatrix, sepalWidthMatrix, petalLengthMatrix, petalWidthMatrix = divideArrayByHistCategory(irisArray)
    if sys.argv[2]=='histSepalLength':
        plotSingle_Hist(sepalLengthMatrix, label="sepal Length")
    elif sys.argv[2]=='histSepalWidth':
        plotSingle_Hist(sepalWidthMatrix, label="sepal Width")
    elif sys.argv[2]=='histPetalLength':
        plotSingle_Hist(petalLengthMatrix, label="petal Length")
    elif sys.argv[2]=='histPetalWidth':
        plotSingle_Hist(petalWidthMatrix, label="petal Width")
    elif sys.argv[2]=='allHist':
        plotAllhist(sepalLengthMatrix, sepalWidthMatrix, petalLengthMatrix, petalWidthMatrix)

    if sys.argv[2]=='scatterSepalLength':
        plotSingleScatter(sepalLengthMatrix, [sepalWidthMatrix, petalLengthMatrix, petalWidthMatrix], "sepal Length", ["sepal Width", "petal Length", "petal Width"])
    elif sys.argv[2]=='scatterSepalWidth':
        plotSingleScatter(sepalWidthMatrix, [sepalLengthMatrix, petalLengthMatrix, petalWidthMatrix], "sepal Width", ["sepal Length", "petal Length", "petal Width"])
    elif sys.argv[2]=='scatterPetalLength':
        plotSingleScatter(petalLengthMatrix, [sepalWidthMatrix, sepalLengthMatrix, petalWidthMatrix], "petal Length", ["sepal Width", "sepal Length", "petal Width"])
    elif sys.argv[2]=='scatterPetalWidth':
        plotSingleScatter(petalWidthMatrix, [sepalWidthMatrix, petalLengthMatrix, sepalLengthMatrix], "petal Width", ["sepal Width", "petal Length", "sepal Length"])
    elif sys.argv[2]=='allScatter':
        plotAllScatter(sepalLengthMatrix, sepalWidthMatrix, petalLengthMatrix, petalWidthMatrix)

    print('Averages:')
    print('Average for sepal length: %d' % (computeAverage(sepalLengthMatrix)))
    print('Average for sepal width: %d' % (computeAverage(sepalWidthMatrix)))
    print('Average for petal length: %d' % (computeAverage(petalLengthMatrix)))
    print('Average for petal width: %d' % (computeAverage(petalWidthMatrix)))

    