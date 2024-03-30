import sys
import numpy
import matplotlib
import matplotlib.pyplot as plt


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