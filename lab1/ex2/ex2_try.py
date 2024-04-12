import sys

class BusInfo:
    def __init__(self, busId, streetId, x, y, second):
        self.busId = busId
        self.streetId = streetId
        self.x = float(x)
        self.y = float(y)
        self.second = float(second)


def getBusesFromFile(argFile):
    busInfoList = []
    with open(argFile) as file:
        for line in file:
            busId, streetId, x, y, second = line.split()[0:5]
            busInfoRow = BusInfo(busId, streetId, x, y, second)
            busInfoList.append(busInfoRow)
    return busInfoList

def getTotalDistanceAndTime(busId, busInfoList):
    validRows = []
    for busRow in busInfoList:
        if(busRow.busId==busId):
            validRows.append(busRow)
    #I order for seconds -> I calculate the distance for each in the right order
    #order the list: at first the lower "second", then the later ones
    validRows.sort(key=lambda busInfo: busInfo.second)
    distance, totalTime, lastx, lasty, lastSecond = 0, 0, None, None, None
    for rowBus in validRows:
        if lastx!=None and lasty!=None and lastSecond!=None:
            distance += ((rowBus.x-lastx)**2 + (rowBus.y-lasty)**2)**(1/2)
            totalTime += rowBus.second-lastSecond
        lastx, lasty, lastSecond = rowBus.x, rowBus.y, rowBus.second
    return distance, totalTime

def getAverageSpeedOfBusLine(busLine, busInfoList):
    #avg speed = distance / seconds
    #get all valid lines and divide in different buses
    #busesResult['busId'] = [list of elements associated]
    busesGroupedByRow = {}
    for row in busInfoList:
        if row.streetId == busLine:
            if row.busId not in busesGroupedByRow:
                busesGroupedByRow[row.busId] = []
            busesGroupedByRow[row.busId].append(row)

    #for every bus list, call getTotalDistance
    metersSum, timeSum = 0, 0
    for busId, rowsOfTheSameBus in busesGroupedByRow.items():
        toAddMeters, toAddTime = getTotalDistanceAndTime(busId, rowsOfTheSameBus)
        metersSum += toAddMeters
        timeSum += toAddTime
    return metersSum / timeSum

if __name__ == '__main__':
    #read all the instances of the file and get a list
    allBusInfoList = getBusesFromFile(sys.argv[1])
    if sys.argv[2]=='-b':
        distance, totalTime = getTotalDistanceAndTime(sys.argv[3], allBusInfoList)
        print('%s - Total Distance: %.1f' % (sys.argv[3], distance))
    if sys.argv[2]=='-l':
        avg = getAverageSpeedOfBusLine(sys.argv[3], allBusInfoList)
        print('%s - Avg Speed: %f' % (sys.argv[3], avg))





    #if passed busId, return total distance (note: x and y; max and min)
    #if passed street, return avg speed using seconds and distances of every bus