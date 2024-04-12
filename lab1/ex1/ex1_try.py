import sys

class Competitor:
    def __init__(self, name, surname, country, scores):
        self.name = name
        self.surname = surname
        self.country = country
        print(scores)
        self.scores = scores
        self.totalScore = self.compute_total_score(scores)

    def compute_total_score(self, scores):
        #score in the class
        #get all the scores:
        allScores = scores[:]
        #first remove max and min
        allScores.pop(allScores.index(min(allScores)))
        allScores.pop(allScores.index(max(allScores)))
        sum = 0.0
        for validScore in allScores:
            print(validScore)
            sum += float(validScore)
        return sum
            

#need now:
# file read
# printBest3
# printBestCountry


def getCompetitorListFromFile(file):
    competitorList = []
    # with open(argPassed) as f:
    for line in file:
            print('%s, %s' % (line[0], line[1]))
            name, surname, country = line.split()[0:3]
            scores = line.split()[3:]
            competitor = Competitor(name, surname, country, scores)
            competitorList.append(competitor)
    return competitorList


def getBest3CompetitorsFromList(competitorsList):
    competitorsList.sort(key=lambda competitor: -competitor.totalScore)
    return competitorsList[0:3]

def getBestCountryFromList(competitorsList):
    countryList = {}
    for comp in competitorsList:
        if comp.country not in countryList:
            countryList[comp.country] = comp.totalScore
        else:
            countryList[comp.country] += comp.totalScore
    
    maxScore = max(countryList.values())
    bestCountry = None
    for country, score in countryList.items():
        if score==maxScore:
            bestCountry = country
            break
    print('MAX: %s' % (maxScore))
    return {"score": maxScore, "countryName": bestCountry}


if __name__ == '__main__':
    #read line by line
    competitorList = None
    with open(sys.argv[1]) as f:
        competitorList = getCompetitorListFromFile(f)
    bestCompetitors = getBest3CompetitorsFromList(competitorList)
    print('Final ranking: ')
    i = 1
    for comp in bestCompetitors:
        print('%d : %s %s - Score: %.1f' % (i, comp.name, comp.surname, comp.totalScore) )
        i+=1
    bestCountry = getBestCountryFromList(competitorList)
    print('Best Country: ')
    print('%s - Total score: %.1f' % (bestCountry['countryName'], bestCountry['score']))




