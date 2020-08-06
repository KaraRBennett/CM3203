from pandas import DataFrame
from random import sample, seed

def generateReport(predictions, expected, text, filename):
    correctPropaganda = []
    correctNonPropaganda = []
    incorrectPropaganda = []
    incorrectNonPropaganda = []

    for i in range(len(predictions)):
        if predictions[i] == 1:
            if expected[i] == 1:
                correctPropaganda.append(text[i])
            else:
                incorrectPropaganda.append(text[i])
        else:
            if expected[i] == 0:
                correctNonPropaganda.append(text[i])
            else:
                incorrectNonPropaganda.append(text[i])

    numberOfPropagandaSentences = len(correctPropaganda) + len(incorrectPropaganda)
    numberOfNonPropagandaSentences = len(correctNonPropaganda) + len(incorrectNonPropaganda)

    print('\nPercentage of corpus correctly labelled as propaganda:\t\t{0:.2f}%'.format(len(correctPropaganda)/numberOfPropagandaSentences*100))
    print('Percentage of corpus correctly labelled as non-propaganda:\t{0:.2f}%'.format(len(correctNonPropaganda)/numberOfNonPropagandaSentences*100))

    generateCSVFiles(correctPropaganda, correctNonPropaganda, incorrectPropaganda, incorrectNonPropaganda, filename)


def generateCSVFiles(correctPropaganda, correctNonPropaganda, incorrectPropaganda, incorrectNonPropaganda, filename):
    seed(0)

    if len(correctPropaganda) >= 50:
        correctPropaganda = sample(correctPropaganda, 50)
    if len(correctNonPropaganda) >= 50:
        correctNonPropaganda = sample(correctNonPropaganda, 50)
    if len(incorrectPropaganda) >= 50:
        incorrectPropaganda = sample(incorrectPropaganda, 50)
    if len(incorrectNonPropaganda) >= 50:
        incorrectNonPropaganda = sample(incorrectNonPropaganda, 50)

    samples = DataFrame({
        'Propaganda Sentences Correctly Labelled' : correctPropaganda,
        'Propaganda Sentences Incorrectly Labelled' : incorrectPropaganda,
        'Non-Propaganda Sentences Correctly Labelled' : correctNonPropaganda,
        'Non-Propaganda Sentences Incorrectly Labelled' : incorrectNonPropaganda
    })

    saveDestination = 'dataAnalysis/modelPerformanceEvaluations/{0}.csv'.format(filename)
    samples.to_csv(saveDestination)

    print('\nModel evaluation saved to {0}'.format(saveDestination))

