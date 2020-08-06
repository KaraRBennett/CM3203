from pandas import DataFrame
from sklearn.model_selection import train_test_split


def createTextLabelCSVs(text, label, targetDirectory, testSize = 0.25):
    print('Creating CSVs')

    trainText, testText, trainLabels, testLabels = train_test_split(
        text,
        label,
        test_size = testSize,
        random_state = 0
    )

    trainData = convertTextLabelstoDataFrame(trainText, trainLabels)
    testData = convertTextLabelstoDataFrame(testText, testLabels)

    trainData.to_csv(targetDirectory + '/train.csv')
    testData.to_csv(targetDirectory + '/test.csv')
    print('CSVs created\n')


def convertTextLabelstoDataFrame(text, labels):
    returnArray = []
    for i in range(len(text)):
        returnArray.append( [text[i], labels[i]] )
    
    return DataFrame(returnArray)
