from readCorpus.fileHandler import getFiles


def readText(directory, fileExtenstion = '.txt'):
    articleID = []
    text = []
    corpusFiles = getFiles(directory, fileExtenstion)

    articleStringIndex = corpusFiles[0].rfind('article') + 7

    for i, file in enumerate(corpusFiles):
        file = open(file, encoding='utf-8')
        for line in file:
            articleID.append(corpusFiles[i][articleStringIndex:-(len(fileExtenstion))])
            text.append(line[:-1])
    return { 'articleID' : articleID, 'text' : text }


def readLabels(directory, fileExtenstion = '.labels'):
    articleID = []
    label = []
    labelFiles = getFiles(directory, fileExtenstion)
    for i, file in enumerate(labelFiles):
        file = open(file)
        for line in file:
            line = line.split('\t')
            articleID.append(labelFiles[i][:-len(fileExtenstion)])
            label.append(line[-1][:-1])
    return { 'articleID' : articleID, 'label' : label}