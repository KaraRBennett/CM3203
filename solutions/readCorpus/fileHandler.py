from os import listdir

def getFiles(directory, fileExtenstion):
    returnFiles = []
    files = listdir(directory)
    for file in files:
        if file[-len(fileExtenstion):] == fileExtenstion:
            if(directory):
                returnFiles.append(directory + '\\' + file)
            else:
                returnFiles.append(file)
    return returnFiles


def readArticle(filename): 
    article = []
    file = open(filename, encoding='utf-8')
    for line in file:
        article.append(line)
    return article