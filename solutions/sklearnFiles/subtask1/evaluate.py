from translateLabels.translateBCLabels import Int2Word

def unseenData(model, vectoriser, data, outputFile='evaluation.txt'):
    output = []

    vectorisedData = vectoriser.transform(data['text'])

    for i in range(len(data['text'])):
        outputString = '{0}\t'.format(data['articleID'][i])
        result = model.predict(vectorisedData[i])
        outputString += Int2Word(result) + "\n"
        output.append(outputString)
    
    print(len(output))
    
    file = open(outputFile, 'w')
    file.writelines(output)
    file.close()