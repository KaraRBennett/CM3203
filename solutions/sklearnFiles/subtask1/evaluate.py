from translateLabels.translateBCLabels import Int2Word

def unseenData(model, vectoriser, data, outputFile='Evaluation.txt'):
    output = []

    vectorisedData = vectoriser.transform(data['text'])

    currentArticleID = data['articleID'][0]
    sentenceNo = 0

    for i in range(len(data['text'])):

        if data['articleID'][i] == currentArticleID:
            sentenceNo += 1
        else:
            currentArticleID = data['articleID'][i]
            sentenceNo = 1
        
        outputString = '{0}\t{1}\t{2}\n'.format(
            currentArticleID,
            sentenceNo,
            Int2Word(model.predict(vectorisedData[i]))
        )
        output.append(outputString)
    
    print(len(output))
    
    file = open('results/subtask1/sklearn-' + outputFile, 'w')
    file.writelines(output)
    file.close()