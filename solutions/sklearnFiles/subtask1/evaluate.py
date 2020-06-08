from translateLabels.translateBCLabels import Int2Word

def unseenData(model, vectoriser, data, outputFile='Evaluation'):
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

    file = open(f'results/subtask1/sklearn-{outputFile}.txt', 'w')
    file.writelines(output)
    file.close()
    print('\nEvaluation file sklearn-{0} can be found in results/subtask1/'.format(outputFile))