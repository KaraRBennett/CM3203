from translateLabels.translateBCLabels import Int2Word

def unseenData(model, vectoriser, data, outputFile='Evaluation'):
    performGlovePreprocessing(vectoriser, data)    
    features = vectoriser.transform(data['text'])
    predictions = model.predict(features)

    assertionError = 'Mismatched length of model predictions {0} and input data {1}'.format(len(predictions), len(data['text']))
    assert len(predictions) == len(data['text']), assertionError

    createOutputFile(data['articleID'], predictions, outputFile)
    

def performGlovePreprocessing(vectoriser, data):
    if vectoriser.__class__.__name__ == 'EmbeddingTransformer':
        for i in range(len(data['text'])):
            data['text'][i] = (data['text'][i]).lower()


def createOutputFile(dataIDs, predictions, outputFile):
    output = [] 
    currentID = dataIDs[0]
    sentenceNo = 0


    for i in range(len(predictions)):

        if dataIDs[i] == currentID:
            sentenceNo += 1
        else:
            currentID = dataIDs[i]
            sentenceNo = 1
        
        outputString = '{0}\t{1}\t{2}\n'.format(
            currentID,
            sentenceNo,
            Int2Word(predictions[i])
        )
        output.append(outputString)

    file = open(f'results/subtask1/sklearn-{outputFile}.txt', 'w')
    file.writelines(output)
    file.close()
    print('\nEvaluation file sklearn-{0} can be found in results/subtask1/'.format(outputFile))