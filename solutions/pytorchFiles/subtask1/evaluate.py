from translateLabels.translateBCLabels import Int2Word

import torch

from nltk import word_tokenize 

def unseenData(model, vocab, data, outputFile='evaluation.txt'):
    output = []

    with torch.no_grad():

        currentArticleID = data['articleID'][0]
        sentenceNo = 0

        for i in range(len(data['text'])):

            if data['articleID'][i] == currentArticleID:
                sentenceNo += 1
            else:
                currentArticleID = data['articleID'][i]
                sentenceNo = 1
            
            outputString = '{0}\t{1}\t'.format(currentArticleID, sentenceNo)


            tokenised = word_tokenize(data['text'][i])
            tensor = torch.LongTensor([vocab[token] for token in tokenised])
            tensor = tensor.unsqueeze(1)
            tensor = tensor.to(torch.device('cuda'))

            if not len(tensor.data) == 0:
                prediction = torch.sigmoid(model(tensor))
                outputString += Int2Word(round(prediction.item())) + "\n"
            
            else:
                outputString += Int2Word(0) + "\n"

            output.append(outputString)

    print(len(output))

    if outputFile[-4:] != '.txt':
        outputFile += '.txt'
    file = open('results/subtask1/pytorch-' + outputFile, 'w')
    file.writelines(output)
    file.close() 