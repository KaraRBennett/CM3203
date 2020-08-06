import torch

def train(model, trainingIterator, optimiser, trainingIterations):
    for i in range(trainingIterations):
        trainingLoss, trainingAccuracy = trainIteration(model, trainingIterator, optimiser)
        print('Iteration: {0:02}\tTraining Loss: {1:.3f}\tTraining Accuracy: {2:.2f}%'.format(i+1, trainingLoss, trainingAccuracy))


def trainIteration(model, iterator, optimiser):
    iterationLoss = 0
    iterationAccuracy = 0

    model.train()

    for batch in iterator:
        if batch.text.size()[1] == 64: #not len(batch.text.data) == 0 or batch.text.:
            optimiser.zero_grad()
            predictions = model(batch.text).squeeze(1)
            lossCriterion = torch.nn.BCEWithLogitsLoss()
            loss = lossCriterion(predictions, batch.label)
            
            roundedPredictions = torch.round(torch.sigmoid(predictions))
            correct = (roundedPredictions == batch.label).float()
            accuracy = correct.sum() / len(correct)

            loss.backward()
            optimiser.step()

            iterationLoss += loss.item()
            iterationAccuracy += accuracy.item()
    
    return iterationLoss / len(iterator), iterationAccuracy / len(iterator)


def testAccuracy(model, iterator):
    model.eval()

    iterationLoss = 0
    iterationAccuracy = 0

    with torch.no_grad():
        
        for batch in iterator:
            if not len(batch.text.data) == 0:
                predictions = model(batch.text).squeeze(1)
                lossCriterion = torch.nn.BCEWithLogitsLoss()
                loss = lossCriterion(predictions, batch.label)

                roundedPredictions = torch.round(torch.sigmoid(predictions)) 
                correct = (roundedPredictions == batch.label).float()
                accuracy = correct.sum() / len(correct)

                iterationLoss += loss.item()
                iterationAccuracy += accuracy.item()
        
    testLoss = iterationLoss / len(iterator)
    testAccuracy = iterationAccuracy / len(iterator)

    print('\nTest Loss: {0:.3f}\tTest Accuracy: {1:.2f}%'.format(testLoss, testAccuracy*100))