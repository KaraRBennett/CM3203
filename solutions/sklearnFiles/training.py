from helpers.processTime import processTime

from pandas import DataFrame
from sklearn.metrics import f1_score


# Method to train a given model using K Folds
def trainModel(model, vectoriser, foldedTrain, foldedTest, test):
    start = processTime()

    foldedScore = []

    # Train model on each set of the K Fold and capture the F₁ score
    if vectoriser.__class__.__name__ == 'EmbeddingTransformer':
        trainWithEmbeddingTransformer(model, vectoriser, foldedTrain, foldedTest, foldedScore)
    else:
        trainWithSklearnVectoriser(model, vectoriser, foldedTrain, foldedTest, foldedScore)
    
    # Print results from testing
    print('\nTesting model against test set')
    testFeatures = vectoriser.transform(test['text'])
    testPrediction = model.predict(testFeatures)
    testScore = f1_score(test['labels'], testPrediction)

    print('\nTime to train: {0}\n\n'.format(processTime(start)))
    

    return model, vectoriser, foldedScore, testScore


def trainWithSklearnVectoriser(model, vectoriser, foldedTrain, foldedTest, foldedScore):
    for i in range(len(foldedTrain['text'])):
        print('Performing folded train iteration {0}'.format(i + 1))
        trainFeatures = vectoriser.fit_transform(foldedTrain['text'][i])
        testFeatures = vectoriser.transform(foldedTest['text'][i])

        model = model.fit(trainFeatures, foldedTrain['labels'][i])

        testPrediction = model.predict(testFeatures)
        score = f1_score(foldedTest['labels'][i], testPrediction)

        foldedScore.append(score)


def trainWithEmbeddingTransformer(model, vectoriser, foldedTrain, foldedTest, foldedScore):
    for i in range(len(foldedTrain['text'])):
        print('Performing folded train iteration {0}'.format(i + 1))

        for j in range(len(foldedTrain['text'][i])):
            foldedTrain['text'][i][j] = (foldedTrain['text'][i][j]).lower()
      
        for j in range(len(foldedTest['text'][i])):
            foldedTest['text'][i][j] = (foldedTest['text'][i][j]).lower()
        
        trainFeatures = vectoriser.transform(foldedTrain['text'][i])
        testFeatures = vectoriser.transform(foldedTest['text'][i])

        model = model.fit(trainFeatures, foldedTrain['labels'][i])
        testPrediction = model.predict(testFeatures)

        score = f1_score(foldedTest['labels'][i], testPrediction)
        foldedScore.append(score)   


# Method that tunes an implemented GridSearchCV object (model) using K Folds
def tuneModel(model, vectoriser, trainCorpus, writeResults=True):
    print('Tuning model\n')
    start = processTime()

    trainFeatures = selectTransformMethod(vectoriser, trainCorpus['text'])
    model = model.fit(trainFeatures, trainCorpus['labels'])
    
    print('\nTuning complete')
    print('Time to tune: {0}\n\n'.format(processTime(start)))


    # Print results summary to console
    print('Best F₁ Score: {0:.4f}'.format(model.best_score_))
        
    print('Best Paramteres: -')
    bestParameters = model.best_params_
    for parameter in bestParameters:
        print('* {0}: {1}'.format(parameter, bestParameters[parameter]))
    print('\n')
    
    # Write comprehensive results of final model to csv file
    if writeResults:
        filename = 'results/subtask1/GridSearch-{0}-{1}.csv'.format(model.estimator.__class__.__name__, type(vectoriser).__name__)
        filedata = DataFrame(data=model.cv_results_)
        filedata.to_csv(filename)
        print('Full results written to: ' + filename)


def selectTransformMethod(vectoriser, transformObject):
    if vectoriser.__class__.__name__ == 'EmbeddingTransformer':
        for i in range(len(transformObject)):
            transformObject[i] = (transformObject[i]).lower()
        return vectoriser.transform(transformObject) 

    else:
        return vectoriser.fit_transform(transformObject)
