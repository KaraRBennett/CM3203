from helpers.processTime import processTime

from pandas import DataFrame
from sklearn.metrics import f1_score


# Method to train a given model using K Folds
def trainModel(model, vectoriser, featureSelector, foldedTrain, foldedTest, test):
    start = processTime()

    foldedScore = []

    for i in range(len(foldedTrain['text'])):
        print('Performing folded train iteraition {0}'.format(i + 1))

        vectoriser, trainFeatures = generateTrainFeatures(vectoriser, foldedTrain['text'][i])
        testFeatures = generateTestFeatures(vectoriser, foldedTest['text'][i])

        if featureSelector:
            trainFeatures = featureSelector.fit_transform(trainFeatures, foldedTrain['labels'][i])
            testFeatures = featureSelector.transform(testFeatures)
        
        model = model.fit(trainFeatures, foldedTrain['labels'][i])
        testPrediction = model.predict(testFeatures)
        score = f1_score(foldedTest['labels'][i], testPrediction)
        foldedScore.append(score)

    
    print('\nTesting model against test set')
    testFeatures = generateTestFeatures(vectoriser, test['text'])

    if featureSelector:
        testFeatures = featureSelector.transform(testFeatures)

    testPrediction = model.predict(testFeatures)
    testScore = f1_score(test['labels'], testPrediction)

    print('\nTime to train: {0}\n\n'.format(processTime(start)))  
    return model, vectoriser, featureSelector, foldedScore, testScore, testPrediction


# Method that tunes an implemented GridSearchCV object (model) using K Folds
def tuneModel(model, vectoriser, featureSelector, trainCorpus, writeResults=True):
    print('Tuning model\n')
    start = processTime()

    vectoriser, trainFeatures = generateTrainFeatures(vectoriser, trainCorpus['text'])

    if featureSelector:
        trainFeatures = featureSelector.fit_transform(trainFeatures)

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


def generateTrainFeatures(vectoriser, transformObject):
    if vectoriser.__class__.__name__ == 'EmbeddingTransformer':
        for i in range(len(transformObject)):
            transformObject[i] = (transformObject[i]).lower()
        return vectoriser, vectoriser.transform(transformObject) 

    else:
        return vectoriser, vectoriser.fit_transform(transformObject)


def generateTestFeatures(vectoriser, transformObject):
    if vectoriser.__class__.__name__ == 'EmbeddingTransformer':
        for i in range(len(transformObject)):
            transformObject[i] = (transformObject[i]).lower()
    
    return vectoriser.transform(transformObject)