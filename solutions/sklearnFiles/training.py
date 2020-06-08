from helpers.processTime import processTime

from pandas import DataFrame
from sklearn.metrics import f1_score

# Method to train a given model using K Folds
def trainModel(model, vectoriser, foldedTrain, foldedTest, test):
    start = processTime()

    foldedScore = []

    # Train model on each set of the K Fold and capture the F₁ score
    for i in range(len(foldedTrain['text'])):
        print('Performing folded train iteration {0}'.format(i + 1))
        trainFeatures = vectoriser.fit_transform(foldedTrain['text'][i])
        testFeatures = vectoriser.transform(foldedTest['text'][i])

        model = model.fit(trainFeatures, foldedTrain['labels'][i])

        testPrediction = model.predict(testFeatures)
        score = f1_score(foldedTest['labels'][i], testPrediction)

        foldedScore.append(score)
    
    # Print results from testing
    print('\nTesting model against test set')
    testFeatures = vectoriser.transform(test['text'])
    testPrediction = model.predict(testFeatures)
    testScore = f1_score(test['labels'], testPrediction)

    print('\nTime to train: {0}\n\n'.format(processTime(start)))
    

    return model, vectoriser, foldedScore, testScore


# Method that tunes an implemented GridSearchCV object (model) using K Folds
def tuneModel(model, vectoriser, trainCorpus, writeResults=True):
    print('Tuning model\n')
    start = processTime()

    trainFeatures = vectoriser.fit_transform(trainCorpus['text'])
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