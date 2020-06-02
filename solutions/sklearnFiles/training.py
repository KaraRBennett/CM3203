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
    

    return model, foldedScore, testScore


# Method that tunes an implemented GridSearchCV object (model) using K Folds
def tuneModel(model, vectoriser, foldedTrain, writeResults=True):
    start = processTime()

    foldedMetrics = []

    # Train model on each K Fold
    for i in range(len(foldedTrain['text'])):
        print('\nPerforming folded train iteration {0}'.format(i + 1))
        trainFeatures = vectoriser.fit_transform(foldedTrain['text'][i])
        model = model.fit(trainFeatures, foldedTrain['labels'][i])

        foldedMetrics.append([])
        foldedMetrics[i].append(model.best_score_)
        foldedMetrics[i].append(model.best_params_)

        print('Current time elapsed: {0}'.format(processTime(start)))
    
    print('\nTime to train: {0}\n\n'.format(processTime(start)))


    # Print results summary to console
    for i in range(len(foldedMetrics)):
        print('Iteration {0} Results\n-------------------'.format(i + 1))
        print('Best F₁ Score: {0:.4f}'.format(foldedMetrics[i][0]))
        print('Best Estimator: {0}'.format(foldedMetrics[i]))
        
        print('Best Paramteres: -')
        for parameter in foldedMetrics[i][1]:
            print('* {0}: {1}'.format(parameter, foldedMetrics[i][1][parameter]))
        
        print('\n')

    
    # Write comprehensive results of final model to csv file
    if writeResults:
        filename = 'results/subtask1/GridSearch-' + model.estimator.__class__.__name__ + '.csv'
        filedata = DataFrame(data=model.cv_results_)
        filedata.to_csv(filename)
        print('Full results written to: ' + filename)