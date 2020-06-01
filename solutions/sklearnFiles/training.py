#import scipy
from sklearn.metrics import f1_score
from datetime import date, datetime

    
def trainModel(model, vectoriser, foldedTrain, foldedTest, test):
    start = datetime.now()

    foldedScore = []

    for i in range(len(foldedTrain['text'])):
        print('Performing folded train iteration {0}'.format(i + 1))
        trainFeatures = vectoriser.fit_transform(foldedTrain['text'][i])
        testFeatures = vectoriser.transform(foldedTest['text'][i])
        model = model.fit(trainFeatures, foldedTrain['labels'][i])

        testPrediction = model.predict(testFeatures)
        score = f1_score(foldedTest['labels'][i], testPrediction)

        foldedScore.append(score)
    
    print('\nTesting model against test set')
    testFeatures = vectoriser.transform(test['text'])
    testPrediction = model.predict(testFeatures)
    testScore = f1_score(test['labels'], testPrediction)

    finish = datetime.now()
    duration = finish - start
    print('\nTime to train: {0:.2f}s\n\n'.format(duration.total_seconds()))
    
    return model, foldedScore, testScore


def tuneModel(model, vectoriser, foldedTrain):
    start = datetime.now()

    foldedMetrics = []

    for i in range(len(foldedTrain['text'])):
        print('\nPerforming folded train iteration {0}'.format(i + 1))
        trainFeatures = vectoriser.fit_transform(foldedTrain['text'][i])
        model = model.fit(trainFeatures, foldedTrain['labels'][i])

        foldedMetrics.append([])
        foldedMetrics[i].append(model.best_score_)
        foldedMetrics[i].append(model.best_params_)

        finish = datetime.now()
        duration = finish - start
        print('Current time elapsed: {0:.2f}'.format(duration.total_seconds()))
    
    finish = datetime.now()
    duration = finish - start
    print('\nTime to train: {0:.2f}s\n\n'.format(duration.total_seconds()))

    for i in range(len(foldedMetrics)):
        print('Iteration {0} Results\n-------------------'.format(i + 1))
        print('Best F₁ Score: {0:.4f}'.format(foldedMetrics[i][0]))
        
        print('Best Paramteres: -')
        for parameter in foldedMetrics[i][1]:
            print('* {0}: {1}'.format(parameter, foldedMetrics[i][1][parameter]))
        
        print('\n')