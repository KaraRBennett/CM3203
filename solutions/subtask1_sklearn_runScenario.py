import sklearnFiles.subtask1.evaluate as evaluate
import readCorpus.binaryClassification as readCorpus 
import sklearnFiles.training as training

from translateLabels.translateBCLabels import Word2Int

import graphviz
import os

import pandas as pd 

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
from statistics import mean
from time import process_time



def runScenario(scenario):
    trainText, testText, trainLabels, testLabels = loadAndSplitData(scenario['trainingTextFiles'], scenario['trainingLabelFiles'])
    foldedTrainingCorpus, foldedTestCorpus = kFoldSplit(trainText, trainLabels)
    trainCorpus = {'text' : trainText, 'labels' : trainLabels}
    testCorpus = {'text' : testText, 'labels' : testLabels}

    if scenario['tuneModel'] == True:
        tune(scenario['model'], scenario['vectoriser'], trainCorpus)
    else:
        model, vectoriser = train(scenario['model'], scenario['vectoriser'], foldedTrainingCorpus, foldedTestCorpus, testCorpus)
    
        if scenario['evaluateUnseenData'] == True:
            evaluateUnseenData(model, vectoriser, scenario['filesToEvaluate'], scenario['evaluationFilename'])

        if type(model).__name__ == "DecisionTreeClassifier":
            captureDecisionTreeGraph(model, vectoriser, scenario['treeGraphFilename'])
    

def loadAndSplitData(textFilesDirectory, labelFilesDirectory):
    print('Loading training data')
    trainingCorpusText = readCorpus.readText(textFilesDirectory)
    trainingCorpusLabels = readCorpus.readLabels(labelFilesDirectory)
    trainingCorpusLabels['label'] = Word2Int(trainingCorpusLabels['label'])

    # Get test train split
    print('Preparing test sets\n')
    trainText, testText, trainLabels, testLabels = train_test_split(
        trainingCorpusText['text'],
        trainingCorpusLabels['label'],
        test_size = 0.25,
        random_state = 0
    )

    return trainText, testText, trainLabels, testLabels


def kFoldSplit(trainText, trainLabels):
    kFoldIndexes = []
    kf = KFold(n_splits = 5, random_state = 5, shuffle=True)
    for i in kf.split(trainText):
        kFoldIndexes.append(i)

    foldedTrainingCorpus = {'text' : [], 'labels' : []}
    foldedTestCorpus = {'text' : [], 'labels' : []}

    for i in range(len(kFoldIndexes)):
        foldedTrainingCorpus['text'].append([])
        foldedTrainingCorpus['labels'].append([])
        foldedTestCorpus['text'].append([])
        foldedTestCorpus['labels'].append([])
        
        for j in kFoldIndexes[i][0]:
            foldedTrainingCorpus['text'][i].append(trainText[j])
            foldedTrainingCorpus['labels'][i].append(trainLabels[j])

        for j in kFoldIndexes[i][1]:
            foldedTestCorpus['text'][i].append(trainText[j])
            foldedTestCorpus['labels'][i].append(trainLabels[j])

    return foldedTrainingCorpus, foldedTestCorpus


def tune(model, vectoriser, trainCorpus):
    training.tuneModel(model, vectoriser, trainCorpus)


def train(model, vectoriser, foldedTrainingCorpus, foldedTestCorpus, testCorpus):
    model, vectoriser, foldedScore, score = training.trainModel(model, vectoriser, foldedTrainingCorpus, foldedTestCorpus, testCorpus)

    # Print results
    for i, s in enumerate(foldedScore):
        print('F₁ Score for fold {0}: {1:.4f}'.format(i, s))
    print('\nMean F₁ of all folds: {0:.4f}'.format(mean(foldedScore)))
    print('\nF₁ Score for test set: {0:.4f}'.format(score))

    return model, vectoriser


def evaluateUnseenData(model, vectoriser, evaluationFiles, evaluationFilename):
    print('\n\nPerforming Evaluation')
    start = process_time()
    data = readCorpus.readText(evaluationFiles)
    evaluate.unseenData(model, vectoriser, data, evaluationFilename)
    finish = process_time()
    print('Evaluation completed in: {0:.2f}'.format(finish-start))


def captureDecisionTreeGraph(model, vectoriser, treeGraphFilename):
    print('Generating Decision Tree Graph')
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    treeData = export_graphviz(
        model,
        out_file=None,
        feature_names=vectoriser.get_feature_names(),
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(treeData)
    graph.render('results/subtask1/' + treeGraphFilename)