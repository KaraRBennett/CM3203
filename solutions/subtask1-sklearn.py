import sklearnFiles.subtask1.evaluate as evaluate
import sklearnFiles.models as models
import readCorpus.binaryClassification as readCorpus 
import sklearnFiles.training as training
import sklearnFiles.vectorisers as vectorisers

from translateLabels.translateBCLabels import Word2Int

import graphviz
import os

import pandas as pd 

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from statistics import mean
from time import process_time


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


# SET PARAMETERS

# Perform GridSearchCV
modelTuning = True

# Evaluate a dataset not used for training
evaluateUnseenData = False
filesToEvaluate = ('../datasets/2019/datasets-v2/datasets/dev-articles')

# If the model 
isTree = False
treeGraphFilename = 'decisionTreeGraph'

# Set the model parameters, the model to be used, and the vectorisation method
modelParameters = None
model = models.svc(modelTuning, modelParameters)
vectoriser = vectorisers.CountVectorizer()



# Get training data
print('Loading training data')

trainingCorpusText = readCorpus.readText('../datasets/2019/datasets-v2/datasets/train-articles')
trainingCorpusLabels = readCorpus.readLabels('../datasets/2019/datasets-v2/datasets/train-labels-SLC')
trainingCorpusLabels['label'] = Word2Int(trainingCorpusLabels['label'])


# Get test train split
print('Preparing test sets\n')

trainText, testText, trainLabels, testLabels = train_test_split(
    trainingCorpusText['text'],
    trainingCorpusLabels['label'],
    test_size = 0.25,
    random_state = 0
)


# Split test train into K Fold sets
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


testCorpus = {'text' : testText, 'labels' : testLabels}
    

# Train / Tune model
if modelTuning:
    training.tuneModel(model, vectoriser, foldedTrainingCorpus)

else:
    model, foldedScore, score = training.trainModel(model, vectoriser, foldedTrainingCorpus, foldedTestCorpus, testCorpus)

    # Print results
    for i, s in enumerate(foldedScore):
        print('F₁ Score for fold {0}: {1:.4f}'.format(i, s))
    print('Mean F₁ of all folds: {0:.4f}'.format(mean(foldedScore)))
    print('\nF₁ Score for test set: {0:.4f}'.format(score))


    # Perform evaluation on unseen data
    if evaluateUnseenData:
        print('Performing Evaluation')
        start = process_time()
        data = readCorpus.readText(filesToEvaluate)
        evaluate.unseenData(model, vectoriser, data)
        finish = process_time()
        print('Evaluation completed in: {0:.2f}'.format(finish-start))


    # Capture Decision Tree Graph
    if isTree:
        print('Generating Decision Tree Graph')
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