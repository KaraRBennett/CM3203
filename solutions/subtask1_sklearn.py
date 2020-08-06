import sklearnFiles.models as models
import sklearnFiles.vectorisers as vectorisers

from subtask1_sklearn_runScenario import runScenario

# The following script is designed for creating and running a single
# scenario to either see the results of one of the scenarios in 
# subtask1_sklearn_allScenarios.py or to see the results of a unique
# scenario. Each of the below variables has specific instructions
# regarding use and acceptable values. Once you have altered the
# variables to create your desired scenario run this file to see the
# results.  



#########################  Build Scenario  ##########################

# trainingTextFiles and trainingLabelFiles are strings that provide 
# the locations of the directories holding all the text and label
# files that compose the training corpus

trainingTextFiles = '../datasets/2019/datasets-v3/datasets/train-articles'
trainingLabelFiles = '../datasets/2019/datasets-v3/datasets/train-labels-SLC'


# stemmingMethod is a string that determines if a stemming method
# should be utilised during data cleaning. Implemented stemming
# methods can be viewed in dataPreprocessing/cleanData.py. Set this 
# variable to None if you do not want to perform word stemming.

stemmingMethod = 'f'


# tuneModel sets whether or not to use GridSearchCV to tune the model
# using the specific parameters selected for each model in models.py.
# To change what parameters are evaluated using GridSearchCV adusjt
# the code in sklearnFiles/models.py.

tuneModel = False


# model defines the model that will be training as part of the
# scenario. Avalible models can be viewed in sklearnFiles/models.py.
# If you wish to provide the model with custom parameters create a 
# dictionary of the desired parametres using modelParamters.

modelParamateres = None
t = {
    'C':			0.1,
    'kernel':			'linear',
    'max_iter':		1000

}
model = models.nearestCentroid(tuneModel, modelParamateres)


# vectoriser defines the type of vectorisation that will be used to
# numerically represent the textual data. Available vectorisers can
# be viewed in sklearnFiles/vectorisers.py

vectoriser = vectorisers.defaultTfidf()


featureSelection = None
nFeatures = 20


# produceReport os a flag that when set to True will produce a csv
# file that contains up to 50 example sentences that were correctly
# labelled for both labels and 50 example sentences that were 
# incorrectly labelled for both both labels.

produceReport = False


# evaluateUnseenData is a flag that when set to True will produce an 
# evaluation file named after this specific scenario in the results
# folder. evaluationTextFiles is a string that provides the location 
# of the directories holding all the text files to be evaluated.
# evaluationFilename is a string that will be used to name the
# produced file.

evaluateUnseenData = False
filesToEvaluate = '../datasets/2019/datasets-v3/datasets/test-articles'
evaluationFilename = ''


# treeGraphFilename is a string that determines the name of the graph
# files that are automatically produced in the model being used is a 
# decision tree and tuneModel is false

treeGraphFilename = None# '3depthDecisionTreeGraph'



##########################  Run Scenario  ###########################

scenario = {
    'trainingTextFiles' : trainingTextFiles,
    'trainingLabelFiles' : trainingLabelFiles,
    'stemmingMethod' : stemmingMethod,
    'tuneModel' : tuneModel,
    'model' : model,
    'vectoriser' : vectoriser,
    'featureSelection' : featureSelection,
    'nFeatures' : nFeatures,
    'produceReport' : produceReport,
    'evaluateUnseenData' : evaluateUnseenData,
    'filesToEvaluate' : filesToEvaluate,
    'evaluationFilename' : evaluationFilename,
    'treeGraphFilename' : treeGraphFilename
}

runScenario(scenario)