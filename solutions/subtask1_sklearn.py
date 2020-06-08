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


# tuneModel sets whether or not to use GridSearchCV to tune the model
# using the specific paramters selected for each model in models.py.
# To change what parameters are evaluated using GridSearchCV adusjt
# the code in sklearnFiles/models.py.

tuneModel = True


# model defines the model that will be training as part of the
# scenario. Avalible models can be viewed in sklearnFiles/models.py.
# If you wish to provide the model with custom parameters create a 
# dictionary of the desired paramteres using modelParamters.

modelParamateres = {'max_iter': 100}
model = models.svc(tuneModel, modelParamateres)


# vectoriser defines the type of vectorisation that will be used to
# numerically represent the textual data. Avalible vectorisers can be
# viewed in sklearnFiles/vectorisers.py

vectoriser = vectorisers.defaultCount() 


# evaluateUnseenData is a flag that when set to true will produce an 
# evaluation file named after this specific scenario in the results
# folder. evaluationTextFiles is a string that provides the location 
# of the directories holding all the text files to be evaluated.
# evaluationFilename is a string that will be used to name the
# produced file.

evaluateUnseenData = True
filesToEvaluate = '../datasets/2019/datasets-v3/datasets/train-articles'
evaluationFilename = 'evaluation'


# treeGraphFilename is a string that determines the name of the graph
# files that are automatically produced in the model being used is a 
# decision tree and tuneModel is false

treeGraphFilename = 'decisionTreeGraph'



##########################  Run Scenario  ###########################

scenario = {
    'trainingTextFiles' : trainingTextFiles,
    'trainingLabelFiles' : trainingLabelFiles,
    'tuneModel' : tuneModel,
    'modelParamters' : modelParamateres,
    'model' : model,
    'vectoriser' : vectoriser,
    'evaluateUnseenData' : evaluateUnseenData,
    'filesToEvaluate' : filesToEvaluate,
    'evaluationFilename' : evaluationFilename,
    'treeGraphFilename' : treeGraphFilename
}

runScenario(scenario)