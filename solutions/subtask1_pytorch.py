from subtask1_pytorch_runScenario import runScenario



usingNewData = False

useGlove = False

# Training Parameters
trainingIterations = 10
batchSize = 64
learningRate = 1e-6
vocabSize = 25000

# Model Parameters
bidirectionalModel = False
cell = 'GRU'
embeddingDimensions = 100
hiddenDimensions = 20
dropout = 0.5
numOfLayers = 2


evaluateUnseenData = False
filesToEvaluate = '../datasets/2019/datasets-v3/datasets/test-articles'
evaluationFileName = 'LSTM-Bi-10'

saveModelWeights = True
modelWeightsFilename = '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}'.format(cell, bidirectionalModel, trainingIterations, useGlove, vocabSize, embeddingDimensions, hiddenDimensions, dropout, numOfLayers).replace('.', '')

scenario = {
    'usingNewData' : usingNewData,
    'useGlove' : useGlove,
    'trainingIterations' : trainingIterations,
    'batchSize' : batchSize,
    'learningRate' : learningRate,
    'bidirectionalModel' : bidirectionalModel,
    'vocabSize' : vocabSize,
    'cell' : cell,
    'embeddingDimensions' : embeddingDimensions,
    'hiddenDimensions' : hiddenDimensions,
    'dropout' : dropout,
    'numOfLayers' : numOfLayers,
    'evaluateUnseenData' : evaluateUnseenData,
    'filesToEvaluate' : filesToEvaluate,
    'evaluationFileName' : evaluationFileName,
    'saveModelWeights' : saveModelWeights,
    'modelWeightsFilename' : modelWeightsFilename
}

runScenario(scenario)