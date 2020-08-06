import readCorpus.binaryClassification as readCorpus

from helpers.createCSV import createTextLabelCSVs
from helpers.processTime import processTime
from pytorchFiles.modelSelection import defineForwardModel, defineBidirectionalModel
from pytorchFiles.subtask1.evaluate import unseenData
from pytorchFiles.tokenisers import defaultSpacyTokeniser
from pytorchFiles.training import train, testAccuracy
from sklearn.metrics import f1_score

import spacy
import torch
import torchtext

from nltk import word_tokenize
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def runScenario(scenario):
    initaliseCuda()
    TEXT, LABEL, trainIterator, testIterator = dataPreperation(scenario)
    model, optimiser = createComponents(scenario, TEXT)
    trainModel(model, optimiser, trainIterator, testIterator, scenario['trainingIterations'])

    if scenario['saveModelWeights']:
        saveModel(model, scenario['modelWeightsFilename'])

    if scenario['evaluateUnseenData']:
        evaluateUnseenData(scenario['filesToEvaluate'], model, TEXT, scenario['evaluationFileName'])


def initaliseCuda():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()
    torch.device('cuda')


def dataPreperation(scenario):
    if scenario['usingNewData']:
        createCSVs()

    TEXT, LABEL = tokenise(scenario['useGlove'])
    trainData, testData = getDatasets(TEXT, LABEL)

    buildVocabulary(
        scenario['useGlove'],
        TEXT,
        LABEL,
        trainData,
        testData,
        scenario['vocabSize']
    )
    trainIterator, testIterator = generateBatches(trainData, testData, scenario['batchSize'])

    return TEXT, LABEL, trainIterator, testIterator


def createComponents(scenario, TEXT):
    model = createModel(TEXT,
        scenario['bidirectionalModel'],
        scenario['cell'],
        scenario['embeddingDimensions'],
        scenario['hiddenDimensions'],
        scenario['numOfLayers'],
        scenario['dropout']
    )

    if (scenario['useGlove']):
        addAdditionalGloveTokens(model, TEXT, scenario['embeddingDimensions'])

    optimiser = createOptimiser(model, scenario['learningRate'])

    return model, optimiser


# Create CSVs that will be used to store the data for utilisation of
# torchtext.data.TabularDataset.
def createCSVs():
    print('Creating CSVs for new data. Unless you change the data you can now set the usingNewData flag to False.')
    trainingCorpusText = readCorpus.readText('../datasets/2019/datasets-v3/datasets/train-articles')
    trainingCorpusLabels = readCorpus.readLabels('../datasets/2019/datasets-v3/datasets/train-labels-SLC')
    createTextLabelCSVs(trainingCorpusText, trainingCorpusLabels, 'pytorchFiles/subtask1/datasets')


# Tokenise the data.
def tokenise(useGlove):
    print('\n\nTokenising data')
    if useGlove:
        TEXT = torchtext.data.Field(tokenize=defaultSpacyTokeniser)
    else:
        TEXT = torchtext.data.Field(tokenize=word_tokenize)
    LABEL = torchtext.data.LabelField(dtype=torch.float)

    return TEXT, LABEL


# Generate trainData and testData variables from the created CSVs.
def getDatasets(TEXT, LABEL):
    datafields = [('text', TEXT), ('label', LABEL)]

    trainData, testData = torchtext.data.TabularDataset.splits(
        path = 'pytorchFiles/subtask1/datasets',
        train = 'train.csv',
        test = 'test.csv',
        format = 'csv',
        skip_header = True,
        fields = datafields
    )

    return trainData, testData


# Build vocabulary.
def buildVocabulary(useGlove, TEXT, LABEL, trainData, testData, vocabSize):
    if useGlove:
        TEXT.build_vocab(
            trainData,
            max_size=vocabSize,
            vectors="glove.6B.100d",
            unk_init=torch.Tensor.normal_
        )
    else:
        TEXT.build_vocab(
            trainData,
            max_size=vocabSize
        )

    LABEL.build_vocab(trainData)


# Generate batches.
def generateBatches(trainData, testData, batchSize):
    print('Generating Batches')
    trainIterator, testIterator = torchtext.data.BucketIterator.splits(
        (trainData, testData),
        batch_size = batchSize,
        sort_key = lambda x: len(x.text),
        sort_within_batch = False,
        device=torch.device('cuda')
    )

    return trainIterator, testIterator


# Create model
def createModel(TEXT, bidirectionalModel, cell, embeddingDimensions, hiddenDimensions, numOfLayers, dropout):
    print('\nPreparing model')
    if bidirectionalModel:
        model = defineBidirectionalModel(cell, len(TEXT.vocab), embeddingDimensions, hiddenDimensions, numOfLayers, dropout)
    else:
        model = defineForwardModel(cell, len(TEXT.vocab), embeddingDimensions, hiddenDimensions, dropout)

    return model


# Add unk and pad tokens.
def addAdditionalGloveTokens(model, TEXT, embeddingDimensions):
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)

    unknown = TEXT.vocab.stoi[TEXT.unk_token]
    padding = TEXT.vocab.stoi[TEXT.pad_token]
    model.embedding.weight.data[unknown] = torch.zeros(embeddingDimensions)
    model.embedding.weight.data[padding] = torch.zeros(embeddingDimensions)


# Crate optimiser.
def createOptimiser(model, learningRate):
    optimiser = torch.optim.Adam(model.parameters(), lr = learningRate)
    return optimiser


# Train and evaluate model.
def trainModel(model, optimiser, trainIterator, testIterator, trainingIterations):
    print('\n\nTraining model')
    start = processTime()

    train(model, trainIterator, optimiser, trainingIterations)

    print('\nTime to train: {0}\n\n\n'.format(processTime(start)))


    print('Evaluating model accuracy\n\n')
    start = processTime()

    testAccuracy(model, testIterator)

    print('\nTime taken: {0}'.format(processTime(start)))


# Save model weights using Torch's save function
def saveModel(model, filename):
    directory = 'pytorchFiles/subtask1/modelWeights/{0}'.format(filename)
    torch.save(model.state_dict(), directory)
    print('Model weights saved to {0}'.format(directory))


# Produce evalaution file for data not used in model training or testing.
def evaluateUnseenData(filesToEvaluate, model, TEXT, filename):
    print('\n\nPerforming Evaluation')
    start = processTime()
    data = readCorpus.readText(filesToEvaluate)
    unseenData(model, TEXT.vocab, data, filename)
    print('Evaluation completed in: {0}'.format(processTime(start)))