import readCorpus.binaryClassification as readCorpus

from helpers.processTime import processTime
from pytorchFiles.training import train, testAccuracy
from pytorchFiles.subtask1.evaluate import unseenData
from pytorchFiles.model import RNN

import torch
import torchtext

from nltk import word_tokenize
from pandas import DataFrame
from sklearn.model_selection import train_test_split


#Set torch to use GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')
cuda = torch.device('cuda:0')
torch.cuda.empty_cache()


#
evaluateUnseenData = True
filesToEvaluate = '../datasets/2019/datasets-v2/datasets/train-articles'
trainingIterations = 10


#Set to True on first run to create necessary data files
createCSVs = False

if createCSVs:
    print('Creating data files')    
    trainingCorpusText = readCorpus.readText('../datasets/2019/datasets-v2/datasets/train-articles')
    trainingCorpusLabels = readCorpus.readLabels('../datasets/2019/datasets-v2/datasets/train-labels-SLC')

    trainText, testText, trainLabels, testLabels = train_test_split(
        trainingCorpusText['text'],
        trainingCorpusLabels['label'],
        test_size = 0.25,
        random_state = 0
    )

    trainData = []
    for i in range(len(trainText)):
        trainData.append( [trainText[i], trainLabels[i]] )
    trainData = DataFrame(trainData)

    testData = []
    for i in range(len(testText)):
        testData.append( [testText[i], testLabels[i]] )
    testData = DataFrame(testData)

    trainData.to_csv('pytorchFiles/subtask1/datasets/train.csv', index=False)
    testData.to_csv('pytorchFiles/subtask1/datasets/test.csv', index=False)
    print('Data files created\n')


print('Processing data\n\n')
TEXT = torchtext.data.Field(tokenize=word_tokenize)
LABEL = torchtext.data.LabelField(dtype=torch.float)

datafields = [('text', TEXT), ('label', LABEL)]

trainData, testData = torchtext.data.TabularDataset.splits(
    path = 'pytorchFiles/subtask1/datasets',
    train = 'train.csv',
    test = 'test.csv',
    format = 'csv',
    skip_header = True,
    fields = datafields
)

'''
print(f'Train Examples: {len(train)}')
print(f'Test Examples: {len(test)}')
print(vars(train[5]))
'''

TEXT.build_vocab(trainData, max_size=10500)
LABEL.build_vocab(trainData)

'''
print(f'TEXT: {len(TEXT.vocab)}')
print(f'LABEL: {len(LABEL.vocab)}')

print(TEXT.vocab.freqs.most_common(50))
print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)
'''


trainIterator, testIterator = torchtext.data.BucketIterator.splits(
    (trainData, testData),
    batch_size = 256,
    sort_key = lambda x: len(x.text),
    sort_within_batch = False,
    device=cuda
)


model = RNN(
    inputDimensions =  len(TEXT.vocab),
    embeddingDimensions = 100,
    hiddenDimensions = 256,
    outputDimensions = 1
)
model=torch.nn.DataParallel(model, device_ids=[0])

optimiser = torch.optim.Adam(model.parameters(), lr = 1e-6)
lossFunction = torch.nn.BCEWithLogitsLoss()

print('Training model\n\n')
start = processTime()

train(model, trainIterator, optimiser, lossFunction, trainingIterations)

print('\nTime to train: {0}\n\n\n'.format(processTime(start)))

print('Evaluating model\n\n')
start = processTime()

testAccuracy(model, testIterator, lossFunction)

print('\nTime to evaluate: {0}'.format(processTime(start)))



if evaluateUnseenData == True:
    print('Performing Evaluation')
    start = processTime()
    data = readCorpus.readText(filesToEvaluate)
    unseenData(model, TEXT.vocab, data)
    print('Evaluation completed in: {0}'.format(processTime(start)))