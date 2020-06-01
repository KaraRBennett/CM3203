import readCorpus as rc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag

trainingCorpusText = rc.readCorpus('..\\2019\\datasets-v2\\datasets\\train-articles')
trainingCorpusLabels = rc.readCorpusLabels('..\\2019\\datasets-v2\\datasets\\train-labels-SLC')

print(len(trainingCorpusText['articleID']))
blankList = [0] * len(trainingCorpusText['articleID'])

datasetFeatures = {
    'articleID':    blankList.copy(),
    'text' :        blankList.copy(),
    'NoT':          blankList.copy(),
    'CC':           blankList.copy(),
    'CD' :          blankList.copy(),
    'DT' :          blankList.copy(),
    'EX' :          blankList.copy(),
    'FW' :          blankList.copy(),
    'IN' :          blankList.copy(),
    'JJ' :          blankList.copy(),
    'JJR' :         blankList.copy(),
    'JJS' :         blankList.copy(),
    'LS' :          blankList.copy(),
    'MD' :          blankList.copy(),
    'NN' :          blankList.copy(),
    'NNS' :         blankList.copy(),
    'NNP' :         blankList.copy(),
    'NNPS' :        blankList.copy(),
    'PDT' :         blankList.copy(),
    'POS' :         blankList.copy(),
    'PRP' :         blankList.copy(), 
    'PRP$' :        blankList.copy(),
    'RB' :          blankList.copy(),
    'RBR' :         blankList.copy(),
    'RBS' :         blankList.copy(),
    'RP' :          blankList.copy(),
    'TO' :          blankList.copy(),
    'UH' :          blankList.copy(),
    'VB' :          blankList.copy(),
    'VBD' :         blankList.copy(),
    'VBG' :         blankList.copy(),
    'VBN' :         blankList.copy(),
    'VBP' :         blankList.copy(),
    'VBZ' :         blankList.copy(),
    'WDT' :         blankList.copy(),
    'WP' :          blankList.copy(),
    'WP$' :         blankList.copy(),
    'WRB' :         blankList.copy(),
    'NONE' :        blankList.copy(),
    'label':        blankList.copy()
}


tokeniser = RegexpTokenizer(r'\w+')

for i in range(len(trainingCorpusText['articleID'])):
    text = trainingCorpusText['text'][i]
    tokens = tokeniser.tokenize(text)
    tagger = Counter([j for i, j in pos_tag(tokens)])

    datasetFeatures['articleID'][i] = trainingCorpusText['articleID'][i]
    datasetFeatures['text'][i] = trainingCorpusText['articleID'][i]
    datasetFeatures['NoT'][i] = tokens
    datasetFeatures['label'][i] = trainingCorpusLabels['label'][i]

    for key in tagger:
        if key in datasetFeatures:
            datasetFeatures[key][i] = tagger[key]
        else:
            print('Invalid key: {0}\nFound in: {1}'.format(key, tagger))

dataset = pd.DataFrame.from_dict(datasetFeatures)

dataset.head(20)

    




