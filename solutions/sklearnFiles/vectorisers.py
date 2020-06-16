import sklearnFiles.stemVectorisers as sv

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from zeugma.embeddings import EmbeddingTransformer




# Count Vectorisers

def defaultCount():
    vectoiser = CountVectorizer(lowercase=False, stop_words='english', ngram_range=(1, 2))
    return vectoiser


# Tfidf Vectorisers

def defaultTfidf():
    vectoiser = TfidfVectorizer(lowercase=False, stop_words='english', ngram_range=(1, 2))
    return vectoiser


# Advanced Vectorisers

def decompositionVectorisor(vectoriser):
    decomposer = TruncatedSVD()
    preproccessor = StandardScaler()
    pipeline = make_pipeline(vectoriser, decomposer, preproccessor)
    return pipeline


def stemmedCount():
    vectoriser = sv.stemmedCount()
    return vectoriser


def stemmedTfidf():
    vectoiser = sv.stemmedTfidf()
    return vectoiser


def gloveWordEmbeddings():
    vectoriser = EmbeddingTransformer('glove')
    return vectoriser