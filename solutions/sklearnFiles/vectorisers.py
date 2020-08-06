from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from zeugma.embeddings import EmbeddingTransformer



# Basic Vectorisers

def defaultCount():
    vectoiser = CountVectorizer(lowercase=False, ngram_range=(1, 2))
    return vectoiser


def defaultTfidf():
    vectoiser = TfidfVectorizer(lowercase=False, ngram_range=(1, 2))
    return vectoiser


def gloveWordEmbeddings():
    vectoriser = EmbeddingTransformer('glove')
    return vectoriser


# Dimenionality Reducing Vectorisers

def decompositionVectoriser(vectoriser):
    decomposer = TruncatedSVD()
    preproccessor = StandardScaler()
    pipeline = make_pipeline(vectoriser, decomposer, preproccessor)
    return pipeline