from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from zeugma.embeddings import EmbeddingTransformer


def caseSensitiveCount():
    vectoriser = CountVectorizer(lowercase=False, ngram_range=(1, 2))
    return vectoriser


def defaultCount():
    vectoriser = CountVectorizer()
    return vectoriser


def defaultTfidf():
    vectoriser = TfidfVectorizer()
    return vectoriser


def gloveWordEmbeddings():
    vectoriser = EmbeddingTransformer('glove')
    return vectoriser