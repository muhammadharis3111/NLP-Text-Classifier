import numpy as np
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfWrapper:
    def __init__(self, max_features=10000):
        self.vec = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, texts):
        return self.vec.fit_transform(texts)

    def transform(self, texts):
        return self.vec.transform(texts)


class EmbeddingVectorizer:
    """Mean-pooling vectorizer for Word2Vec / GloVe keyed-vector models."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.kv = None

    def _load(self):
        if self.kv is None:
            self.kv = api.load(self.model_name)
        self.dim = self.kv.vector_size

    def _doc_vec(self, text):
        words = [w for w in text.split() if w in self.kv]
        if not words:
            return np.zeros(self.dim)
        return np.mean(self.kv[words], axis=0)

    def fit_transform(self, texts):
        self._load()
        return np.array([self._doc_vec(t) for t in texts])

    def transform(self, texts):
        self._load()
        return np.array([self._doc_vec(t) for t in texts])


def get_vectorizer(name):
    if name == "tfidf":
        return TfidfWrapper()
    elif name == "word2vec":
        return EmbeddingVectorizer("glove-twitter-200")
    elif name == "glove":
        return EmbeddingVectorizer("glove-wiki-gigaword-100")
    raise ValueError(f"Unknown vectorizer: {name}")
