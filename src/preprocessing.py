import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for pkg in ["punkt_tab", "stopwords", "wordnet"]:
    nltk.download(pkg, quiet=True)

_stop = set(stopwords.words("english"))
_lem = WordNetLemmatizer()


def preprocess(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [_lem.lemmatize(w) for w in tokens if w not in _stop and len(w) > 1]
    return " ".join(tokens)
