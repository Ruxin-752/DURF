"""Text tokenization / preprocessing, ported from the paper (English only).

Faithful reproduction of the relevant helpers in
``science/observations/text_analysis.py``:

- ``limited_punc_tokenization``: split an utterance into phrases on
  ``! . , ; |`` so each phrase becomes its own Gaussian observation;
- ``preprocess_phrase``: the classifier-style normalization (tokenize, WordNet
  lemmatize, drop English stopwords, map single digits to words);
- ``nn_tokenize``: the neural-network tokenizer (``nn_preprocess_chat_phrase``)
  used to build the Route 2 vocabulary.

We keep the paper's rule for reference-vector normalization (sum to 1) in
``normalize_reference_vector`` so literal and pragmatic observations match the
original numerics.
"""

from __future__ import annotations

import re
import string
from functools import lru_cache

import numpy as np


_NUMBER_MAP = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def limited_punc_tokenization(text: str | None) -> list[str]:
    """Segment an utterance on ``! . , ; |`` into non-empty phrases."""

    if not text:
        return []
    phrases = re.split(r"[!.,;|]", str(text).strip())
    return [phrase.strip() for phrase in phrases if phrase.strip() != ""]


@lru_cache(maxsize=1)
def _lemmatizer():
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _english_stopwords() -> frozenset[str]:
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english"))


def _word_tokenize(text: str) -> list[str]:
    from nltk import word_tokenize

    return word_tokenize(text)


def preprocess_phrase(phrase: str | None) -> str:
    """Classifier-style preprocessing (``_preprocess_chat_phrase``)."""

    phrase = (phrase or "").translate(str.maketrans("", "", string.punctuation))
    tokens = _word_tokenize(phrase)
    lemmatizer = _lemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    stops = _english_stopwords()
    tokens = [token for token in tokens if token not in stops]
    tokens = [_NUMBER_MAP.get(token, token) for token in tokens]
    return " ".join(tokens)


def nn_tokenize(phrase: str | None) -> list[str]:
    """Neural-network tokenizer (``nn_preprocess_chat_phrase``)."""

    phrase = (phrase or "").replace("|", " ").replace("'", "")
    phrase = phrase.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    ).lower()
    tokens = _word_tokenize(phrase)
    lemmatizer = _lemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [_NUMBER_MAP.get(token, token) for token in tokens]
    if not tokens:
        tokens = [""]
    return tokens


def normalize_reference_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a reference vector to sum to 1 (paper convention).

    The paper's reference vectors are non-negative feature indicators/counts
    divided by their sum. If the vector is all zeros it is returned unchanged.
    """

    vector = np.asarray(vector, dtype=float)
    total = np.abs(vector).sum()
    if total == 0:
        return vector
    return vector / total
