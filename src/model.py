"""Topic model and sentiment layer for the CBA Google Play reviews.

Turns raw review text into the two inputs the tracking module needs:

    tp       (review x topic) topical prevalence matrix, each row summing to 1
    periods  the sampling period each review belongs to

plus a VADER sentiment score per review, which is the extension the source
paper names as missing.

Choices worth knowing about
---------------------------
NMF on TF-IDF, not LDA. Half of these reviews are under ten words. LDA needs
enough words per document to estimate a mixture, and on very short text it
smears every review across every topic. NMF copes better with short text and
its document weights are non-negative, so normalising each row gives a valid
prevalence matrix.

Sentiment words are removed from the topic vocabulary. Left in, the model
spends half its topics on "great", "easy", "love" and "useless", which are
reactions, not quality determinants. Taking them out makes the topics about
*what* people discuss, and the sentiment layer then says *how they feel* about
it. That is the attention-versus-satisfaction split from the README, applied
at the modelling stage.

One model over the whole corpus. Topic numbers from separate fits are not
comparable, so every period is scored by the same fitted model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

N_TOPICS = 10
MIN_WORDS = 5
SEED = 0

# Domain words that appear everywhere and so separate nothing.
_DOMAIN = {
    "app", "commbank", "cba", "bank", "banking", "commonwealth", "comm",
}

# Reactions rather than subjects. Handled by the sentiment layer instead.
_REACTIONS = {
    "good", "great", "easy", "love", "best", "excellent", "awesome", "nice",
    "thanks", "thank", "amazing", "perfect", "fantastic", "brilliant", "happy",
    "helpful", "useful", "handy", "simple", "convenient", "friendly", "user",
    "terrible", "useless", "bad", "worst", "horrible", "poor", "hate",
    "annoying", "better",
}

# Filler that survives the standard English list in informal review text.
_FILLER = {
    "just", "use", "using", "get", "like", "really", "would", "also", "one",
    "even", "dont", "can", "im", "ive", "ve", "don", "ever", "far", "always",
    "never", "used", "apps", "please", "fix", "work", "works", "working",
    "doesn", "does", "doesnt", "need", "needs", "way", "time", "new", "pretty",
    "much", "well", "still", "keep", "keeps", "make", "makes", "every",
    "thing", "things", "want", "go", "got", "say", "could", "know", "think",
    "see", "lot", "day", "days", "now", "already", "since", "ok", "okay",
    "give", "back", "first", "try", "tried", "problem", "problems", "issue",
    "issues",
}

STOP_WORDS = sorted(ENGLISH_STOP_WORDS | _DOMAIN | _REACTIONS | _FILLER)

# Hand labels, read off each topic's top terms (outputs/topic_terms.csv).
# NMF does not return topics in a fixed order, so a label is attached to the
# topic whose top terms contain its anchor word, never to a topic number.
# If a refit moves the topics around, labelling fails loudly instead of
# quietly putting "Cardless cash" on a topic about login.
# Another analyst could word these differently: the labels are a judgement,
# the term lists are the evidence.
TOPIC_ANCHORS = {
    "tap": "Tap & pay (Android NFC)",
    "latest": "Updates breaking the app",
    "install": "Phone compatibility & install",
    "fingerprint": "Login (fingerprint, PIN)",
    "customer": "Customer service & staff",
    "details": "Account access & details",
    "transfer": "Money transfers",
    "credit": "Cards, cardless cash & notifications",
    "widget": "Quick balance & home screen",
    "won let": "App won't open or let me in",
}


def load_reviews(path) -> pd.DataFrame:
    """Read the collected reviews and keep the ones long enough to model."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["text"] = df["text"].fillna("")
    df = df[df["text"].str.split().str.len() >= MIN_WORDS]
    return df.drop_duplicates("review_id").reset_index(drop=True)


def fit_topics(texts: pd.Series, n_topics: int = N_TOPICS):
    """Fit one NMF model over the corpus.

    Returns (weights, model, vectorizer). ``weights`` is the raw NMF document
    matrix, not yet normalised: some rows are all zero because every word in
    the review was filtered out, and the caller decides what to do with them.
    """
    vec = TfidfVectorizer(
        stop_words=STOP_WORDS,
        min_df=10,
        max_df=0.4,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vec.fit_transform(texts)
    model = NMF(n_components=n_topics, init="nndsvd", random_state=SEED, max_iter=600)
    weights = model.fit_transform(X)
    return weights, model, vec


def top_terms(model, vec, n: int = 12) -> list[list[str]]:
    terms = np.array(vec.get_feature_names_out())
    return [list(terms[c.argsort()[::-1][:n]]) for c in model.components_]


def label_topics(terms_per_topic: list[list[str]]) -> list[str]:
    """Attach each hand label to the one topic whose top terms hold its anchor."""
    labels = [None] * len(terms_per_topic)
    for anchor, label in TOPIC_ANCHORS.items():
        hits = [i for i, t in enumerate(terms_per_topic) if anchor in t]
        if len(hits) != 1:
            raise ValueError(
                f"anchor {anchor!r} matched topics {hits}; the topics have "
                "shifted, so re-read topic_terms.csv and update TOPIC_ANCHORS"
            )
        if labels[hits[0]] is not None:
            raise ValueError(f"topic {hits[0]} matched two anchors")
        labels[hits[0]] = label
    return [l or f"topic_{i}" for i, l in enumerate(labels)]


def to_prevalence(weights: np.ndarray, labels: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Normalise NMF weights into a topical prevalence matrix.

    Returns the matrix for rows with any topic weight, and the boolean mask of
    which rows those were, so the caller can align the other columns.
    """
    totals = weights.sum(axis=1)
    keep = totals > 0
    tp = weights[keep] / totals[keep, None]
    return pd.DataFrame(tp, columns=labels), keep


def sentiment(texts: pd.Series) -> np.ndarray:
    """VADER compound score per review, from -1 (negative) to +1 (positive)."""
    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(t)["compound"] for t in texts])


def topic_sentiment(tp: pd.DataFrame, scores: np.ndarray, periods=None) -> pd.DataFrame | pd.Series:
    """Prevalence-weighted mean sentiment per topic (optionally per period).

    A review that is 80% about login and 20% about transfers contributes 80%
    of its sentiment to login. Without ``periods`` returns one value per topic.
    """
    weighted = tp.mul(scores, axis=0)
    if periods is None:
        return weighted.sum() / tp.sum()
    key = pd.Series(list(periods), index=tp.index, name="period")
    return weighted.groupby(key).sum() / tp.groupby(key).sum()
