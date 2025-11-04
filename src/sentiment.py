"""Sentiment scoring utilities.

Primary: VADER (if installed). Fallback: simple rule-based score.
"""
from typing import Iterable
import pandas as pd


def vader_score_texts(texts: Iterable[str]) -> list[float]:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception:
        raise RuntimeError("vaderSentiment not available")
    an = SentimentIntensityAnalyzer()
    return [an.polarity_scores(str(t))["compound"] for t in texts]


def simple_lexicon_score(texts: Iterable[str]) -> list[float]:
    """Simple fallback sentiment scorer using keyword matches. Returns values in [-1,1]."""
    pos_words = {"up", "gain", "positive", "beat", "surge", "rise", "bull"}
    neg_words = {"down", "miss", "drop", "decline", "loss", "bear", "fall"}
    out = []
    for t in texts:
        s = str(t).lower()
        score = 0
        for w in pos_words:
            if w in s:
                score += 1
        for w in neg_words:
            if w in s:
                score -= 1
        # normalize
        if score > 0:
            val = min(1.0, score / 3.0)
        elif score < 0:
            val = max(-1.0, score / 3.0)
        else:
            val = 0.0
        out.append(val)
    return out


def daily_sentiment_from_agg(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Given an aggregated headlines DataFrame with index Date and column 'text', return DataFrame with 'sentiment'.

    Uses VADER if available, else a simple lexicon.
    """
    df = df_agg.copy()
    if df.empty:
        return pd.DataFrame(columns=["Date", "sentiment"]).set_index("Date")
    texts = df["text"].astype(str).tolist()
    try:
        scores = vader_score_texts(texts)
    except Exception:
        scores = simple_lexicon_score(texts)
    df["sentiment"] = scores
    return df[["sentiment"]]
