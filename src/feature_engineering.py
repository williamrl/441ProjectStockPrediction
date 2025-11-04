"""Feature engineering: align price returns and daily sentiment.
"""
import pandas as pd
import numpy as np


def compute_daily_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    """Given price DataFrame indexed by Date with 'Close', compute daily returns.

    Returns DataFrame with 'return' column indexed by Date.
    """
    df = df_prices.copy()
    df.index = pd.to_datetime(df.index).normalize()
    df["return"] = df["Close"].pct_change()
    return df[["return"]]


def compute_rolling_volatility(df_returns: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Compute rolling volatility (std of returns) over `window` days. Returns DataFrame with 'vol'."""
    df = df_returns.copy()
    df["vol"] = df["return"].rolling(window=window, min_periods=1).std().fillna(0.0)
    return df[["vol"]]


def merge_price_sentiment(df_returns: pd.DataFrame, df_sentiment: pd.DataFrame) -> pd.DataFrame:
    """Merge returns and sentiment into one DataFrame.

    - Forward-fills sentiment where missing for trading days (assumes no news days still follow prior sentiment).
    - Drops NaN rows in returns (e.g., first row).
    """
    df1 = df_returns.copy()
    df2 = df_sentiment.copy()
    # ensure both indexes are timezone-naive to avoid join errors
    def _to_naive_index(idx):
        try:
            # pandas DatetimeIndex has tz attribute
            if getattr(idx, "tz", None) is not None:
                try:
                    return idx.tz_convert(None)
                except Exception:
                    return idx.tz_localize(None)
        except Exception:
            pass
        return pd.to_datetime(idx)

    df1.index = _to_naive_index(df1.index)
    df2.index = _to_naive_index(df2.index)
    df = df1.join(df2, how="left")
    # forward fill sentiment for trading days
    df["sentiment"] = df["sentiment"].ffill().fillna(0.0)
    df = df.dropna(subset=["return"]).copy()
    # Fill any remaining sentiment NaNs with 0
    df["sentiment"] = df["sentiment"].fillna(0.0)
    return df


def make_feature_matrix(df_merged: pd.DataFrame, feature_cols: list | None = None):
    """Return feature matrix X and index.

    If feature_cols is None, include 'return' and 'sentiment' if present, plus any additional numeric columns (e.g., 'vol').
    Returns (X, index)
    """
    df = df_merged.copy()
    if feature_cols is None:
        cols = []
        if "return" in df.columns:
            cols.append("return")
        if "sentiment" in df.columns:
            cols.append("sentiment")
        # include other numeric columns (like 'vol')
        for c in df.columns:
            if c not in cols and pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    else:
        cols = feature_cols

    X = df[cols].fillna(0.0).values
    return X, df.index
