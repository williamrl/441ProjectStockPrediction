import numpy as np
import pandas as pd

from src.feature_engineering import merge_price_sentiment, compute_daily_returns, make_feature_matrix
from src.hmm_model import MarketHMM


def test_feature_merge_and_hmm():
    # synthetic prices: random walk
    rng = np.random.RandomState(0)
    n = 100
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")
    prices = 100 + np.cumsum(rng.normal(scale=1.0, size=n))
    df_prices = pd.DataFrame({"Close": prices}, index=dates)

    # synthetic sentiment: correlated with price returns
    df_returns = compute_daily_returns(df_prices)
    # create sentiment that is positive when returns are positive
    sent = df_returns["return"].fillna(0.0).apply(lambda x: 0.5 if x > 0 else -0.5)
    df_sent = sent.to_frame()

    df_merged = merge_price_sentiment(df_returns, df_sent)
    X, idx = make_feature_matrix(df_merged)

    assert X.shape[0] == len(idx)
    assert X.shape[1] == 2

    model = MarketHMM(n_states=2, random_state=0)
    model.fit(X)
    states = model.predict_states(X)
    assert len(states) == X.shape[0]
