"""Demo runner for the pipeline.

Usage examples:
    python run_pipeline.py --ticker AAPL --period 1y
"""
import argparse
import os
from datetime import datetime
import pandas as pd
import numpy as np

from src.data_fetcher import fetch_stock_yfinance, fetch_headlines_newsapi, aggregate_headlines_by_date, fetch_crypto_coingecko
from src.sentiment import daily_sentiment_from_agg
from src.feature_engineering import compute_daily_returns, merge_price_sentiment, make_feature_matrix
from src.hmm_model import MarketHMM
from src.visualize import plot_price_states


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--period", default="1y")
    p.add_argument("--asset-type", choices=["stock", "crypto"], default="stock", help="Asset type: stock (default) or crypto")
    p.add_argument("--newsapi_key", default=os.environ.get("NEWSAPI_KEY"))
    p.add_argument("--save", default=None)
    args = p.parse_args()

    if args.asset_type == "stock":
        print(f"Fetching stock data for {args.ticker} period={args.period}...")
        try:
            df_prices = fetch_stock_yfinance(args.ticker, period=args.period)
        except Exception as e:
            print("Failed to fetch stock data:", e)
            return
    else:
        # crypto: use CoinGecko (period string like '1y' -> days)
        print(f"Fetching crypto data for {args.ticker} period={args.period} (CoinGecko)...")
        # convert period to days
        per = args.period
        days = 365
        try:
            if per.endswith('y'):
                years = int(per[:-1]) if per[:-1] else 1
                days = years * 365
            elif per.endswith('d'):
                days = int(per[:-1])
            else:
                days = int(per)
        except Exception:
            days = 365
        try:
            df_prices = fetch_crypto_coingecko(args.ticker, days=days)
        except Exception as e:
            print("Failed to fetch crypto data:", e)
            return

    start = df_prices.index.min().strftime("%Y-%m-%d")
    end = df_prices.index.max().strftime("%Y-%m-%d")
    print(f"Fetching headlines between {start} and {end} (query={args.ticker})...")
    df_headlines = fetch_headlines_newsapi(start, end, query=args.ticker, api_key=args.newsapi_key)
    df_agg = aggregate_headlines_by_date(df_headlines)
    df_sent = daily_sentiment_from_agg(df_agg)

    df_returns = compute_daily_returns(df_prices)
    # for crypto, compute rolling volatility and merge as extra feature
    if args.asset_type == "crypto":
        from src.feature_engineering import compute_rolling_volatility

        df_vol = compute_rolling_volatility(df_returns, window=7)
        # merge volatility into returns before merging sentiment
        df_returns = df_returns.join(df_vol, how="left")

    df_merged = merge_price_sentiment(df_returns, df_sent)
    X, idx = make_feature_matrix(df_merged)

    print("Fitting HMM...")
    model = MarketHMM(n_states=2)
    model.fit(X)
    states = model.predict_states(X)

    # Compute next-day probabilities and expected return from current (last) state
    current_state = int(states[-1])
    probs = model.next_state_probabilities(current_state)
    exp_ret = model.expected_next_return(current_state, return_index=0)
    # simple signal logic
    bull_state = int(np.argmax(model.means_[:, 0]) if getattr(model, "means_", None) is not None else 1)
    bull_prob = float(probs[bull_state])
    signal = "HOLD"
    if exp_ret > 0.001 or bull_prob > 0.6:
        signal = "BUY (likely rising)"
    elif exp_ret < -0.001 or bull_prob < 0.4:
        signal = "SELL (likely dropping)"
    summary = (
        f"Current state: {current_state}\n"
        f"Next-state probabilities: {probs.tolist()}\n"
        f"Expected next-day return (approx): {exp_ret:.4f}\n"
        f"Bull state index: {bull_state}, bull probability: {bull_prob:.2f}\n"
        f"Signal: {signal}\n"
    )
    print(summary)
    # save summary next to figure if requested
    if args.save:
        try:
            with open(args.save.replace('.png', '.txt'), 'w') as fh:
                fh.write(summary)
        except Exception:
            pass

    # attach states back to df_merged
    df_merged = df_merged.iloc[-len(states):].copy()
    df_merged["state"] = states

    print("Plotting results...")
    # normalize price index to dates for plotting/alignment
    df_plot_prices = df_prices.copy()
    df_plot_prices.index = pd.to_datetime(df_plot_prices.index).normalize()
    # select the same index span as df_merged
    try:
        df_plot_prices = df_plot_prices.loc[df_merged.index]
    except Exception:
        # if strict selection fails, align by taking the last N rows
        df_plot_prices = df_plot_prices.iloc[-len(states) :]
    fig = plot_price_states(df_plot_prices, states, title=f"{args.ticker} Close with inferred states")
    if args.save:
        fig.savefig(args.save)
        print("Saved figure to", args.save)
    else:
        try:
            import matplotlib.pyplot as plt

            plt.show()
        except Exception:
            print("Matplotlib interactive show not available; you can pass --save to write a PNG.")


if __name__ == "__main__":
    main()
