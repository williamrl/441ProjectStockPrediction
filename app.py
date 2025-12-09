"""Flask web application for stock/crypto prediction using HMM."""
import os
import io
import base64
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from src.data_fetcher import (
    fetch_stock_yfinance,
    fetch_crypto_coingecko,
    fetch_headlines_newsapi,
    aggregate_headlines_by_date,
)
from src.sentiment import daily_sentiment_from_agg
from src.feature_engineering import (
    compute_daily_returns,
    compute_rolling_volatility,
    merge_price_sentiment,
    make_feature_matrix,
)
from src.hmm_model import MarketHMM
from src.visualize import plot_price_states

app = Flask(__name__)

# Store results in a session dict (in production, use a proper database)
results_cache = {}


def run_prediction(ticker, period, asset_type):
    """Run the full prediction pipeline and return results."""
    try:
        # Fetch price data
        if asset_type == "stock":
            df_prices = fetch_stock_yfinance(ticker, period=period)
        else:  # crypto
            days = 365
            if period.endswith('y'):
                years = int(period[:-1]) if period[:-1] else 1
                days = years * 365
            elif period.endswith('d'):
                days = int(period[:-1])
            df_prices = fetch_crypto_coingecko(ticker, days=days)

        # Fetch and process headlines
        start = df_prices.index.min().strftime("%Y-%m-%d")
        end = df_prices.index.max().strftime("%Y-%m-%d")
        api_key = os.environ.get("NEWSAPI_KEY")
        df_headlines = fetch_headlines_newsapi(start, end, query=ticker, api_key=api_key)
        df_agg = aggregate_headlines_by_date(df_headlines)
        df_sent = daily_sentiment_from_agg(df_agg)

        # Compute features
        df_returns = compute_daily_returns(df_prices)
        if asset_type == "crypto":
            df_vol = compute_rolling_volatility(df_returns, window=7)
            df_returns = df_returns.join(df_vol, how="left")

        df_merged = merge_price_sentiment(df_returns, df_sent)
        X, idx = make_feature_matrix(df_merged)

        # Train HMM
        model = MarketHMM(n_states=2)
        model.fit(X)
        states = model.predict_states(X)

        # Generate predictions
        current_state = int(states[-1])
        probs = model.next_state_probabilities(current_state)
        exp_ret = model.expected_next_return(current_state, return_index=0)
        bull_state = int(np.argmax(model.means_[:, 0]) if getattr(model, "means_", None) is not None else 1)
        bull_prob = float(probs[bull_state])

        # Determine signal
        signal = "HOLD"
        signal_color = "warning"
        if exp_ret > 0.001 or bull_prob > 0.6:
            signal = "BUY (likely rising)"
            signal_color = "success"
        elif exp_ret < -0.001 or bull_prob < 0.4:
            signal = "SELL (likely dropping)"
            signal_color = "danger"

        # Generate plot
        df_merged_states = df_merged.iloc[-len(states):].copy()
        df_merged_states["state"] = states
        
        df_plot_prices = df_prices.copy()
        df_plot_prices.index = pd.to_datetime(df_plot_prices.index).normalize()
        try:
            df_plot_prices = df_plot_prices.loc[df_merged_states.index]
        except Exception:
            df_plot_prices = df_plot_prices.iloc[-len(states):]

        fig = plot_price_states(df_plot_prices, states, title=f"{ticker} Price with Inferred States")
        
        # Convert plot to base64
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)

        # Current price
        current_price = float(df_prices["Close"].iloc[-1])
        price_change = float(df_returns["return"].iloc[-1] * 100) if "return" in df_returns.columns else 0

        return {
            "success": True,
            "ticker": ticker,
            "asset_type": asset_type,
            "current_price": current_price,
            "price_change": price_change,
            "current_state": current_state,
            "state_probabilities": [float(p) for p in probs],
            "expected_return": float(exp_ret),
            "bull_probability": bull_prob,
            "signal": signal,
            "signal_color": signal_color,
            "chart": f"data:image/png;base64,{img_base64}",
            "data_points": len(df_prices),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """API endpoint for predictions."""
    data = request.get_json()
    ticker = data.get("ticker", "").upper()
    period = data.get("period", "1y")
    asset_type = data.get("asset_type", "stock")

    if not ticker:
        return jsonify({"success": False, "error": "Ticker is required"}), 400

    # Run prediction
    result = run_prediction(ticker, period, asset_type)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
