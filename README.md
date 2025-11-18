# 441ProjectStockPrediction

This project demonstrates a pipeline that combines historical stock price data with daily news-headline sentiment to train a Hidden Markov Model (HMM) that infers hidden market states (e.g., bullish/bearish) and predicts short-term trends.# StockPredictionHMMs

This project demonstrates a pipeline that combines historical stock price data with daily news-headline sentiment to train a Hidden Markov Model (HMM) that infers hidden market states (e.g., bullish/bearish) and predicts short-term trends.

Features:
- Fetch stock data from Yahoo Finance (via `yfinance`).
- Fetch headlines via NewsAPI/Finnhub (optional) or fall back to local CSV/sample data.
- Compute daily sentiment scores using VADER.
- Align sentiment with price movements and train a Gaussian HMM (from `hmmlearn`).
- Visualize inferred hidden states over time.

Quick start
1. Create a Python environment and install the requirements:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. (Optional) Export a NewsAPI key if you want automatic headline fetching:

```powershell
$env:NEWSAPI_KEY = "your_key_here"
```

3. Run the demo pipeline:

```powershell
python run_pipeline.py --ticker AAPL --period 2y
```

Notes
- The code is written to fall back to a local/sample dataset when API keys or network are unavailable.
- See `tests/test_pipeline.py` for a small synthetic-data unit test that runs offline.
