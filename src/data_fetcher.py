"""Data fetching utilities: stock prices and headlines.

Functions try to use online APIs (yfinance, NewsAPI) and fall back to local/sample data.
"""
from datetime import datetime, timedelta
import os
import pandas as pd


def fetch_stock_yfinance(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch daily stock data for `ticker` using yfinance. Returns DataFrame indexed by date with 'Close'.

    Falls back with a clear exception if yfinance is not installed.
    """
    try:
        import yfinance as yf
    except Exception as exc:
        raise RuntimeError("yfinance is required for online stock fetching") from exc

    tf = yf.Ticker(ticker)
    df = tf.history(period=period)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} period={period}")
    df = df[["Close"]].copy()
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "Date"
    return df


def load_sample_headlines() -> pd.DataFrame:
    """Return a small sample headlines DataFrame (Date, headline).

    This is used when no NewsAPI key or network is available.
    """
    sample = [
        (datetime.today().date() - timedelta(days=i), f"Sample headline {i} about market move")
        for i in range(30)
    ]
    df = pd.DataFrame(sample, columns=["Date", "headline"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df


def fetch_headlines_newsapi(start: str, end: str, query: str = "stock", api_key: str | None = None) -> pd.DataFrame:
    """Fetch headlines using NewsAPI.org. Returns DataFrame with Date and headline columns.

    If `api_key` is None or the request fails, returns sample headlines.
    """
    if api_key is None:
        return load_sample_headlines()

    try:
        import requests
    except Exception:
        return load_sample_headlines()


def fetch_crypto_coingecko(ticker: str, vs_currency: str = "usd", days: int = 365) -> pd.DataFrame:
    """Fetch historical price data for a crypto ticker using CoinGecko public API.

    ticker: like 'BTC-USD' or 'BTC'. We match the symbol to CoinGecko coin id.
    Returns DataFrame indexed by Date with 'Close'. Dates are normalized (UTC).
    """
    try:
        import requests
    except Exception:
        raise RuntimeError("requests is required to fetch CoinGecko data")

    # extract symbol (e.g., BTC from BTC-USD)
    symbol = ticker.split("-")[0].lower()

    # get coin list and find matching symbol
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/coins/list", timeout=10)
        resp.raise_for_status()
        coins = resp.json()
    except Exception as exc:
        raise RuntimeError("failed to fetch coin list from CoinGecko") from exc

    coin_id = None
    for c in coins:
        if c.get("symbol", "").lower() == symbol:
            coin_id = c.get("id")
            break
    if coin_id is None:
        raise RuntimeError(f"Could not map symbol {symbol} to a CoinGecko id")

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
        prices = j.get("prices", [])  # list of [ts, price]
        if not prices:
            raise RuntimeError("no price data returned from CoinGecko")
        # convert to DataFrame; timestamps are in ms
        rows = []
        for ts, price in prices:
            dt = pd.to_datetime(int(ts), unit="ms", utc=True).normalize()
            rows.append((dt, float(price)))
        df = pd.DataFrame(rows, columns=["Date", "Close"]).drop_duplicates(subset=["Date"]).set_index("Date")
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        return df
    except Exception as exc:
        raise RuntimeError("failed to fetch market chart from CoinGecko") from exc

    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "from": start, "to": end, "language": "en", "pageSize": 100}
    headers = {"Authorization": api_key}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        articles = j.get("articles", [])
        rows = []
        for a in articles:
            d = a.get("publishedAt")
            title = a.get("title") or ""
            if not d:
                continue
            dt = pd.to_datetime(d).normalize()
            rows.append((dt, title))
        df = pd.DataFrame(rows, columns=["Date", "headline"]).drop_duplicates()
        return df
    except Exception:
        return load_sample_headlines()


def aggregate_headlines_by_date(df_headlines: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple headlines per date into a single text per day.

    Returns DataFrame indexed by Date with a 'text' column.
    """
    if df_headlines.empty:
        return pd.DataFrame(columns=["Date", "text"]).set_index("Date")
    df = df_headlines.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    agg = df.groupby("Date").headline.apply(lambda s: " \n ".join(s.astype(str))).reset_index()
    agg = agg.set_index("Date")
    agg.columns = ["text"]
    return agg
