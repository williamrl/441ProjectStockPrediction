# Flask Web GUI for Stock Prediction

A simple web interface for the HMM stock/crypto prediction pipeline.

## Setup

1. Install Flask (already added to requirements.txt):
```bash
pip install flask
```

## Running the App

Start the Flask development server:

```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

## Features

- **Ticker Input**: Enter any stock (AAPL, MSFT, etc.) or crypto symbol (BTC, ETH, etc.)
- **Asset Type Selection**: Choose between stock or cryptocurrency
- **Time Period Selection**: Select 3 months, 6 months, 1 year, 2 years, or 5 years of data
- **Real-time Analysis**: 
  - Fetches historical prices
  - Fetches related news headlines
  - Analyzes sentiment
  - Trains Hidden Markov Model
  - Generates predictions and visualizations

## Output

The app displays:
- **Current Price**: Latest market price
- **Daily Change**: Percentage change from previous day
- **Market Signal**: BUY, SELL, or HOLD recommendation
- **Model Metrics**: Current state, bull probability, state transitions
- **Price Chart**: Visualization showing inferred market states over time

## Environment Variables

(Optional) Set your NewsAPI key for better headline fetching:

```powershell
$env:NEWSAPI_KEY = "your_api_key_here"
```

Without an API key, the app will use sample headlines (though still fully functional).

## API Endpoint

**POST** `/api/predict`

Request body:
```json
{
  "ticker": "AAPL",
  "period": "1y",
  "asset_type": "stock"
}
```

Response:
```json
{
  "success": true,
  "ticker": "AAPL",
  "current_price": 150.25,
  "price_change": 2.5,
  "signal": "BUY (likely rising)",
  "chart": "data:image/png;base64,..."
}
```

## Troubleshooting

- **Port 5000 already in use?** Change the port in `app.py`: `app.run(port=5001)`
- **SSL errors?** They're harmless for local development
- **Slow first run?** Downloading a year of price data takes time
