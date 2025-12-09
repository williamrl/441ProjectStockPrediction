// JavaScript for prediction app

async function runPrediction() {
    const ticker = document.getElementById('ticker').value.trim().toUpperCase();
    const period = document.getElementById('period').value;
    const assetType = document.getElementById('assetType').value;

    // Validate input
    if (!ticker) {
        showError('Please enter a ticker symbol');
        return;
    }

    // Show loading state
    hideAllStates();
    document.getElementById('loadingState').classList.remove('d-none');
    document.getElementById('predictBtn').disabled = true;
    document.getElementById('spinner').style.display = 'inline-block';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                ticker: ticker,
                period: period,
                asset_type: assetType,
            }),
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
            hideAllStates();
            document.getElementById('resultsState').classList.remove('d-none');
            document.getElementById('chartState').classList.remove('d-none');
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        showError(`Network error: ${error.message}`);
    } finally {
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('spinner').style.display = 'none';
    }
}

function displayResults(data) {
    // Update ticker title
    const assetTypeLabel = data.asset_type === 'crypto' ? 'Cryptocurrency' : 'Stock';
    document.getElementById('tickerTitle').textContent = `${data.ticker} - ${assetTypeLabel}`;

    // Update price info
    document.getElementById('currentPrice').textContent = `$${data.current_price.toFixed(2)}`;
    const priceChangeEl = document.getElementById('priceChange');
    const priceChangeText = data.price_change >= 0 
        ? `+${data.price_change.toFixed(2)}%` 
        : `${data.price_change.toFixed(2)}%`;
    priceChangeEl.textContent = priceChangeText;
    priceChangeEl.className = data.price_change >= 0 ? 'text-success' : 'text-danger';

    // Update data points
    document.getElementById('dataPoints').textContent = `Analyzed ${data.data_points} data points`;

    // Update signal
    const signalEl = document.getElementById('signal');
    signalEl.textContent = data.signal;
    signalEl.className = 'text-center mb-0 ' + (
        data.signal.includes('BUY') ? 'buy' :
        data.signal.includes('SELL') ? 'sell' :
        'hold'
    );

    // Update signal subtext
    const returnText = `Expected return: ${(data.expected_return * 100).toFixed(2)}%`;
    document.getElementById('signalSubtext').textContent = returnText;

    // Update model metrics
    document.getElementById('currentState').textContent = `State ${data.current_state}`;
    document.getElementById('bullProb').textContent = `${(data.bull_probability * 100).toFixed(1)}%`;
    
    const probText = data.state_probabilities
        .map((p, i) => `State ${i}: ${(p * 100).toFixed(1)}%`)
        .join(' → ');
    document.getElementById('stateProbs').textContent = probText;

    // Update chart
    document.getElementById('chart').src = data.chart;
}

function showError(message) {
    hideAllStates();
    const errorEl = document.getElementById('errorState');
    document.getElementById('errorMessage').textContent = message;
    errorEl.classList.remove('d-none');
}

function hideAllStates() {
    document.getElementById('loadingState').classList.add('d-none');
    document.getElementById('errorState').classList.add('d-none');
    document.getElementById('resultsState').classList.add('d-none');
    document.getElementById('chartState').classList.add('d-none');
}

// Allow Enter key to trigger prediction
document.getElementById('ticker')?.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        runPrediction();
    }
});

// Default: run prediction on page load with AAPL
window.addEventListener('load', function() {
    // Optional: uncomment to auto-run on load
    // runPrediction();
});
