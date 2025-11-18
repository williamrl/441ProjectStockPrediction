"""Visualization utilities for plotting prices and inferred hidden states."""
import matplotlib.pyplot as plt
import pandas as pd


def plot_price_states(df_prices: pd.DataFrame, states: list[int], title: str = "Price with hidden states", savepath: str | None = None):
    """Plot closing price and color background by hidden state.

    df_prices: DataFrame indexed by Date with 'Close'.
    states: list/array of same length as df_prices (after alignment).
    """
    df = df_prices.copy()
    df = df.reset_index()
    df = df.iloc[-len(states) :].copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Close"], label="Close")
    # shade backgrounds per state
    prev = None
    start_idx = 0
    for i, s in enumerate(states):
        if prev is None:
            prev = s
            start_idx = i
            continue
        if s != prev:
            ax.axvspan(df.loc[start_idx, "Date"], df.loc[i - 1, "Date"], alpha=0.12, color=("green" if prev == 1 else "red"))
            prev = s
            start_idx = i
    # final span
    ax.axvspan(df.loc[start_idx, "Date"], df.loc[df.index[-1], "Date"], alpha=0.12, color=("green" if prev == 1 else "red"))
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    return fig
