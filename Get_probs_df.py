# ---------------------------------------------------------
# Build master live probabilities file + weekly returns
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import date, timedelta


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "Output"
WEBSITE_DIR = BASE_DIR / "Website"

MASTER_FILE = WEBSITE_DIR / "Probabilities_All.csv"


# ---------------------------------------------------------
# Your existing download function
# ---------------------------------------------------------

def batch_download(ticks, start, end, batch_size=200):
    out = []
    for i in range(0, len(ticks), batch_size):
        batch = ticks[i : i + batch_size]
        df = yf.download(batch, start=start, end=end, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=True)
        out.append(df)

    if not out:
        return pd.DataFrame()

    return pd.concat(out, axis=1)


# ---------------------------------------------------------
# 1. Find all weekly probability files
# ---------------------------------------------------------

probability_files = sorted(OUTPUT_DIR.glob("Probabilities_*.csv"))

# Only keep files that look like:
# Probabilities_2026-08-20.csv
probability_files = [
    f for f in probability_files
    if len(f.stem) == len("Probabilities_2026-08-20")]

if not probability_files:
    raise FileNotFoundError(
        f"No weekly probability files found in {OUTPUT_DIR}")

print(f"Found {len(probability_files)} probability files.")

# ---------------------------------------------------------
# 2. Read and concatenate all weekly files
# ---------------------------------------------------------

dfs = []

for file in probability_files:

    print(f"Reading: {file.name}")
    df = pd.read_csv(file)

    # Extract date from filename
    prediction_date = file.stem.replace("Probabilities_", "")
    df["Date"] = pd.to_datetime(prediction_date)
    dfs.append(df)


all_probs = pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------
# 3. Clean up
# ---------------------------------------------------------

all_probs["Date"] = pd.to_datetime(all_probs["Date"])

all_probs["Ticker"] = all_probs["Ticker"].astype(str).str.upper()

# Remove any accidental duplicate Date/Ticker rows
all_probs = (
    all_probs
    .drop_duplicates(subset=["Date", "Ticker"], keep="last")
    .sort_values(["Date", "Ticker"])
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# 4. Create weekly_return column
# ---------------------------------------------------------

if "weekly_return" not in all_probs.columns:
    all_probs["weekly_return"] = np.nan


# ---------------------------------------------------------
# 5. Identify prediction dates and tickers
# ---------------------------------------------------------

prediction_dates = sorted(all_probs["Date"].dropna().unique())

tickers = sorted(all_probs["Ticker"].dropna().unique())

print(f"Prediction dates: {len(prediction_dates)}")
print(f"Tickers: {len(tickers)}")


# ---------------------------------------------------------
# 6. Download price history
# ---------------------------------------------------------

min_date = pd.Timestamp(min(prediction_dates))

max_date = pd.Timestamp(max(prediction_dates))

# Need the following Thursday for the final prediction date.
# yfinance's end date is exclusive, so give it a little extra room.
download_start = min_date.date()

download_end = (max_date + timedelta(days=10)).date()

print()
print(f"Downloading prices from {download_start} to {download_end}...")
print()


prices = batch_download(
    tickers,
    start=download_start,
    end=download_end,
    batch_size=200
)

if prices.empty:
    raise RuntimeError("No price data was downloaded.")


# ---------------------------------------------------------
# 7. Extract Open prices into a simple table
# ---------------------------------------------------------

open_prices = pd.DataFrame(index=prices.index)

# batch_download with multiple tickers returns a MultiIndex.
# Expected structure:
#
# Ticker -> Open / High / Low / Close / ...
#
if isinstance(prices.columns, pd.MultiIndex):

    for ticker in tickers:

        if ticker in prices.columns.get_level_values(0):

            try:
                open_prices[ticker] = prices[ticker]["Open"]
            except KeyError:
                pass

        elif ticker in prices.columns.get_level_values(1):

            try:
                open_prices[ticker] = prices["Open"][ticker]
            except KeyError:
                pass

else:
    # Safety case if only one ticker was downloaded
    if "Open" in prices.columns:
        open_prices[tickers[0]] = prices["Open"]


open_prices.index = pd.to_datetime(open_prices.index).normalize()


# ---------------------------------------------------------
# 8. Calculate Thursday -> following Thursday return
# ---------------------------------------------------------

for prediction_date in prediction_dates:

    prediction_date = pd.Timestamp(prediction_date)

    # We only want Thursday predictions
    if prediction_date.weekday() != 3:
        print(
            f"WARNING: {prediction_date.date()} is not Thursday. "
            f"Skipping."
        )
        continue

    next_thursday = prediction_date + timedelta(days=7)

    # Need both dates in price history
    if prediction_date not in open_prices.index:
        print(
            f"WARNING: No open price for {prediction_date.date()}. "
            f"Skipping."
        )
        continue

    if next_thursday not in open_prices.index:
        print(
            f"No following Thursday price yet for "
            f"{prediction_date.date()} "
            f"(expected {next_thursday.date()}). "
            f"Leaving weekly_return blank."
        )
        continue

    current_week_rows = all_probs["Date"] == prediction_date

    current_tickers = all_probs.loc[
        current_week_rows, "Ticker"
    ].unique()

    for ticker in current_tickers:

        if ticker not in open_prices.columns:
            print(
                f"WARNING: No price data for {ticker}. "
                f"Leaving return blank."
            )
            continue

        entry_price = open_prices.at[
            prediction_date,
            ticker
        ]

        exit_price = open_prices.at[
            next_thursday,
            ticker
        ]

        # Skip missing prices
        if pd.isna(entry_price) or pd.isna(exit_price):
            continue

        # Avoid division by zero
        if entry_price == 0:
            continue

        weekly_return = (exit_price / entry_price) - 1

        row_mask = (
            (all_probs["Date"] == prediction_date)
            &
            (all_probs["Ticker"] == ticker)
        )

        all_probs.loc[row_mask, "weekly_return"] = weekly_return


# ---------------------------------------------------------
# 9. Save master file
# ---------------------------------------------------------

all_probs = (
    all_probs
    .sort_values(["Date", "Ticker"])
    .reset_index(drop=True)
)

all_probs.to_csv(
    MASTER_FILE,
    index=False
)

print()
print("=" * 60)
print("DONE")
print("=" * 60)
print(f"Saved: {MASTER_FILE}")
print(f"Rows: {len(all_probs):,}")
print(f"Columns: {len(all_probs.columns)}")

print()
print("Return coverage:")
print(
    all_probs["weekly_return"]
    .notna()
    .value_counts()
)