import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(page_title="ML Portfolio Tracker",
                   page_icon="📈",
                   layout="wide")

st.title("ML Portfolio Tracker")

st.markdown("""
    <style>
    .about-box {
        padding: 1.2rem 1.4rem;
        border-radius: 0.6rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
        background-color: var(--background-color);
        color: var(--text-color);
    }

    .about-box h3 {
        margin-top: 0;
        margin-bottom: 0.7rem;
    }

    .about-box p {
        margin-bottom: 0.6rem;
    }

    @media (prefers-color-scheme: light) {
        .about-box {
            --background-color: #f5f7fa;
            --border-color: #d9dee7;
            --text-color: #1f2937;
        }
    }

    @media (prefers-color-scheme: dark) {
        .about-box {
            --background-color: #20242b;
            --border-color: #3a414c;
            --text-color: #f1f3f5;
        }
    }
    </style>

    <div class="about-box">
        <h3>About this tracker</h3>

        This dashboard tracks the live performance of this machine learning stock-selection strategy.

        Each week, fresh model predictions are generated and recorded at the time they are made. The portfolios and performance shown here are based solely 
        on those live predictions — they are not reconstructed from historical backtests.

        The controls let you explore how those same live predictions would have performed using different models, directions, and Top N selections.
    </div>
    """,
            unsafe_allow_html=True)

# ---------------------------------------------------------
# Find the master CSV
# ---------------------------------------------------------

WEBSITE_DIR = Path(__file__).resolve().parent
MASTER_FILE = WEBSITE_DIR / "Probabilities_All.csv"

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

if not MASTER_FILE.exists():
    st.error(f"Could not find the master file:\n{MASTER_FILE}")
    st.stop()

df = pd.read_csv(MASTER_FILE)
df["Date"] = pd.to_datetime(df["Date"])
max_N = df['Ticker'].nunique()

# ---------------------------------------------------------
# Define available models
# ---------------------------------------------------------

models = {
    "Ensemble": "Ensemble",
    "Random Forest": "RF",
    "XGBoost": "XGBoost",
    "LGBM": "LGBM",
    "CatBoost": "CatBoost"
}

# ---------------------------------------------------------
# Daily return columns
# ---------------------------------------------------------

daily_returns = {
    "thu_to_fri_return": "Thu → Fri",
    "fri_to_mon_return": "Fri → Mon",
    "mon_to_tue_return": "Mon → Tue",
    "tue_to_wed_return": "Tue → Wed",
    "wed_to_thu_return": "Wed → Thu"
}

# ---------------------------------------------------------
# Controls
# ---------------------------------------------------------

st.subheader("Portfolio Settings")

col1, col2, col3 = st.columns(3)

with col1:
    selected_model = st.selectbox("Model", list(models.keys()))

with col2:
    selected_direction = st.selectbox("Direction",
                                      ["Long", "Short", "Long/Short"])

with col3:
    top_n = st.number_input("Top N", min_value=1, max_value=max_N, value=3, step=1)

top_n = int(top_n)

# ---------------------------------------------------------
# Determine probability columns
# ---------------------------------------------------------

model_name = models[selected_model]

long_probability_column = (f"{model_name}_Probability_Long")

short_probability_column = (f"{model_name}_Probability_Short")

required_columns = []

if selected_direction == "Long":
    required_columns = [long_probability_column]

elif selected_direction == "Short":
    required_columns = [short_probability_column]

else:
    required_columns = [long_probability_column, short_probability_column]

for column in required_columns:
    if column not in df.columns:
        st.error(f"Could not find probability column: {column}")
        st.stop()

# ---------------------------------------------------------
# Check return columns
# ---------------------------------------------------------

if "weekly_return" not in df.columns:
    st.error("Could not find weekly return column: weekly_return")
    st.stop()

for column in daily_returns:

    if column not in df.columns:

        st.error(f"Could not find daily return column: {column}")

        st.stop()

# ---------------------------------------------------------
# Calculate completed weekly portfolio returns
# ---------------------------------------------------------

weekly_results = []

for prediction_date, week_df in df.groupby("Date"):
    week_df = week_df.copy()

    # -----------------------------------------------------
    # Select stocks for this prediction week
    # -----------------------------------------------------

    # Long-only
    if selected_direction == "Long":

        selected_stocks = (week_df.dropna(
            subset=[long_probability_column]).sort_values(
                long_probability_column, ascending=False).head(top_n).copy())

        if selected_stocks.empty:
            continue

        selected_stocks["Position Weight"] = (1 / len(selected_stocks))
        selected_stocks["Position Direction"] = ("Long")

    # -----------------------------------------------------
    # Short-only
    # -----------------------------------------------------

    elif selected_direction == "Short":

        selected_stocks = (week_df.dropna(
            subset=[short_probability_column]).sort_values(
                short_probability_column, ascending=True).head(top_n).copy())

        if selected_stocks.empty:
            continue

        selected_stocks["Position Weight"] = (1 / len(selected_stocks))
        selected_stocks["Position Direction"] = ("Short")

    # -----------------------------------------------------
    # Long/Short
    # -----------------------------------------------------

    else:
        long_stocks = (week_df.dropna(
            subset=[long_probability_column]).sort_values(
                long_probability_column, ascending=False).head(top_n).copy())

        short_stocks = (week_df.dropna(
            subset=[short_probability_column]).sort_values(
                short_probability_column, ascending=True).head(top_n).copy())

        if long_stocks.empty or short_stocks.empty:
            continue

        long_stocks["Position Weight"] = (
            1 / (len(long_stocks) + len(short_stocks)))

        short_stocks["Position Weight"] = (
            1 / (len(long_stocks) + len(short_stocks)))

        long_stocks["Position Direction"] = ("Long")
        short_stocks["Position Direction"] = ("Short")
        selected_stocks = pd.concat([long_stocks, short_stocks], ignore_index=True)

    # A weekly result is valid only when every selected holding has its
    # complete Thursday-to-Thursday return. This prevents holidays and
    # incomplete current weeks from being presented as a full weekly profit.
    if selected_stocks["weekly_return"].isna().any():
        continue

    # Long positions use the stock return directly; short positions reverse
    # it. Equal weights are assigned when the positions are selected above.
    position_returns = selected_stocks["weekly_return"].where(
        selected_stocks["Position Direction"] == "Long",
        -selected_stocks["weekly_return"],
    )

    portfolio_return = (
        position_returns * selected_stocks["Position Weight"]
    ).sum()

    weekly_results.append({
        "Prediction Date": prediction_date,
        "Return Date": prediction_date + pd.Timedelta(days=7),
        "Period": "Thu → Thu",
        "Portfolio Return": portfolio_return,
    })

results = pd.DataFrame(weekly_results)

if results.empty:
    st.warning("There are no completed weeks with available returns.")
    st.stop()

results = (results.sort_values("Return Date").reset_index(drop=True))

# ---------------------------------------------------------
# Calculate cumulative return
# ---------------------------------------------------------

results["Cumulative Return"] = ((1 + results["Portfolio Return"]).cumprod() -1)

# ---------------------------------------------------------
# Calculate drawdown
# ---------------------------------------------------------

wealth_index = (1 + results["Cumulative Return"])

running_max = wealth_index.cummax()

results["Drawdown"] = (wealth_index / running_max) - 1

# ---------------------------------------------------------
# Calculate statistics
# ---------------------------------------------------------

cumulative_return = (results["Cumulative Return"].iloc[-1])

portfolio_returns = results["Portfolio Return"]

average_weekly_return = portfolio_returns.mean()

win_rate = (portfolio_returns > 0).mean()

annualization_factor = 52 ** 0.5

return_volatility = portfolio_returns.std()

sharpe_ratio = (
    portfolio_returns.mean() / return_volatility * annualization_factor
    if return_volatility != 0
    else 0
)

downside_returns = portfolio_returns.clip(upper=0)
downside_deviation = (downside_returns.pow(2).mean()) ** 0.5

sortino_ratio = (
    portfolio_returns.mean() / downside_deviation * annualization_factor
    if downside_deviation != 0
    else 0
)

max_drawdown = (results["Drawdown"].min())

# ---------------------------------------------------------
# Display headline statistics
# ---------------------------------------------------------

st.subheader("Performance")

stat1, stat2, stat3, stat4, stat5, stat6 = st.columns(6)

with stat1:
    st.metric("Cumulative Return", f"{cumulative_return:.2%}")

with stat2:
    st.metric("Average Weekly", f"{average_weekly_return:.2%}")

with stat3:
    st.metric("Win Rate", f"{win_rate:.1%}")

with stat4:
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

with stat5:
    st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")

with stat6:
    st.metric("Max Drawdown", f"{max_drawdown:.2%}")

# ---------------------------------------------------------
# Cumulative return chart
# ---------------------------------------------------------

st.subheader("Cumulative Return")

# Add the initial 0% starting point
first_prediction_date = (results["Prediction Date"].min())

starting_point = pd.DataFrame({
    "Return Date": [first_prediction_date],
    "Cumulative Return": [0.0]
})

chart_data = pd.concat(
    [starting_point, results[["Return Date", "Cumulative Return"]]],
    ignore_index=True)

chart_data = (chart_data.drop_duplicates(
    subset=["Return Date"],
    keep="last").sort_values("Return Date").reset_index(drop=True))

# Create readable date labels
chart_data["Date Label"] = (chart_data["Return Date"].dt.strftime("%d/%m/%Y"))

# Plotly line chart
fig = px.line(chart_data,
              x="Date Label",
              y="Cumulative Return",
              markers=True,
              labels={
                  "Date Label": "Date",
                  "Cumulative Return": "Cumulative Return"
              })

fig.update_yaxes(tickformat=".1%")

fig.update_layout(hovermode="x unified", xaxis=dict(type="category"))

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Weekly portfolio returns chart
# ---------------------------------------------------------

st.subheader("Weekly Portfolio Returns")

weekly_chart = results[["Return Date", "Portfolio Return"]].copy()

weekly_chart["Date Label"] = (
    weekly_chart["Return Date"].dt.strftime("%d/%m/%Y"))

fig_weekly = px.bar(weekly_chart,
                    x="Date Label",
                    y="Portfolio Return",
                    labels={
                        "Date Label": "Week ending",
                        "Portfolio Return": "Weekly Return"
                    })

fig_weekly.update_yaxes(tickformat=".1%")

fig_weekly.update_layout(hovermode="x unified", xaxis=dict(type="category"))

st.plotly_chart(fig_weekly, use_container_width=True)

# ---------------------------------------------------------
# Current portfolio
# ---------------------------------------------------------

st.subheader("Current Portfolio")

latest_date = df["Date"].max()

current_df = df[df["Date"] == latest_date].copy()

st.caption(f"Based on predictions generated on "
           f"{latest_date.strftime('%Y-%m-%d')}")

# ---------------------------------------------------------
# Current Long-only
# ---------------------------------------------------------

if selected_direction == "Long":

    current_long = (current_df.dropna(
        subset=[long_probability_column]).sort_values(
            long_probability_column, ascending=False).head(top_n).copy())

    current_long["Direction"] = "Long"

    display_current = current_long[[
        "Ticker", long_probability_column, "Close", "Direction"
    ]].copy()

    display_current = display_current.rename(columns={
        long_probability_column: "Probability of Increase",
        "Close": "Price"
    })

# ---------------------------------------------------------
# Current Short-only
# ---------------------------------------------------------

elif selected_direction == "Short":

    current_short = (current_df.dropna(
        subset=[short_probability_column]).sort_values(
            short_probability_column, ascending=True).head(top_n).copy())

    current_short["Direction"] = "Short"

    display_current = current_short[[
        "Ticker", short_probability_column, "Close", "Direction"
    ]].copy()

    display_current = display_current.rename(columns={
        short_probability_column: "Probability of Increase",
        "Close": "Price"
    })

# ---------------------------------------------------------
# Current Long/Short
# ---------------------------------------------------------

else:

    current_long = (current_df.dropna(
        subset=[long_probability_column]).sort_values(
            long_probability_column, ascending=False).head(top_n).copy())

    current_long["Direction"] = "Long"

    current_short = (current_df.dropna(
        subset=[short_probability_column]).sort_values(
            short_probability_column, ascending=True).head(top_n).copy())

    current_short["Direction"] = "Short"

    display_current = pd.concat([
        current_long[[
            "Ticker", long_probability_column, "Close", "Direction"
        ]].rename(columns={
            long_probability_column: "Probability of Increase",
            "Close": "Price"
        }), current_short[[
            "Ticker", short_probability_column, "Close", "Direction"
        ]].rename(columns={
            short_probability_column: "Probability of Increase",
            "Close": "Price"
        })
    ],
                                ignore_index=True)

# ---------------------------------------------------------
# Display current positions
# ---------------------------------------------------------

st.dataframe(display_current, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Historical portfolio picks
# ---------------------------------------------------------

st.subheader("Historical Portfolio Picks")

st.caption("Expand a prediction week to see the stocks selected "
           "and their daily stock returns.")

completed_dates = (results[[
    "Prediction Date", "Return Date"
]].drop_duplicates(subset=["Prediction Date"]).sort_values(
    "Prediction Date", ascending=False).to_dict("records"))

for week in completed_dates:

    prediction_date = week["Prediction Date"]

    week_results = results[results["Prediction Date"] == prediction_date]

    week_total_return = week_results["Portfolio Return"].iloc[0]

    if selected_direction == "Long/Short":

        position_count = (f"{top_n} long + {top_n} short")

    else:

        position_count = (f"{top_n} stocks")

    expander_title = (f"{prediction_date.strftime('%Y-%m-%d')}  |  "
                      f"Weekly Profit: {week_total_return:.2%}  |  "
                      f"{position_count}")

    with st.expander(expander_title):

        historical_week = df[df["Date"] == prediction_date].copy()

        # -------------------------------------------------
        # Long
        # -------------------------------------------------

        if selected_direction == "Long":

            selected_week = (historical_week.dropna(
                subset=[long_probability_column]).sort_values(
                    long_probability_column,
                    ascending=False).head(top_n).copy())

            selected_week["Direction"] = "Long"

            selected_week["Probability of Increase"] = (
                selected_week[long_probability_column])

            selected_week["Weekly Profit"] = (selected_week["weekly_return"])

            display_week = selected_week[[
                "Ticker", "Probability of Increase", "Close", "Weekly Profit",
                *daily_returns.keys()
            ]].copy()

        # -------------------------------------------------
        # Short
        # -------------------------------------------------

        elif selected_direction == "Short":

            selected_week = (historical_week.dropna(
                subset=[short_probability_column]).sort_values(
                    short_probability_column,
                    ascending=True).head(top_n).copy())

            selected_week["Direction"] = "Short"

            selected_week["Probability of Increase"] = (
                selected_week[short_probability_column])

            selected_week["Weekly Profit"] = (-selected_week["weekly_return"])

            display_week = selected_week[[
                "Ticker", "Probability of Increase", "Close", "Weekly Profit",
                *daily_returns.keys()
            ]].copy()

        # -------------------------------------------------
        # Long/Short
        # -------------------------------------------------

        else:

            long_week = (historical_week.dropna(
                subset=[long_probability_column]).sort_values(
                    long_probability_column,
                    ascending=False).head(top_n).copy())

            long_week["Direction"] = "Long"

            long_week["Probability of Increase"] = (long_week[long_probability_column])

            long_week["Weekly Profit"] = (long_week["weekly_return"])

            short_week = (historical_week.dropna(
                subset=[short_probability_column]).sort_values(
                    short_probability_column,
                    ascending=True).head(top_n).copy())

            short_week["Direction"] = "Short"

            short_week["Probability of Increase"] = (short_week[short_probability_column])

            short_week["Weekly Profit"] = (-short_week["weekly_return"])

            st.markdown("### Long")

            long_display = long_week[[
                "Ticker", "Probability of Increase", "Close", "Weekly Profit",
                *daily_returns.keys()
            ]].copy()

            long_display = long_display.rename(columns=daily_returns)

            long_display["Probability of Increase"] = (
                long_display["Probability of Increase"].map(
                    lambda x: f"{x:.2%}"))

            long_display["Weekly Profit"] = (
                long_display["Weekly Profit"].map(lambda x: f"{x:.2%}"))

            for column in daily_returns.values():

                long_display[column] = (long_display[column].map(
                    lambda x: f"{x:.2%}" if pd.notna(x) else ""))

            st.dataframe(long_display,
                         use_container_width=True,
                         hide_index=True)

            st.markdown("### Short")

            short_display = short_week[[
                "Ticker", "Probability of Increase", "Close", "Weekly Profit",
                *daily_returns.keys()
            ]].copy()

            short_display = short_display.rename(columns=daily_returns)

            short_display["Probability of Increase"] = (
                short_display["Probability of Increase"].map(
                    lambda x: f"{x:.2%}"))

            short_display["Weekly Profit"] = (
                short_display["Weekly Profit"].map(lambda x: f"{x:.2%}"))

            for column in daily_returns.values():

                short_display[column] = (short_display[column].map(
                    lambda x: f"{x:.2%}" if pd.notna(x) else ""))

            st.dataframe(short_display,
                         use_container_width=True,
                         hide_index=True)

            continue

        # -------------------------------------------------
        # Format Long-only / Short-only
        # -------------------------------------------------

        display_week = display_week.rename(columns=daily_returns)

        display_week["Probability of Increase"] = (
            display_week["Probability of Increase"].map(lambda x: f"{x:.2%}"))

        display_week["Weekly Profit"] = (
            display_week["Weekly Profit"].map(lambda x: f"{x:.2%}"))

        for column in daily_returns.values():

            display_week[column] = (display_week[column].map(
                lambda x: f"{x:.2%}" if pd.notna(x) else ""))

        st.dataframe(display_week, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Weekly results table
# ---------------------------------------------------------

st.subheader("Weekly Portfolio Returns")

display_results = results.copy()

display_results["Prediction Date"] = (
    display_results["Prediction Date"].dt.strftime("%Y-%m-%d"))

display_results["Return Date"] = (
    display_results["Return Date"].dt.strftime("%Y-%m-%d"))

display_results["Portfolio Return"] = (
    display_results["Portfolio Return"].map(lambda x: f"{x:.2%}"))

display_results["Cumulative Return"] = (
    display_results["Cumulative Return"].map(lambda x: f"{x:.2%}"))

display_results["Drawdown"] = (
    display_results["Drawdown"].map(lambda x: f"{x:.2%}"))

display_results = display_results[[
    "Prediction Date", "Return Date", "Period", "Portfolio Return",
    "Cumulative Return", "Drawdown"
]]

st.dataframe(display_results, use_container_width=True, hide_index=True)
