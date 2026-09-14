import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="ML Portfolio Tracker",
    page_icon="📈",
    layout="wide"
)

st.title("ML Portfolio Tracker")

```python
st.markdown(
    """
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

        <p>
        This dashboard tracks the <strong>live performance</strong> of our
        machine learning stock-selection strategy.
        </p>

        <p>
        Each week, fresh model predictions are generated and recorded at the
        time they are made. The portfolios and performance shown here are
        based solely on those <strong>live predictions</strong> — they are not
        reconstructed from historical backtests.
        </p>

        <p>
        The controls let you explore how those same live predictions would have
        performed using different models, directions, and Top N selections.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
```

# ---------------------------------------------------------
# Find the master CSV
# ---------------------------------------------------------

WEBSITE_DIR = Path(__file__).resolve().parent

MASTER_FILE = WEBSITE_DIR / "Probabilities_All.csv"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

if not MASTER_FILE.exists():

    st.error(
        f"Could not find the master file:\n{MASTER_FILE}"
    )

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

    selected_model = st.selectbox(
        "Model",
        list(models.keys())
    )


with col2:

    selected_direction = st.selectbox(
        "Direction",
        [ "Long/Short", "Long", "Short"]
    )


with col3:

    top_n = st.number_input(
        "Top N",
        min_value=1,
        max_value=max_N,
        value=3,
        step=1
    )

top_n = int(top_n)


# ---------------------------------------------------------
# Determine probability columns
# ---------------------------------------------------------

model_name = models[selected_model]

long_probability_column = (
    f"{model_name}_Probability_Long"
)

short_probability_column = (
    f"{model_name}_Probability_Short"
)


required_columns = []

if selected_direction == "Long":

    required_columns = [
        long_probability_column
    ]

elif selected_direction == "Short":

    required_columns = [
        short_probability_column
    ]

else:

    required_columns = [
        long_probability_column,
        short_probability_column
    ]


for column in required_columns:

    if column not in df.columns:

        st.error(
            f"Could not find probability column: {column}"
        )

        st.stop()


# ---------------------------------------------------------
# Check daily return columns
# ---------------------------------------------------------

for column in daily_returns:

    if column not in df.columns:

        st.error(
            f"Could not find daily return column: {column}"
        )

        st.stop()


# ---------------------------------------------------------
# Calculate daily portfolio returns
# ---------------------------------------------------------

daily_results = []


for prediction_date, week_df in df.groupby("Date"):

    week_df = week_df.copy()


    # -----------------------------------------------------
    # Select stocks for this prediction week
    # -----------------------------------------------------

    # Long-only
    if selected_direction == "Long":

        selected_stocks = (
            week_df
            .dropna(subset=[long_probability_column])
            .sort_values(
                long_probability_column,
                ascending=False
            )
            .head(top_n)
            .copy()
        )

        if selected_stocks.empty:
            continue


        selected_stocks["Position Weight"] = (
            1 / len(selected_stocks)
        )

        selected_stocks["Position Direction"] = (
            "Long"
        )


    # -----------------------------------------------------
    # Short-only
    # -----------------------------------------------------

    elif selected_direction == "Short":

        selected_stocks = (
            week_df
            .dropna(subset=[short_probability_column])
            .sort_values(
                short_probability_column,
                ascending=True
            )
            .head(top_n)
            .copy()
        )

        if selected_stocks.empty:
            continue


        selected_stocks["Position Weight"] = (
            1 / len(selected_stocks)
        )

        selected_stocks["Position Direction"] = (
            "Short"
        )


    # -----------------------------------------------------
    # Long/Short
    # -----------------------------------------------------

    else:

        long_stocks = (
            week_df
            .dropna(subset=[long_probability_column])
            .sort_values(
                long_probability_column,
                ascending=False
            )
            .head(top_n)
            .copy()
        )


        short_stocks = (
            week_df
            .dropna(subset=[short_probability_column])
            .sort_values(
                short_probability_column,
                ascending=True
            )
            .head(top_n)
            .copy()
        )


        if long_stocks.empty or short_stocks.empty:
            continue


        long_stocks["Position Weight"] = (
            1 / (len(long_stocks) + len(short_stocks))
        )

        short_stocks["Position Weight"] = (
            1 / (len(long_stocks) + len(short_stocks))
        )


        long_stocks["Position Direction"] = (
            "Long"
        )

        short_stocks["Position Direction"] = (
            "Short"
        )


        selected_stocks = pd.concat(
            [
                long_stocks,
                short_stocks
            ],
            ignore_index=True
        )


    # -----------------------------------------------------
    # Calculate each day's portfolio return
    # -----------------------------------------------------

    for return_column, return_label in daily_returns.items():

        available_returns = selected_stocks[
            return_column
        ].dropna()


        if available_returns.empty:
            continue


        # Long positions use the stock return directly.
        #
        # Short positions reverse the stock return.
        #
        # Position weights are equal across the portfolio.

        position_returns = selected_stocks[
            return_column
        ].copy()


        adjusted_returns = []

        for _, row in selected_stocks.iterrows():

            stock_return = row[return_column]

            if pd.isna(stock_return):
                continue


            if row["Position Direction"] == "Long":

                adjusted_return = stock_return

            else:

                adjusted_return = -stock_return


            adjusted_returns.append(
                adjusted_return
                * row["Position Weight"]
            )


        if not adjusted_returns:
            continue


        portfolio_return = sum(
            adjusted_returns
        )


        # -------------------------------------------------
        # Determine actual calendar date of this return
        # -------------------------------------------------

        if return_column == "thu_to_fri_return":

            return_date = (
                prediction_date
                + pd.Timedelta(days=1)
            )

        elif return_column == "fri_to_mon_return":

            return_date = (
                prediction_date
                + pd.Timedelta(days=4)
            )

        elif return_column == "mon_to_tue_return":

            return_date = (
                prediction_date
                + pd.Timedelta(days=5)
            )

        elif return_column == "tue_to_wed_return":

            return_date = (
                prediction_date
                + pd.Timedelta(days=6)
            )

        elif return_column == "wed_to_thu_return":

            return_date = (
                prediction_date
                + pd.Timedelta(days=7)
            )


        daily_results.append({
            "Prediction Date": prediction_date,
            "Return Date": return_date,
            "Period": return_label,
            "Portfolio Return": portfolio_return
        })


# ---------------------------------------------------------
# Create daily results dataframe
# ---------------------------------------------------------

results = pd.DataFrame(daily_results)


if results.empty:

    st.warning(
        "There are no completed periods with available returns."
    )

    st.stop()


results = (
    results
    .sort_values("Return Date")
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# Calculate cumulative return
# ---------------------------------------------------------

results["Cumulative Return"] = (
    (1 + results["Portfolio Return"])
    .cumprod()
    - 1
)


# ---------------------------------------------------------
# Calculate drawdown
# ---------------------------------------------------------

wealth_index = (
    1 + results["Cumulative Return"]
)

running_max = wealth_index.cummax()

results["Drawdown"] = (
    wealth_index / running_max
) - 1


# ---------------------------------------------------------
# Calculate statistics
# ---------------------------------------------------------

cumulative_return = (
    results["Cumulative Return"].iloc[-1]
)


average_daily_return = (
    results["Portfolio Return"].mean()
)


win_rate = (
    results["Portfolio Return"] > 0
).mean()


best_day = (
    results["Portfolio Return"].max()
)


worst_day = (
    results["Portfolio Return"].min()
)


max_drawdown = (
    results["Drawdown"].min()
)


# ---------------------------------------------------------
# Display headline statistics
# ---------------------------------------------------------

st.subheader("Performance")

stat1, stat2, stat3, stat4, stat5, stat6 = st.columns(6)


with stat1:

    st.metric(
        "Cumulative Return",
        f"{cumulative_return:.2%}"
    )


with stat2:

    st.metric(
        "Average Daily",
        f"{average_daily_return:.2%}"
    )


with stat3:

    st.metric(
        "Win Rate",
        f"{win_rate:.1%}"
    )


with stat4:

    st.metric(
        "Best Day",
        f"{best_day:.2%}"
    )


with stat5:

    st.metric(
        "Worst Day",
        f"{worst_day:.2%}"
    )


with stat6:

    st.metric(
        "Max Drawdown",
        f"{max_drawdown:.2%}"
    )


# ---------------------------------------------------------
# Cumulative return chart
# ---------------------------------------------------------

# ---------------------------------------------------------
# Cumulative return chart
# ---------------------------------------------------------

st.subheader("Cumulative Return")


# Add the initial 0% starting point
first_prediction_date = (
    results["Prediction Date"].min()
)


starting_point = pd.DataFrame({
    "Return Date": [
        first_prediction_date
    ],
    "Cumulative Return": [
        0.0
    ]
})


chart_data = pd.concat(
    [
        starting_point,

        results[
            [
                "Return Date",
                "Cumulative Return"
            ]
        ]
    ],
    ignore_index=True
)


chart_data = (
    chart_data
    .drop_duplicates(
        subset=["Return Date"],
        keep="last"
    )
    .sort_values("Return Date")
    .reset_index(drop=True)
)


# Create readable date labels
chart_data["Date Label"] = (
    chart_data["Return Date"]
    .dt.strftime("%d/%m/%Y")
)


# Plotly line chart
fig = px.line(
    chart_data,
    x="Date Label",
    y="Cumulative Return",
    markers=True,
    labels={
        "Date Label": "Date",
        "Cumulative Return": "Cumulative Return"
    }
)


fig.update_yaxes(
    tickformat=".1%"
)


fig.update_layout(
    hovermode="x unified",
    xaxis=dict(
        type="category"
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------------------------------------
# Daily returns chart
# ---------------------------------------------------------

# ---------------------------------------------------------
# Daily portfolio returns chart
# ---------------------------------------------------------

st.subheader("Daily Portfolio Returns")


daily_chart = results[
    [
        "Return Date",
        "Portfolio Return"
    ]
].copy()


daily_chart["Date Label"] = (
    daily_chart["Return Date"]
    .dt.strftime("%d/%m/%Y")
)


fig_daily = px.bar(
    daily_chart,
    x="Date Label",
    y="Portfolio Return",
    labels={
        "Date Label": "Date",
        "Portfolio Return": "Daily Return"
    }
)


fig_daily.update_yaxes(
    tickformat=".1%"
)


fig_daily.update_layout(
    hovermode="x unified",
    xaxis=dict(
        type="category"
    )
)


st.plotly_chart(
    fig_daily,
    use_container_width=True
)


# ---------------------------------------------------------
# Current portfolio
# ---------------------------------------------------------

st.subheader("Current Portfolio")


latest_date = df["Date"].max()

current_df = df[
    df["Date"] == latest_date
].copy()


st.caption(
    f"Based on predictions generated on "
    f"{latest_date.strftime('%Y-%m-%d')}"
)


# ---------------------------------------------------------
# Current Long-only
# ---------------------------------------------------------

if selected_direction == "Long":

    current_long = (
        current_df
        .dropna(subset=[long_probability_column])
        .sort_values(
            long_probability_column,
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    current_long["Direction"] = "Long"

    display_current = current_long[
        [
            "Ticker",
            long_probability_column,
            "Close",
            "Direction"
        ]
    ].copy()

    display_current = display_current.rename(
        columns={
            long_probability_column: "Probability",
            "Close": "Price"
        }
    )


# ---------------------------------------------------------
# Current Short-only
# ---------------------------------------------------------

elif selected_direction == "Short":

    current_short = (
        current_df
        .dropna(subset=[short_probability_column])
        .sort_values(
            short_probability_column,
            ascending=True
        )
        .head(top_n)
        .copy()
    )

    current_short["Direction"] = "Short"

    display_current = current_short[
        [
            "Ticker",
            short_probability_column,
            "Close",
            "Direction"
        ]
    ].copy()

    display_current = display_current.rename(
        columns={
            short_probability_column: "Probability",
            "Close": "Price"
        }
    )


# ---------------------------------------------------------
# Current Long/Short
# ---------------------------------------------------------

else:

    current_long = (
        current_df
        .dropna(subset=[long_probability_column])
        .sort_values(
            long_probability_column,
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    current_long["Direction"] = "Long"


    current_short = (
        current_df
        .dropna(subset=[short_probability_column])
        .sort_values(
            short_probability_column,
            ascending=True
        )
        .head(top_n)
        .copy()
    )

    current_short["Direction"] = "Short"


    display_current = pd.concat(
        [
            current_long[
                [
                    "Ticker",
                    long_probability_column,
                    "Close",
                    "Direction"
                ]
            ].rename(
                columns={
                    long_probability_column: "Probability",
                    "Close": "Price"
                }
            ),

            current_short[
                [
                    "Ticker",
                    short_probability_column,
                    "Close",
                    "Direction"
                ]
            ].rename(
                columns={
                    short_probability_column: "Probability",
                    "Close": "Price"
                }
            )
        ],
        ignore_index=True
    )


# ---------------------------------------------------------
# Display current positions
# ---------------------------------------------------------

st.dataframe(
    display_current,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Historical portfolio picks
# ---------------------------------------------------------

st.subheader("Historical Portfolio Picks")

st.caption(
    "Expand a prediction week to see the stocks selected "
    "and their daily returns."
)


completed_dates = (
    results[
        [
            "Prediction Date",
            "Return Date"
        ]
    ]
    .drop_duplicates(
        subset=["Prediction Date"]
    )
    .sort_values(
        "Prediction Date",
        ascending=False
    )
    .to_dict("records")
)


for week in completed_dates:

    prediction_date = week["Prediction Date"]

    week_results = results[
        results["Prediction Date"] == prediction_date
    ]

    # Total return over the full week
    week_total_return = (
        (1 + week_results["Portfolio Return"])
        .prod()
        - 1
    )


    if selected_direction == "Long/Short":

        position_count = (
            f"{top_n} long + {top_n} short"
        )

    else:

        position_count = (
            f"{top_n} stocks"
        )


    expander_title = (
        f"{prediction_date.strftime('%Y-%m-%d')}  |  "
        f"Weekly Return: {week_total_return:.2%}  |  "
        f"{position_count}"
    )


    with st.expander(expander_title):

        historical_week = df[
            df["Date"] == prediction_date
        ].copy()


        # -------------------------------------------------
        # Long
        # -------------------------------------------------

        if selected_direction == "Long":

            selected_week = (
                historical_week
                .dropna(
                    subset=[long_probability_column]
                )
                .sort_values(
                    long_probability_column,
                    ascending=False
                )
                .head(top_n)
                .copy()
            )

            selected_week["Direction"] = "Long"

            selected_week["Probability"] = (
                selected_week[
                    long_probability_column
                ]
            )

            selected_week["Weekly Return"] = (
                selected_week["weekly_return"]
            )


            display_week = selected_week[
                [
                    "Ticker",
                    "Probability",
                    "Close",
                    "Weekly Return",
                    *daily_returns.keys()
                ]
            ].copy()


        # -------------------------------------------------
        # Short
        # -------------------------------------------------

        elif selected_direction == "Short":

            selected_week = (
                historical_week
                .dropna(
                    subset=[short_probability_column]
                )
                .sort_values(
                    short_probability_column,
                    ascending=True
                )
                .head(top_n)
                .copy()
            )

            selected_week["Direction"] = "Short"

            selected_week["Probability"] = (
                selected_week[
                    short_probability_column
                ]
            )

            selected_week["Weekly Return"] = (
                -selected_week["weekly_return"]
            )


            display_week = selected_week[
                [
                    "Ticker",
                    "Probability",
                    "Close",
                    "Weekly Return",
                    *daily_returns.keys()
                ]
            ].copy()


        # -------------------------------------------------
        # Long/Short
        # -------------------------------------------------

        else:

            long_week = (
                historical_week
                .dropna(
                    subset=[long_probability_column]
                )
                .sort_values(
                    long_probability_column,
                    ascending=False
                )
                .head(top_n)
                .copy()
            )

            long_week["Direction"] = "Long"

            long_week["Probability"] = (
                long_week[
                    long_probability_column
                ]
            )

            long_week["Weekly Return"] = (
                long_week["weekly_return"]
            )


            short_week = (
                historical_week
                .dropna(
                    subset=[short_probability_column]
                )
                .sort_values(
                    short_probability_column,
                    ascending=True
                )
                .head(top_n)
                .copy()
            )

            short_week["Direction"] = "Short"

            short_week["Probability"] = (
                short_week[
                    short_probability_column
                ]
            )

            short_week["Weekly Return"] = (
                -short_week["weekly_return"]
            )


            st.markdown("### Long")

            long_display = long_week[
                [
                    "Ticker",
                    "Probability",
                    "Close",
                    "Weekly Return",
                    *daily_returns.keys()
                ]
            ].copy()


            long_display = long_display.rename(
                columns=daily_returns
            )


            long_display["Probability"] = (
                long_display["Probability"]
                .map(lambda x: f"{x:.2%}")
            )


            long_display["Weekly Return"] = (
                long_display["Weekly Return"]
                .map(lambda x: f"{x:.2%}")
            )


            for column in daily_returns.values():

                long_display[column] = (
                    long_display[column]
                    .map(
                        lambda x:
                        f"{x:.2%}"
                        if pd.notna(x)
                        else ""
                    )
                )


            st.dataframe(
                long_display,
                use_container_width=True,
                hide_index=True
            )


            st.markdown("### Short")

            short_display = short_week[
                [
                    "Ticker",
                    "Probability",
                    "Close",
                    "Weekly Return",
                    *daily_returns.keys()
                ]
            ].copy()


            short_display = short_display.rename(
                columns=daily_returns
            )


            short_display["Probability"] = (
                short_display["Probability"]
                .map(lambda x: f"{x:.2%}")
            )


            short_display["Weekly Return"] = (
                short_display["Weekly Return"]
                .map(lambda x: f"{x:.2%}")
            )


            for column in daily_returns.values():

                short_display[column] = (
                    short_display[column]
                    .map(
                        lambda x:
                        f"{x:.2%}"
                        if pd.notna(x)
                        else ""
                    )
                )


            st.dataframe(
                short_display,
                use_container_width=True,
                hide_index=True
            )


            continue


        # -------------------------------------------------
        # Format Long-only / Short-only
        # -------------------------------------------------

        display_week = display_week.rename(
            columns=daily_returns
        )


        display_week["Probability"] = (
            display_week["Probability"]
            .map(lambda x: f"{x:.2%}")
        )


        display_week["Weekly Return"] = (
            display_week["Weekly Return"]
            .map(lambda x: f"{x:.2%}")
        )


        for column in daily_returns.values():

            display_week[column] = (
                display_week[column]
                .map(
                    lambda x:
                    f"{x:.2%}"
                    if pd.notna(x)
                    else ""
                )
            )


        st.dataframe(
            display_week,
            use_container_width=True,
            hide_index=True
        )


# ---------------------------------------------------------
# Daily results table
# ---------------------------------------------------------

st.subheader("Daily Portfolio Returns")


display_results = results.copy()


display_results["Prediction Date"] = (
    display_results["Prediction Date"]
    .dt.strftime("%Y-%m-%d")
)


display_results["Return Date"] = (
    display_results["Return Date"]
    .dt.strftime("%Y-%m-%d")
)


display_results["Portfolio Return"] = (
    display_results["Portfolio Return"]
    .map(lambda x: f"{x:.2%}")
)


display_results["Cumulative Return"] = (
    display_results["Cumulative Return"]
    .map(lambda x: f"{x:.2%}")
)


display_results["Drawdown"] = (
    display_results["Drawdown"]
    .map(lambda x: f"{x:.2%}")
)


display_results = display_results[
    [
        "Prediction Date",
        "Return Date",
        "Period",
        "Portfolio Return",
        "Cumulative Return",
        "Drawdown"
    ]
]


st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)
