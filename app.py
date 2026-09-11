import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import timedelta

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="ML Portfolio Tracker",
    page_icon="📈",
    layout="wide"
)

st.title("ML Portfolio Tracker")

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

max_N = df["Ticker"].nunique()

# ---------------------------------------------------------
# Define available models
# ---------------------------------------------------------

models = {
    "Ensemble": "Ensemble",
    "Random Forest": "RF",
    "XGBoost": "XGBoost",
    "LGBM": "LGBM",
    "CatBoost": "CatBoost",
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
        ["Long/Short", "Long", "Short"]
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


if selected_direction == "Long":

    required_columns = [long_probability_column]

elif selected_direction == "Short":

    required_columns = [short_probability_column]

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
# Calculate historical weekly portfolio returns
# ---------------------------------------------------------

weekly_results = []


for prediction_date, week_df in df.groupby("Date"):

    # Only completed weeks
    week_df = week_df.dropna(
        subset=["weekly_return"]
    ).copy()

    if week_df.empty:
        continue


    # -----------------------------------------------------
    # Long-only
    # -----------------------------------------------------

    if selected_direction == "Long":

        long_df = (
            week_df
            .dropna(subset=[long_probability_column])
            .sort_values(
                long_probability_column,
                ascending=False
            )
            .head(top_n)
        )

        if long_df.empty:
            continue

        portfolio_return = (
            long_df["weekly_return"].mean()
        )

        number_of_stocks = len(long_df)


    # -----------------------------------------------------
    # Short-only
    # -----------------------------------------------------

    elif selected_direction == "Short":

        short_df = (
            week_df
            .dropna(subset=[short_probability_column])
            .sort_values(
                short_probability_column,
                ascending=True
            )
            .head(top_n)
        )

        if short_df.empty:
            continue

        portfolio_return = (
            -short_df["weekly_return"]
        ).mean()

        number_of_stocks = len(short_df)


    # -----------------------------------------------------
    # Long/Short
    # -----------------------------------------------------

    else:

        # Long the top N long probabilities
        long_df = (
            week_df
            .dropna(subset=[long_probability_column])
            .sort_values(
                long_probability_column,
                ascending=False
            )
            .head(top_n)
        )

        # Short the bottom N short probabilities
        short_df = (
            week_df
            .dropna(subset=[short_probability_column])
            .sort_values(
                short_probability_column,
                ascending=True
            )
            .head(top_n)
        )

        if long_df.empty or short_df.empty:
            continue

        long_return = (
            long_df["weekly_return"].mean()
        )

        short_return = (
            -short_df["weekly_return"]
        ).mean()

        # Long and short sides each represent 50%
        portfolio_return = (
            long_return + short_return
        ) / 2

        number_of_stocks = (
            len(long_df) + len(short_df)
        )


    # -----------------------------------------------------
    # The return happens ONE WEEK AFTER the prediction.
    #
    # Therefore:
    #
    # Prediction Date = date the stocks were selected
    # Return Date     = date the weekly return is realised
    # -----------------------------------------------------

    return_date = prediction_date + timedelta(days=7)


    weekly_results.append({
        "Prediction Date": prediction_date,
        "Return Date": return_date,
        "Portfolio Return": portfolio_return,
        "Stocks": number_of_stocks
    })


# ---------------------------------------------------------
# Create results dataframe
# ---------------------------------------------------------

results = pd.DataFrame(weekly_results)


if results.empty:

    st.warning(
        "There are no completed weeks with available returns."
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

wealth_index = 1 + results["Cumulative Return"]

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

average_weekly_return = (
    results["Portfolio Return"].mean()
)

win_rate = (
    results["Portfolio Return"] > 0
).mean()

best_week = (
    results["Portfolio Return"].max()
)

worst_week = (
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
        "Average Weekly",
        f"{average_weekly_return:.2%}"
    )

with stat3:
    st.metric(
        "Win Rate",
        f"{win_rate:.1%}"
    )

with stat4:
    st.metric(
        "Best Week",
        f"{best_week:.2%}"
    )

with stat5:
    st.metric(
        "Worst Week",
        f"{worst_week:.2%}"
    )

with stat6:
    st.metric(
        "Max Drawdown",
        f"{max_drawdown:.2%}"
    )


# ---------------------------------------------------------
# Cumulative return chart
# ---------------------------------------------------------

st.subheader("Cumulative Return")


# Add the starting point:
#
# The first prediction is made on the first prediction date,
# but its return is only realised one week later.
#
# Therefore the portfolio starts at 0% on the first
# prediction date.

first_prediction_date = results["Prediction Date"].min()

chart_data = pd.concat(
    [
        pd.DataFrame({
            "Date": [first_prediction_date],
            "Cumulative Return": [0.0]
        }),

        results[
            ["Return Date", "Cumulative Return"]
        ].rename(
            columns={"Return Date": "Date"}
        )
    ],
    ignore_index=True
)

chart_data = (
    chart_data
    .drop_duplicates(subset=["Date"], keep="last")
    .sort_values("Date")
    .set_index("Date")
)


st.line_chart(
    chart_data
)


# ---------------------------------------------------------
# Weekly return chart
# ---------------------------------------------------------

st.subheader("Weekly Returns")

weekly_chart = (
    results
    .set_index("Return Date")[
        ["Portfolio Return"]
    ]
)

st.bar_chart(
    weekly_chart
)


# ---------------------------------------------------------
# Current portfolio
# ---------------------------------------------------------

st.subheader("Current Portfolio")


# Most recent prediction date
latest_date = df["Date"].max()

current_df = df[
    df["Date"] == latest_date
].copy()


st.caption(
    f"Based on predictions generated on "
    f"{latest_date.strftime('%Y-%m-%d')}"
)


# ---------------------------------------------------------
# Current long/short selections
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


else:

    # Long side
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


    # Short side
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


    # Combine both sides
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
    "Expand a week to see the stocks selected by the strategy "
    "and how each position contributed to that week's return."
)


# Work backwards so the most recent completed week appears first
completed_dates = (
    results
    .sort_values("Return Date", ascending=False)
    [["Prediction Date", "Return Date", "Portfolio Return"]]
    .to_dict("records")
)


for week in completed_dates:

    prediction_date = week["Prediction Date"]
    return_date = week["Return Date"]
    portfolio_return = week["Portfolio Return"]


    # -----------------------------------------------------
    # Create the expander title
    # -----------------------------------------------------

    direction_label = selected_direction

    if selected_direction == "Long/Short":
        position_count = f"{top_n} long + {top_n} short"
    else:
        position_count = f"{top_n} stocks"


    expander_title = (
        f"{prediction_date.strftime('%Y-%m-%d')} "
        f"→ {return_date.strftime('%Y-%m-%d')}  |  "
        f"Return: {portfolio_return:.2%}  |  "
        f"{position_count}"
    )


    with st.expander(expander_title):

        # Get the rows corresponding to this prediction week
        week_df = df[
            df["Date"] == prediction_date
        ].copy()


        # -------------------------------------------------
        # Long-only
        # -------------------------------------------------

        if selected_direction == "Long":

            long_df = (
                week_df
                .dropna(subset=[long_probability_column])
                .sort_values(
                    long_probability_column,
                    ascending=False
                )
                .head(top_n)
                .copy()
            )


            # Each position gets equal weight
            weight = 1 / len(long_df)


            long_df["Rank"] = range(
                1,
                len(long_df) + 1
            )

            long_df["Direction"] = "Long"

            long_df["Position Return"] = (
                long_df["weekly_return"]
            )

            long_df["Contribution"] = (
                long_df["Position Return"] * weight
            )


            display_df = long_df[
                [
                    "Rank",
                    "Ticker",
                    long_probability_column,
                    "Close",
                    "Position Return",
                    "Contribution"
                ]
            ].copy()


            display_df = display_df.rename(
                columns={
                    long_probability_column: "Probability",
                    "Close": "Entry Price"
                }
            )


        # -------------------------------------------------
        # Short-only
        # -------------------------------------------------

        elif selected_direction == "Short":

            short_df = (
                week_df
                .dropna(subset=[short_probability_column])
                .sort_values(
                    short_probability_column,
                    ascending=True
                )
                .head(top_n)
                .copy()
            )


            # Each position gets equal weight
            weight = 1 / len(short_df)


            short_df["Rank"] = range(
                1,
                len(short_df) + 1
            )

            short_df["Direction"] = "Short"

            # Reverse stock return for short position
            short_df["Position Return"] = (
                -short_df["weekly_return"]
            )

            short_df["Contribution"] = (
                short_df["Position Return"] * weight
            )


            display_df = short_df[
                [
                    "Rank",
                    "Ticker",
                    short_probability_column,
                    "Close",
                    "Position Return",
                    "Contribution"
                ]
            ].copy()


            display_df = display_df.rename(
                columns={
                    short_probability_column: "Probability",
                    "Close": "Entry Price"
                }
            )


        # -------------------------------------------------
        # Long/Short
        # -------------------------------------------------

        else:

            # ---------------------------------------------
            # Long side
            # ---------------------------------------------

            long_df = (
                week_df
                .dropna(subset=[long_probability_column])
                .sort_values(
                    long_probability_column,
                    ascending=False
                )
                .head(top_n)
                .copy()
            )


            # ---------------------------------------------
            # Short side
            # ---------------------------------------------

            short_df = (
                week_df
                .dropna(subset=[short_probability_column])
                .sort_values(
                    short_probability_column,
                    ascending=True
                )
                .head(top_n)
                .copy()
            )


            # Each position has equal weight across the
            # entire long/short portfolio.
            #
            # e.g. Top N = 5:
            # 5 long + 5 short = 10 positions
            # each position = 10%
            total_positions = (
                len(long_df) + len(short_df)
            )

            weight = 1 / total_positions


            # ---------------------------------------------
            # Long positions
            # ---------------------------------------------

            long_df["Rank"] = range(
                1,
                len(long_df) + 1
            )

            long_df["Direction"] = "Long"

            long_df["Position Return"] = (
                long_df["weekly_return"]
            )

            long_df["Contribution"] = (
                long_df["Position Return"] * weight
            )


            # ---------------------------------------------
            # Short positions
            # ---------------------------------------------

            short_df["Rank"] = range(
                1,
                len(short_df) + 1
            )

            short_df["Direction"] = "Short"

            short_df["Position Return"] = (
                -short_df["weekly_return"]
            )

            short_df["Contribution"] = (
                short_df["Position Return"] * weight
            )


            # ---------------------------------------------
            # Prepare long table
            # ---------------------------------------------

            long_display = long_df[
                [
                    "Rank",
                    "Ticker",
                    long_probability_column,
                    "Close",
                    "Direction",
                    "Position Return",
                    "Contribution"
                ]
            ].copy()


            long_display = long_display.rename(
                columns={
                    long_probability_column: "Probability",
                    "Close": "Entry Price"
                }
            )


            # ---------------------------------------------
            # Prepare short table
            # ---------------------------------------------

            short_display = short_df[
                [
                    "Rank",
                    "Ticker",
                    short_probability_column,
                    "Close",
                    "Direction",
                    "Position Return",
                    "Contribution"
                ]
            ].copy()


            short_display = short_display.rename(
                columns={
                    short_probability_column: "Probability",
                    "Close": "Entry Price"
                }
            )


            # ---------------------------------------------
            # Display Long / Short separately
            # ---------------------------------------------

            st.markdown("### Long")

            long_display["Probability"] = (
                long_display["Probability"]
                .map(lambda x: f"{x:.2%}")
            )

            long_display["Position Return"] = (
                long_display["Position Return"]
                .map(lambda x: f"{x:.2%}")
            )

            long_display["Contribution"] = (
                long_display["Contribution"]
                .map(lambda x: f"{x:.2%}")
            )

            st.dataframe(
                long_display,
                use_container_width=True,
                hide_index=True
            )


            st.markdown("### Short")

            short_display["Probability"] = (
                short_display["Probability"]
                .map(lambda x: f"{x:.2%}")
            )

            short_display["Position Return"] = (
                short_display["Position Return"]
                .map(lambda x: f"{x:.2%}")
            )

            short_display["Contribution"] = (
                short_display["Contribution"]
                .map(lambda x: f"{x:.2%}")
            )

            st.dataframe(
                short_display,
                use_container_width=True,
                hide_index=True
            )


            # Move to next week because Long/Short has
            # already displayed everything inside this branch.
            continue


        # -------------------------------------------------
        # Format Long-only / Short-only table
        # -------------------------------------------------

        display_df["Probability"] = (
            display_df["Probability"]
            .map(lambda x: f"{x:.2%}")
        )

        display_df["Position Return"] = (
            display_df["Position Return"]
            .map(lambda x: f"{x:.2%}")
        )

        display_df["Contribution"] = (
            display_df["Contribution"]
            .map(lambda x: f"{x:.2%}")
        )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
