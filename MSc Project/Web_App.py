import streamlit as st
import pandas as pd
import joblib
from df_functions import df_transform
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Player Injury Risk Predictor", layout="wide")
st.title("Player Injury Risk Prediction Dashboard")

def load_model():
    rf_model = joblib.load("final_rf_model.pkl")
    return rf_model

model = load_model()
st.success("Random Forest ML model loaded successfully!")


st.subheader("Upload Player Match Data")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        raw_df = pd.read_csv(uploaded_file, parse_dates=["Date"], date_format="%d-%m-%Y",encoding="ISO-8859-1")
        st.subheader("Preview of Uploaded Data")
        st.write(raw_df.head())

        required_cols = ["Player", "Date", "Team", "Opp", "Comp", "Venue", "Pos.", "Minutes"]
        missing = [c for c in required_cols if c not in raw_df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            status = st.empty()

            status.info("Processing data...")

            transformed_df = df_transform(raw_df)

            if transformed_df is not None:
               status.success("Processing successful!")
            else:
               status.error("Processing failed. Please check your input.")

            if "Start" in transformed_df.columns:
                transformed_df["Start"] = transformed_df["Start"].map({"Y": 1, "N": 0}).fillna(0).astype(int)

            st.subheader("Preview of Transformed Data")
            st.write(transformed_df.head())


            predictions = model.predict(transformed_df)
            probabilities = model.predict_proba(transformed_df)[:, 1]

            transformed_df["Predicted_Injury"] = predictions
            transformed_df["Injury_Probability"] = probabilities

            st.sidebar.header("Filters")

            comp_options = sorted(transformed_df["Comp"].dropna().unique().tolist())
            pos_options = sorted(
                transformed_df.get("PrimaryPosition", transformed_df.get("Pos.", [])).dropna().unique().tolist()
            )

            selected_comp = st.sidebar.multiselect("Competition", comp_options)
            selected_pos = st.sidebar.multiselect("Primary position", pos_options)

            df_filtered = transformed_df.copy()

            if selected_comp:
                df_filtered = df_filtered[df_filtered["Comp"].isin(selected_comp)]

            if "PrimaryPosition" in df_filtered.columns and selected_pos:
                df_filtered = df_filtered[df_filtered["PrimaryPosition"].isin(selected_pos)]

            if df_filtered.empty:
                st.warning("No data after applying filters. Please adjust your selections.")
            else:
                tab1, tab2 = st.tabs(["Matchday Predictions", "Player Risk Overview"])

                with tab1:
                    st.subheader("Matchday Injury Risk Predictions")

                    display_cols = ["Player", "Date","Comp", "Injury_Probability", "Predicted_Injury"]
                    st.dataframe(
                        df_filtered[display_cols]
                        .sort_values(["Player", "Date"], ascending=[True, True])
                        .reset_index(drop=True)
                    )

                    st.markdown("---")
                    st.subheader("High-Risk Alerts")

                    risk_thr = st.slider(
                        "Select high-risk probability threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.6,
                        step=0.01,
                    )

                    high_risk = (
                        df_filtered[df_filtered["Injury_Probability"] >= risk_thr]
                        .sort_values(["Date", "Injury_Probability"], ascending=[True, False])
                    )

                    if high_risk.empty:
                        st.info("No players exceed the current high-risk threshold.")
                    else:
                        st.dataframe(
                            high_risk[["Player", "Date", "Comp", "Injury_Probability"]]
                            .reset_index(drop=True)
                        )

                    st.subheader("Injury Probability Over Time (Top Risk Players)")

                    top_players = (
                        df_filtered.groupby("Player", as_index=False)["Injury_Probability"]
                        .mean()
                        .sort_values("Injury_Probability", ascending=False)
                        .head(5)["Player"]
                        .tolist()
                    )

                    filtered = df_filtered[df_filtered["Player"].isin(top_players)]

                    fig_time = px.line(
                        filtered,
                        x="Date",
                        y="Injury_Probability",
                        color="Player",
                        markers=True,
                        title="Injury Risk Progression Over Time (Top 5 Players)",
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    fig_time.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Predicted Probability")
                    st.plotly_chart(fig_time, use_container_width=True)

                with tab2:

                    player_summary = (
                        df_filtered.groupby("Player", as_index=False)
                        .agg({
                            "Injury_Probability": "mean",
                            "Predicted_Injury": "max",
                            "Date": "max"
                        })
                        .sort_values("Injury_Probability", ascending=False)
                    )

                    st.subheader("Top 10 Players by Average Injury Risk")

                    top_players_chart = player_summary.head(10)
                    fig_bar = px.bar(
                        top_players_chart,
                        x="Injury_Probability",
                        y="Player",
                        orientation="h",
                        color="Injury_Probability",
                        color_continuous_scale="Reds",
                        title="Top 10 Players by Average Injury Risk"
                    )
                    fig_bar.update_layout(
                        template="plotly_dark",
                        xaxis_title="Injury Risk Probability",
                        yaxis_title="Player",
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    st.markdown("---")

                    st.subheader("Injury Probability vs Minutes Played")
                    fig_scatter = px.scatter(
                        df_filtered,
                        x="Minutes",
                        y="Injury_Probability",
                        color="PrimaryPosition",
                        hover_name="Player",
                        opacity=0.7,
                        color_discrete_sequence=px.colors.qualitative.Safe,
                        title="Correlation between Minutes Played and Injury Probability"
                    )
                    fig_scatter.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_scatter, use_container_width=True)

                    valid_corr = df_filtered[["Minutes", "Injury_Probability"]].dropna()
                    if len(valid_corr) > 1:
                        corr = np.corrcoef(valid_corr["Minutes"], valid_corr["Injury_Probability"])[0, 1]
                        st.markdown(f"**Correlation coefficient (Minutes vs Injury Probability):** {corr:.3f}")
                    else:
                        st.info("Not enough data to compute correlation.")

                    st.markdown("---")

                    avg_risk = df_filtered["Injury_Probability"].mean()
                    high_risk_count = (df_filtered["Predicted_Injury"] == 1).sum()
                    total_players = df_filtered["Player"].nunique()

                    st.markdown(f"""
                    ##  Summary Insights
                    - **Average injury probability:** {avg_risk:.2%}  
                    - **Players predicted as high-risk:** {high_risk_count}  
                    - **Unique players evaluated:** {total_players}  
                    """)

                    st.markdown("---")

                    st.subheader("Player Workload & Risk Analysis")

                    players = sorted(df_filtered["Player"].unique())
                    selected_player = st.selectbox("Select a player:", players)

                    player_data = df_filtered[df_filtered["Player"] == selected_player].sort_values("Date")

                    st.write("Recent matches for selected player")
                    st.dataframe(
                        player_data[
                            ["Date", "Comp", "Minutes", "RollingAvg_14d", "FixtureCount5d", "Injury_Probability"]
                        ].reset_index(drop=True)
                    )

                    fig_workload = px.line(
                        player_data,
                        x="Date",
                        y="RollingAvg_14d",
                        title=f"14-Day Rolling Workload - {selected_player}",
                        labels={"RollingAvg_14d": "Minutes (last 14 days)"}
                    )
                    fig_workload.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_workload, use_container_width=True)

                    fig_risk = px.line(
                        player_data,
                        x="Date",
                        y="Injury_Probability",
                        title=f"Injury Probability Over Time - {selected_player}",
                        labels={"Injury_Probability": "Predicted Injury Probability"}
                    )
                    fig_risk.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_risk, use_container_width=True)


                    if "FixtureCount5d" in player_data.columns:
                        fig_fix = px.bar(
                            player_data,
                            x="Date",
                            y="FixtureCount5d",
                            title=f"Fixture Count - {selected_player}"
                        )
                        fig_fix.update_layout(template="plotly_dark")
                        st.plotly_chart(fig_fix, use_container_width=True)


                st.download_button(
                    label=" Download Predictions as CSV",
                    data=df_filtered.to_csv(index=False),
                    file_name="injury_predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info(" Upload a CSV file to begin analysis.")
