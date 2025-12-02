import pandas as pd
import numpy as np


def get_season(date):
    year = date.year
    if date.month >= 8:  
        return f"{year}/{str(year+1)[-2:]}"
    else:  
        return f"{year-1}/{str(year)[-2:]}"

pos_load = {
     "GK":2, "DF":3, "MF":5, "FW":4
}

hi_comp = {"Premier League","Champions League","Europa League"}
def comp_int(x):
    if x in hi_comp:
        return 5
    elif x == "FA Cup":
        return 4
    elif x == "EFL Cup" or x == "FA Community Shield":
        return 3
    else:
        return 1

def pos_adjust(x):
    if x == "FW,MF":
        return "FW"
    elif x == "MF,FW":
        return "MF"
    elif x == "MF,DF":
        return "MF"
    elif x == "DF,MF":
        return "DF"
    elif x == "DF,FW":
        return "DF"
    elif x == "FW,DF":
        return "FW"
    else:
        return x

def fixture_5d_count(df):
    
    df = df.copy()
    rolling_counts = []
    for player, grp in df.groupby("Player"):
        grp = grp.set_index("Date").sort_index()
        # Count matches played in the last 5D (non-zero minutes)
        count5 = grp["Minutes"].gt(0).rolling("5D").sum().values  # .gt(0) = played match
        rolling_counts.extend(count5)
    df["FixtureCount5d"] = rolling_counts
    return df

def rolling7d(df):
    df = df.copy()
    rolling_vals = []
    for player, grp in df.groupby("Player"):
        grp = grp.set_index("Date").sort_index()
        # this is an array of length grp.shape[0]
        roll7 = grp["Minutes"].rolling("7D").sum().values
        rolling_vals.extend(roll7)
    df["RollingAvg_7d"] = rolling_vals
    return df
	
def rolling14d(df):
    df = df.copy()
    rolling_vals = []
    for player, grp in df.groupby("Player"):
        grp = grp.set_index("Date").sort_index()
        # this is an array of length grp.shape[0]
        roll7 = grp["Minutes"].rolling("14D").sum().values
        rolling_vals.extend(roll7)
    df["RollingAvg_14d"] = rolling_vals
    return df

def age_band(age):
    if age <= 21:
        return "developing"
    elif age > 21 and age <=28:
        return "peak"
    elif age > 28 and age <=32:
        return "mature"
    else:
        return "decline"


def df_transform(source):
    df = source.copy()
    df["Date"] = pd.to_datetime(df["Date"]) #validation
    
    # True active span for each player
    span = df.groupby("Player")["Date"].agg(["min","max"]).reset_index()
    span.rename(columns={"min":"first_game","max":"last_game"}, inplace=True)

    # Collect team-level match dates
    team_dates = df["Date"].drop_duplicates().sort_values()

    
    frames = []
    for row in span.itertuples(index=False):
        player = row.Player
        valid_dates = team_dates[(team_dates >= row.first_game) & (team_dates <= row.last_game)]
        cal = pd.DataFrame({"Player": player, "Date": valid_dates})
        frames.append(cal)
    calendar_df = pd.concat(frames, ignore_index=True)

    
    full_df = calendar_df.merge(df, on=["Player","Date"], how="left")

    
    full_df["Minutes"] = full_df["Minutes"].fillna(0)
    for col in ["Pos.","Team","Height(m)", "Weight(kg)", "Age"]:
        if col in full_df.columns:
            full_df[col] = full_df.groupby("Player")[col].ffill().bfill()

    full_df['Age'] = full_df['Age'].astype('int32')
    
    
    full_df = full_df.sort_values(["Player","Date"]).reset_index(drop=True)

    
    full_df = rolling14d(full_df)
    full_df = rolling7d(full_df)
    full_df = fixture_5d_count(full_df)
    full_df["FixtureCongestionFlag"] = full_df["FixtureCount5d"] >= 2
    full_df["Age_Group"] = full_df["Age"].apply(age_band)
    full_df['BMI'] = (full_df['Weight(kg)'] / (full_df['Height(m)'] ** 2)).round(decimals=3)
    full_df["Age_Weight"] = np.exp(-((full_df["Age"] - 26) ** 2) / (2 * 5 ** 2))

    
    if "Pos." in full_df.columns:
        full_df["PrimaryPosition"] = full_df["Pos."].apply(pos_adjust)
        full_df["PositionLoadScore"] = full_df["Pos."].map(pos_load).fillna(3)
    if "Comp" in full_df.columns:
        full_df["CompIntensity"] = full_df["Comp"].apply(comp_int)
    full_df["Season"] = full_df["Date"].apply(get_season)
    full_df["BMI_PosLoad"] = full_df["BMI"] * full_df["PositionLoadScore"]
    full_df["Age_Load_Index"] = full_df["Age_Weight"] * full_df["PositionLoadScore"]
    full_df["Start"] = full_df["Start"].map({"Y": 1, "N": 0}).fillna(0).astype(int)

    return full_df
