import pandas as pd

# Load player-level dataset
df = pd.read_csv('outputs/player_match_data.csv')

# Show first few rows
print("🔍 Sample rows:")
print(df.head())

# Check shape and columns
print("\n📊 Shape of data:", df.shape)
print("🧠 Columns:", list(df.columns))

# Distribution of runs
print("\n📈 Run Distribution:")
print(df['runs_scored'].describe())

# Check for outliers or anomalies
print("\n🎯 Top 10 highest scores:")
print(df.sort_values(by='runs_scored', ascending=False).head(10))

# Check missing values
print("\n🚨 Missing values:")
print(df.isnull().sum())

# Unique values
print("\n📍 Unique venues:", df['venue'].nunique())
print("🏏 Unique players:", df['batter'].nunique())
print("🧢 Unique teams:", df['batting_team'].nunique())
