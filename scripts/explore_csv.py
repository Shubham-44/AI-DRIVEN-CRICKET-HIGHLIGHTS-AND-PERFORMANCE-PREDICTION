import pandas as pd

# Load the CSV
df = pd.read_csv('outputs/ipl_matches.csv')

# Show top 5 rows
print("🔍 Sample rows:")
print(df.head())

# Show structure
print("\n🧠 Columns in the dataset:")
print(df.columns)

# Show unique batters and bowlers
print("\n🏏 Total unique batters:", df['batter'].nunique())
print("🏏 Total unique bowlers:", df['bowler'].nunique())

# Matches and dates
print("\n📅 Matches in dataset:", df['match_id'].nunique())
print("📍 Match dates range:", df['date'].min(), "to", df['date'].max())

# Top 10 venues
print("\n🏟️ Most common venues:")
print(df['venue'].value_counts().head(10))

# Null values
print("\n🚨 Missing values check:")
print(df.isnull().sum())
