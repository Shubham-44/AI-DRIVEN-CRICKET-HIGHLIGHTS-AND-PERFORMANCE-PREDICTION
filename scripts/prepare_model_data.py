import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load player-level dataset
df = pd.read_csv('outputs/player_match_data.csv')

# 🧠 1. Extract year from date (useful for trend or season-based modeling)
df['year'] = pd.to_datetime(df['date']).dt.year

# 🧠 2. Encode categorical features
label_cols = ['batter', 'batting_team', 'opponent_team', 'venue']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoder for future use (in prediction)

# 🧠 3. Define Features (X) and Target (y)
feature_cols = ['batter_enc', 'batting_team_enc', 'opponent_team_enc', 'venue_enc', 'year']
X = df[feature_cols]
y = df['runs_scored']

# ✅ Save to files for modeling
X.to_csv('outputs/X_features.csv', index=False)
y.to_csv('outputs/y_target.csv', index=False)

print("✅ Features and target data saved:")
print("🔸 X shape:", X.shape)
print("🔸 y shape:", y.shape)
