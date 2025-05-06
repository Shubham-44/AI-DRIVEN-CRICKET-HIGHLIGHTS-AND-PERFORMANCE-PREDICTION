import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import pickle

# Load enhanced data
df = pd.read_csv('outputs/enhanced_player_stats.csv')

# Drop rows with NaN (e.g., strike_rate might be NaN for 0 balls faced)
df = df.dropna()

# Extract year from date
df['year'] = pd.to_datetime(df['date']).dt.year

# Label encode categorical features
le_venue = LabelEncoder()
le_team = LabelEncoder()

df['venue_enc'] = le_venue.fit_transform(df['venue'])
df['team_enc'] = le_team.fit_transform(df['batting_team'])

# Define features and target
feature_cols = ['balls_faced', 'fours', 'sixes', 'strike_rate', 'venue_enc', 'team_enc', 'year']
X = df[feature_cols]
y = df['total_runs']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "outputs/enhanced_rf_model.pkl")
pickle.dump(le_venue, open("outputs/venue_encoder.pkl", "wb"))
pickle.dump(le_team, open("outputs/team_encoder.pkl", "wb"))

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("✅ Enhanced Model Performance:")
print(f"🔹 MAE:  {mae:.2f}")
print(f"🔹 RMSE: {rmse:.2f}")
print(f"🔹 R²:   {r2:.4f}")

# Optional: Feature importance
importances = model.feature_importances_
plt.figure(figsize=(8, 5))
plt.title("📊 Feature Importances - Enhanced Model")
plt.bar(feature_cols, importances)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/feature_importance_enhanced.png")
plt.show()
