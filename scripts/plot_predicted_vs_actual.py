import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load enhanced data
df = pd.read_csv('outputs/enhanced_player_stats.csv').dropna()
df['year'] = pd.to_datetime(df['date']).dt.year

# Encode categorical
le_venue = LabelEncoder()
le_team = LabelEncoder()
df['venue_enc'] = le_venue.fit_transform(df['venue'])
df['team_enc'] = le_team.fit_transform(df['batting_team'])

# Define features and target
features = ['balls_faced', 'strike_rate', 'fours', 'sixes', 'venue_enc', 'team_enc', 'year']
X = df[features]
y = df['total_runs']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Plot
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.4, color='green', label='Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal')
plt.xlabel("Actual Runs")
plt.ylabel("Predicted Runs")
plt.title("📈 Predicted vs Actual Runs")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/predicted_vs_actual.png")
plt.show()
