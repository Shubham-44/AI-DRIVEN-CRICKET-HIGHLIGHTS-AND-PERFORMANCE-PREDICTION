import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

# Load features and target
X = pd.read_csv('outputs/X_features.csv')
y = pd.read_csv('outputs/y_target.csv').values.ravel()

# Optional: One-hot encode categorical features for better XGBoost performance
X_encoded = pd.get_dummies(X, columns=['batter_enc', 'batting_team_enc', 'opponent_team_enc', 'venue_enc'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    objective='reg:squarederror'
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("✅ XGBoost Model Performance:")
print(f"🔹 MAE:  {mae:.2f}")
print(f"🔹 RMSE: {rmse:.2f}")
print(f"🔹 R²:   {r2:.4f}")

# Feature importance plot
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=10)
plt.title("📊 Top Feature Importances - XGBoost")
plt.tight_layout()
plt.savefig("outputs/feature_importance_xgb.png")
plt.show()
