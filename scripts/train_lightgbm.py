import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time

# Load dataset
X = pd.read_csv('outputs/X_features.csv')
y = pd.read_csv('outputs/y_target.csv').values.ravel()

# One-Hot Encoding (same as XGBoost setup)
X_encoded = pd.get_dummies(X, columns=['batter_enc', 'batting_team_enc', 'opponent_team_enc', 'venue_enc'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Convert to LightGBM dataset format
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Define model parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': -1
}

# Track training time
start = time.time()

# Train the model
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)


end = time.time()
training_time = end - start

# Predict and evaluate
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("✅ LightGBM Model Performance:")
print(f"🔹 MAE:  {mae:.2f}")
print(f"🔹 RMSE: {rmse:.2f}")
print(f"🔹 R²:   {r2:.4f}")
print(f"⏱️ Training Time: {training_time:.2f} seconds")

# Plot feature importances
lgb.plot_importance(model, max_num_features=10, importance_type='gain')
plt.title("📊 Top Feature Importances - LightGBM")
plt.tight_layout()
plt.savefig("outputs/feature_importance_lgbm.png")
plt.show()
