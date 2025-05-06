import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load preprocessed features and target
X = pd.read_csv('outputs/X_features.csv')
y = pd.read_csv('outputs/y_target.csv').values.ravel()  # flatten y

# 1. Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Predictions
y_pred = model.predict(X_test)

# 4. Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("✅ Model Performance:")
print(f"🔹 MAE:  {mae:.2f}")
print(f"🔹 RMSE: {rmse:.2f}")
print(f"🔹 R²:   {r2:.4f}")

# 5. Feature Importance Plot
feature_names = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.title("📊 Feature Importances - Random Forest")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), feature_names[indices], rotation=45)
plt.tight_layout()
plt.savefig("outputs/feature_importance_rf.png")
plt.show()
