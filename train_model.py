import pandas as pd
import numpy as np
import pickle
import json
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# Step 1: Load the Dataset
# ==========================================
df = pd.read_csv('used_car_price.csv')
print("Dataset loaded successfully! Shape:", df.shape)

# ==========================================
# Step 2: Data Cleaning and Preprocessing
# ==========================================
# 2.1 Fill missing values
df['service_history'] = df['service_history'].fillna(df['service_history'].mode()[0])

# 2.2 FEATURE ENGINEERING: Create 'car_age'
current_year = 2026
df['car_age'] = current_year - df['make_year']

# 2.3 One-Hot Encoding
categorical_cols = ['fuel_type', 'brand', 'transmission', 'color', 'service_history', 'insurance_valid']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Data shape after encoding:", df_encoded.shape)

# ==========================================
# Step 3: Split Data
# ==========================================
X = df_encoded.drop('price_usd', axis=1)
y = df_encoded['price_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")

# ==========================================
# Step 4: Hyperparameter Tuning for Random Forest
# ==========================================
print("\n--- Performing Hyperparameter Tuning for Random Forest ---")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                               n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
best_rf = rf_random.best_estimator_
print("Best Random Forest Parameters:", rf_random.best_params_)

# ==========================================
# Step 5: Train and Compare All Models
# ==========================================
models = {
    "Tuned Random Forest": best_rf,
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression()
}

results = {}
print("\n--- Model Training and Evaluation ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MAE": mae, "R2": r2, "Model": model}
    print(f"\n{name}:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  R² Score: {r2 * 100:.2f}%")

# ==========================================
# Step 6: Select Best Model (excluding Linear Regression)
# ==========================================
best_model_name = max(
    [k for k in results if k != "Linear Regression"],
    key=lambda k: results[k]['R2']
)
best_model = results[best_model_name]['Model']
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {results[best_model_name]['R2'] * 100:.2f}%")

# ==========================================
# Step 7: SHAP Values (Explainable AI)
# ==========================================
print("\n--- Generating SHAP Values ---")
try:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    # Save SHAP values for later use in the app
    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_df.to_csv('shap_values.csv', index=False)
    print("SHAP values saved as 'shap_values.csv'")
except Exception as e:
    print(f"SHAP values could not be generated: {e}")

# ==========================================
# Step 8: Save Best Model and Feature Names
# ==========================================
with open('car_price_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

feature_names = X.columns.tolist()
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f)

# Save feature importances
if hasattr(best_model, 'feature_importances_'):
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    feature_importances.to_csv('feature_importances.csv', index=False)
    print("Feature importances saved!")
else:
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': [0] * len(X.columns)})
    feature_importances.to_csv('feature_importances.csv', index=False)

print("\n✅ All models and artifacts saved successfully!")