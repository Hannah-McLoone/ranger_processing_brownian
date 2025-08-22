"""
weighted by intensity
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df_loaded = pd.read_csv('features_speed_and_rank.csv')




# If you need X and y back:
X_loaded = df_loaded.iloc[:, :-2].values  # All columns except the last
y_loaded = df_loaded.iloc[:, -2].values   # Last column (speed)
intensity = df_loaded.iloc[:, -1].values

# Train/test split
X_train, X_test, y_train, y_test, intensity_train, intensity_test= train_test_split(X_loaded, y_loaded,intensity, test_size=0.2, random_state=42)

scaler = StandardScaler() # MinMaxScaler for bounded or RobustScaler for heavy outliers
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    'Hist gradient boosting regressor': HistGradientBoostingRegressor(),
    'XGB': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6),
    'extra trees': ExtraTreesRegressor(),
    'LGB': LGBMRegressor()
}

results = []

for name, model in models.items():
    try:
        model.fit(X_train_scaled, y_train, sample_weight=intensity_train)

        y_pred = model.predict(X_test_scaled)

        # Evaluation with sample weights (intensity_test)
        r2 = r2_score(y_test, y_pred, sample_weight=intensity_test)
        mae = mean_absolute_error(y_test, y_pred, sample_weight=intensity_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=intensity_test))

        results.append({
            "Model": name,
            "R² Score": r2,
            "MAE": mae,
            "RMSE": rmse
        })

        print(f"✅ {name}")
        print(f"   R² Score : {r2:.4f}")
        print(f"   MAE      : {mae:.4f}")
        print(f"   RMSE     : {rmse:.4f}")
        print("-" * 40)

    except Exception as e:
        print(f"❌ {name} failed with error: {e}")
        print("-" * 40)

results_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False)
print("\nSummary of all models:")
print(results_df)
